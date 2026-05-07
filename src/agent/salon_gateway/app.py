from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys

import httpx
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from salon_gateway.ai.upload_cache import SimulateUploadStore
from salon_gateway.ai.furnishing_compose_prompt import build_furnishing_compose_prompt
from salon_gateway.ai.home_furnishing_prompt import build_home_furnishing_prompt
from salon_gateway.ai.resolve_image import resolve_base_image_for_dashscope
from salon_gateway.ai.wan27_image import Wan27ImageClient
from salon_gateway.ai.wanxiang import WanxiangClient
from salon_gateway.booking.conversation_image_session import ConversationImageStore
from salon_gateway.booking.idempotency import IdempotencyCache
from salon_gateway.booking.session import BookingSessionStore
from salon_gateway.config import SalonGatewaySettings, get_settings
from salon_gateway.furnishing.registry import FurnishingRegistry
from salon_gateway.ingress.wecom import (
    WecomIngress,
    parse_inbound_message,
    parse_sender_recipient,
    render_text_reply,
)
from salon_gateway.models.booking import BookingDraft
from salon_gateway.models.conversation_image import ConversationImageSnap
from salon_gateway.models.furnishing import (
    FurnishingAssetsListResponse,
    FurnishingComposePreviewRequest,
)
from salon_gateway.models.image_preview import ImagePreviewRequest, ImagePreviewResponse
from salon_gateway.models.simulate import SimulateWecomTextIn
from salon_gateway.orchestrator.pipeline import SalonPipeline, default_pipeline
from salon_gateway.sink.feishu import FeishuBitableSink
from salon_gateway.sink.null_sink import LoggingSink

_wecom: WecomIngress | None = None
_pipeline: SalonPipeline | None = None
_sink: FeishuBitableSink | LoggingSink | None = None
_idempotency = IdempotencyCache()
_booking_sessions = BookingSessionStore()
_conversation_images = ConversationImageStore()
_SIMULATE_UPLOADS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "simulate_uploads"
_SIMULATE_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
_CONSULT_SUMMARY_REFRESH_DELAY_SECONDS = 30.0
_consult_summary_tasks: dict[str, asyncio.Task[None]] = {}
_BOOKING_FLUSH_QUEUE_MAXSIZE = 2000
_booking_flush_queue: asyncio.Queue[BookingDraft] = asyncio.Queue(maxsize=_BOOKING_FLUSH_QUEUE_MAXSIZE)
_booking_flush_worker_task: asyncio.Task[None] | None = None
_booking_flush_enqueued_total = 0
_booking_flush_dropped_total = 0
_booking_flush_processed_total = 0
_booking_flush_failed_total = 0
_booking_flush_warn_latched = False

_CN_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_URL_RE = re.compile(r"https?://[^\s)\]>]+", re.IGNORECASE)
_APPOINT_KEYWORDS = ("预约", "到店", "上门", "周", "今天", "明天", "后天", "点", "下午", "上午")


@lru_cache(maxsize=8)
def _furnishing_registry_cached(path_key: str) -> FurnishingRegistry:
    return FurnishingRegistry(Path(path_key))


def _get_wecom(settings: SalonGatewaySettings) -> WecomIngress:
    global _wecom
    if _wecom is None:
        _wecom = WecomIngress(settings)
    return _wecom


def _get_pipeline(settings: SalonGatewaySettings) -> SalonPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = default_pipeline(settings)
    return _pipeline


def _get_sink(settings: SalonGatewaySettings) -> FeishuBitableSink | LoggingSink:
    global _sink
    if _sink is None:
        if (
            settings.feishu_app_id
            and settings.feishu_app_secret
            and settings.feishu_bitable_app_token
            and settings.feishu_bitable_table_id
        ):
            _sink = FeishuBitableSink(settings)
        else:
            _sink = LoggingSink()
    return _sink


def _enqueue_booking_flush(draft: BookingDraft) -> None:
    """Non-blocking enqueue; drop oldest when queue is full to protect chat latency."""
    global _booking_flush_enqueued_total, _booking_flush_dropped_total
    try:
        _booking_flush_queue.put_nowait(draft)
        _booking_flush_enqueued_total += 1
    except asyncio.QueueFull:
        dropped: BookingDraft | None = None
        try:
            dropped = _booking_flush_queue.get_nowait()
            _booking_flush_queue.task_done()
            _booking_flush_dropped_total += 1
        except Exception:
            pass
        try:
            _booking_flush_queue.put_nowait(draft)
            _booking_flush_enqueued_total += 1
            logger.warning(
                "booking flush queue full, dropped oldest draft cid={} then enqueued new one",
                (dropped.conversation_id if dropped else ""),
            )
        except Exception:
            logger.error(
                "booking flush queue full, failed to enqueue draft cid={}",
                (draft.conversation_id or ""),
            )
    _maybe_warn_booking_flush_backlog()


def _booking_flush_queue_stats() -> dict[str, int]:
    return {
        "size": _booking_flush_queue.qsize(),
        "maxsize": _BOOKING_FLUSH_QUEUE_MAXSIZE,
        "enqueued_total": _booking_flush_enqueued_total,
        "dropped_total": _booking_flush_dropped_total,
        "processed_total": _booking_flush_processed_total,
        "failed_total": _booking_flush_failed_total,
    }


def _maybe_warn_booking_flush_backlog() -> None:
    global _booking_flush_warn_latched
    maxsize = int(_booking_flush_queue.maxsize or 0)
    size = int(_booking_flush_queue.qsize())
    if maxsize <= 0:
        return
    ratio = size / maxsize
    settings = get_settings()
    threshold = float(settings.booking_flush_queue_warn_ratio)
    if ratio >= threshold:
        if not _booking_flush_warn_latched:
            logger.warning(
                "booking flush queue backlog high: size={} maxsize={} ratio={:.3f} threshold={:.3f}",
                size,
                maxsize,
                ratio,
                threshold,
            )
            _booking_flush_warn_latched = True
    elif ratio < max(0.0, threshold - 0.1):
        _booking_flush_warn_latched = False


def _prometheus_perf_metrics(stats: dict[str, object]) -> str:
    q = stats.get("booking_flush_queue") if isinstance(stats, dict) else {}
    qd = q if isinstance(q, dict) else {}
    ttfb = stats.get("ttfb_ms") if isinstance(stats, dict) else {}
    elapsed = stats.get("elapsed_ms") if isinstance(stats, dict) else {}
    ttfb_d = ttfb if isinstance(ttfb, dict) else {}
    elapsed_d = elapsed if isinstance(elapsed, dict) else {}
    lines = [
        "# HELP salon_chat_fast_ratio Rolling fast-model hit ratio.",
        "# TYPE salon_chat_fast_ratio gauge",
        f"salon_chat_fast_ratio {float(stats.get('fast_ratio', 0.0) or 0.0):.6f}",
        "# HELP salon_chat_ttfb_ms Time-to-first-byte latency in milliseconds.",
        "# TYPE salon_chat_ttfb_ms gauge",
        f"salon_chat_ttfb_ms{{quantile=\"p50\"}} {float(ttfb_d.get('p50', 0.0) or 0.0):.3f}",
        f"salon_chat_ttfb_ms{{quantile=\"p95\"}} {float(ttfb_d.get('p95', 0.0) or 0.0):.3f}",
        f"salon_chat_ttfb_ms{{quantile=\"max\"}} {float(ttfb_d.get('max', 0.0) or 0.0):.3f}",
        "# HELP salon_chat_elapsed_ms End-to-end latency in milliseconds.",
        "# TYPE salon_chat_elapsed_ms gauge",
        f"salon_chat_elapsed_ms{{quantile=\"p50\"}} {float(elapsed_d.get('p50', 0.0) or 0.0):.3f}",
        f"salon_chat_elapsed_ms{{quantile=\"p95\"}} {float(elapsed_d.get('p95', 0.0) or 0.0):.3f}",
        f"salon_chat_elapsed_ms{{quantile=\"max\"}} {float(elapsed_d.get('max', 0.0) or 0.0):.3f}",
        "# HELP salon_booking_flush_queue_size Pending booking flush jobs.",
        "# TYPE salon_booking_flush_queue_size gauge",
        f"salon_booking_flush_queue_size {int(qd.get('size', 0) or 0)}",
        "# HELP salon_booking_flush_queue_maxsize Booking flush queue capacity.",
        "# TYPE salon_booking_flush_queue_maxsize gauge",
        f"salon_booking_flush_queue_maxsize {int(qd.get('maxsize', 0) or 0)}",
        "# HELP salon_booking_flush_total Booking flush counters.",
        "# TYPE salon_booking_flush_total counter",
        f"salon_booking_flush_total{{status=\"enqueued\"}} {int(qd.get('enqueued_total', 0) or 0)}",
        f"salon_booking_flush_total{{status=\"processed\"}} {int(qd.get('processed_total', 0) or 0)}",
        f"salon_booking_flush_total{{status=\"failed\"}} {int(qd.get('failed_total', 0) or 0)}",
        f"salon_booking_flush_total{{status=\"dropped\"}} {int(qd.get('dropped_total', 0) or 0)}",
    ]
    return "\n".join(lines) + "\n"


async def _booking_flush_worker() -> None:
    global _booking_flush_processed_total, _booking_flush_failed_total
    settings = get_settings()
    while True:
        draft = await _booking_flush_queue.get()
        try:
            sink = _get_sink(settings)
            ok = False
            max_attempts = int(settings.booking_flush_retry_max_attempts)
            base_ms = int(settings.booking_flush_retry_base_delay_ms)
            for attempt in range(max_attempts):
                try:
                    await sink.append_booking(draft)
                    _booking_flush_processed_total += 1
                    ok = True
                    break
                except Exception as e:
                    is_last = attempt >= max_attempts - 1
                    if is_last:
                        raise
                    backoff_s = (base_ms * (2**attempt)) / 1000.0
                    logger.warning(
                        "async booking flush retry cid={} attempt={}/{} backoff_s={:.3f} err={}",
                        (draft.conversation_id or ""),
                        attempt + 1,
                        max_attempts,
                        backoff_s,
                        e,
                    )
                    await asyncio.sleep(backoff_s)
            if not ok:
                _booking_flush_failed_total += 1
        except asyncio.CancelledError:
            _booking_flush_queue.task_done()
            raise
        except Exception as e:
            _booking_flush_failed_total += 1
            logger.warning("async booking flush failed cid={}: {}", (draft.conversation_id or ""), e)
        finally:
            _booking_flush_queue.task_done()


def _detect_ticket_type(text: str, has_assets: bool) -> str:
    t = (text or "").strip()
    if has_assets:
        return "产品咨询"
    if any(k in t for k in ("预约", "到店", "上门")):
        return "预约咨询"
    return "方案咨询"


def _extract_phone(text: str) -> str | None:
    m = _CN_PHONE_RE.search(text or "")
    return m.group(1) if m else None


def _extract_preview_url(reply: str) -> str | None:
    if not reply:
        return None
    for u in _URL_RE.findall(reply):
        if u.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            return u
    urls = _URL_RE.findall(reply)
    return urls[0] if urls else None


def _extract_appointment_time(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    if any(k in t for k in _APPOINT_KEYWORDS):
        return t[:120]
    return None


def _extract_handoff(action: str | None) -> tuple[bool | None, str | None]:
    act = (action or "").strip()
    if act == "handoff_no":
        return False, None
    if act == "handoff_yes":
        return True, "用户点击转人工"
    return None, None


def _is_handoff_action(action: str | None) -> bool:
    act = (action or "").strip()
    return act in ("handoff_yes", "handoff_no")


async def _auto_save_consult_ticket(
    settings: SalonGatewaySettings,
    *,
    from_user: str,
    content: str,
    reply: str,
    channel: str,
    image_url: str | None = None,
    selected_asset_ids: list[str] | None = None,
    action: str | None = None,
    session_phone: str | None = None,
    content_summary: str | None = None,
) -> None:
    text = (content or "").strip()
    assets = [x.strip() for x in (selected_asset_ids or []) if str(x).strip()]
    handoff_requested, handoff_reason = _extract_handoff(action)
    phone = _extract_phone(f"{text} {session_phone or ''}".strip())
    summary_final = (content_summary or "").strip() or None
    draft = BookingDraft(
        conversation_id=(from_user or "").strip(),
        channel=channel,
        external_user_id=(from_user or "").strip() or None,
        wechat_id=(from_user or "").strip() or None,
        phone=phone,
        ticket_type=_detect_ticket_type(text, bool(assets)),
        content_summary=summary_final,
        appointment_time=_extract_appointment_time(text),
        appointment_intent=("有意向" if _extract_appointment_time(text) else "待确认"),
        product_ids=assets or None,
        room_image_url=(image_url or "").strip() or None,
        preview_url=_extract_preview_url(reply),
        ticket_status="沟通中",
        contact_status=("已留资" if phone else "待留资"),
        handoff_requested=handoff_requested,
        handoff_reason=handoff_reason,
        status="pending",
    )
    _enqueue_booking_flush(draft)


async def _upsert_consult_content_summary(
    settings: SalonGatewaySettings,
    *,
    from_user: str,
    channel: str,
    content_summary: str,
    session_phone: str | None = None,
) -> None:
    summary = (content_summary or "").strip()
    if not summary:
        return
    user = (from_user or "").strip()
    phone = _extract_phone(session_phone or "")
    draft = BookingDraft(
        conversation_id=user,
        channel=channel,
        external_user_id=user or None,
        wechat_id=user or None,
        phone=phone,
        content_summary=summary,
        status="pending",
    )
    _enqueue_booking_flush(draft)


async def _refresh_consult_summary_later(
    settings: SalonGatewaySettings,
    *,
    from_user: str,
    channel: str,
) -> None:
    try:
        await asyncio.sleep(_CONSULT_SUMMARY_REFRESH_DELAY_SECONDS)
        pipe = _get_pipeline(settings)
        summary = pipe.consult_content_summary(from_user, "")
        await _upsert_consult_content_summary(
            settings,
            from_user=from_user,
            channel=channel,
            content_summary=summary,
            session_phone=pipe.profile_phone(from_user),
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("background consult summary refresh failed: {}", e)


def _schedule_consult_summary_refresh(
    settings: SalonGatewaySettings,
    *,
    from_user: str,
    channel: str,
) -> None:
    user = (from_user or "").strip()
    if not user:
        return
    old = _consult_summary_tasks.get(user)
    if old and not old.done():
        old.cancel()
    task = asyncio.create_task(
        _refresh_consult_summary_later(
            settings,
            from_user=user,
            channel=channel,
        )
    )
    _consult_summary_tasks[user] = task

    def _drop_done(done: asyncio.Task[None]) -> None:
        if _consult_summary_tasks.get(user) is done:
            _consult_summary_tasks.pop(user, None)

    task.add_done_callback(_drop_done)


def _normalize_secret(s: str) -> str:
    return (s or "").strip().strip("\ufeff").strip()


def _secret_fingerprint(s: str) -> str:
    """Short SHA-256 prefix for logs (compare locally to .env without pasting the secret)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _persist_simulate_upload(upload_file_id: str, data: bytes, mime: str) -> None:
    fid = (upload_file_id or "").strip()
    if not fid:
        return
    (_SIMULATE_UPLOADS_DIR / f"{fid}.bin").write_bytes(data)
    (_SIMULATE_UPLOADS_DIR / f"{fid}.mime").write_text(
        (mime or "image/jpeg").strip() or "image/jpeg",
        encoding="utf-8",
    )


def _load_persisted_simulate_upload(upload_file_id: str) -> tuple[bytes, str] | None:
    fid = (upload_file_id or "").strip()
    if not fid:
        return None
    blob = _SIMULATE_UPLOADS_DIR / f"{fid}.bin"
    if not blob.is_file():
        return None
    mime_path = _SIMULATE_UPLOADS_DIR / f"{fid}.mime"
    mime = "image/jpeg"
    if mime_path.is_file():
        try:
            mime = (mime_path.read_text(encoding="utf-8") or "").strip() or mime
        except Exception:
            pass
    try:
        return blob.read_bytes(), mime
    except Exception:
        return None


def _bearer_or_header(
    authorization: str | None,
    x_salon_token: str | None,
) -> str:
    got = _normalize_secret(x_salon_token or "")
    if got:
        return got
    if not authorization:
        return ""
    raw = _normalize_secret(authorization)
    # Any whitespace after scheme (RFC-style); avoids Tab-only gap breaking partition(" ").
    parts = raw.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return _normalize_secret(parts[1])
    return raw


def _auth_internal(
    settings: SalonGatewaySettings,
    authorization: str | None,
    x_salon_token: str | None,
) -> None:
    allowed = settings.internal_booking_tokens_accepted
    if not allowed:
        raise HTTPException(status_code=404, detail="internal booking disabled")
    got = _bearer_or_header(authorization, x_salon_token)
    if got not in allowed:
        lens = sorted({len(x) for x in allowed})
        afps = sorted({_secret_fingerprint(x) for x in allowed})
        logger.error(
            "internal_booking unauthorized: has_authorization_header={} has_x_salon_token={} parsed_token_len={} parsed_token_sha256_12={} accepted_token_lengths={} accepted_token_sha256_12={}",
            bool(_normalize_secret(authorization or "")),
            bool(_normalize_secret(x_salon_token or "")),
            len(got),
            _secret_fingerprint(got),
            lens,
            afps,
        )
        raise HTTPException(status_code=401, detail="unauthorized")


def _auth_simulate(
    settings: SalonGatewaySettings,
    authorization: str | None,
    x_salon_token: str | None,
) -> None:
    expected = _normalize_secret(settings.simulate_token or "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="simulate disabled: set SALON_SIMULATE_TOKEN in gateway .env",
        )
    if _bearer_or_header(authorization, x_salon_token) != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    del app
    global _booking_flush_worker_task
    settings = get_settings()
    logger.remove()
    logger.add(sys.stderr, level=(settings.log_level or "INFO").upper())
    global _booking_flush_queue, _booking_flush_warn_latched
    queue_max = int(settings.booking_flush_queue_maxsize)
    if queue_max > 0 and queue_max != _booking_flush_queue.maxsize and _booking_flush_queue.qsize() == 0:
        _booking_flush_queue = asyncio.Queue(maxsize=queue_max)
        _booking_flush_warn_latched = False
        logger.info("booking flush queue resized to {}", queue_max)
    _booking_flush_worker_task = asyncio.create_task(_booking_flush_worker())
    try:
        yield
    finally:
        if _booking_flush_worker_task and not _booking_flush_worker_task.done():
            _booking_flush_worker_task.cancel()
            try:
                await _booking_flush_worker_task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="Furnishing gateway (WeCom -> LangGraph -> Feishu)", lifespan=_lifespan)

# 家居素材库本地 JPG → 公网 HTTPS（与反代前缀一致，如 https://quizmesh.tech/salon/furnishing-asset-files/…）
_FURNISHING_IMAGES_DIR = Path(__file__).resolve().parent / "data" / "furnishing_images"
if _FURNISHING_IMAGES_DIR.is_dir():
    _furnishing_dir = str(_FURNISHING_IMAGES_DIR)
    app.mount(
        "/furnishing-asset-files",
        StaticFiles(directory=_furnishing_dir),
        name="furnishing_asset_files",
    )
    # 直连网关（无反代 strip）时与公网路径一致；勿复用同一 StaticFiles 实例挂两处
    app.mount(
        "/salon/furnishing-asset-files",
        StaticFiles(directory=_furnishing_dir),
        name="furnishing_asset_files_salon",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/internal/dashscope-diag")
async def dashscope_diag(
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
) -> dict[str, object]:
    """诊断：检查 DashScope Key 是否有效（提交一个极小的测试任务）。

    鉴权同 /internal/booking（Bearer 或 X-Salon-Token）。
    返回 key_sha256_12 供与控制台 Key 指纹比对。
    """
    settings = get_settings()
    _auth_internal(settings, authorization, x_salon_token)
    key = (settings.dashscope_api_key or "").strip()
    info: dict[str, object] = {
        "dashscope_key_configured": bool(key),
        "dashscope_key_len": len(key),
        "dashscope_key_sha256_12": hashlib.sha256(key.encode()).hexdigest()[:12] if key else "",
        "wanxiang_model": settings.wanxiang_model,
        "dashscope_base": settings.dashscope_base_url,
    }
    if not key:
        return {**info, "status": "error", "detail": "SALON_DASHSCOPE_API_KEY not set"}

    import httpx as _httpx
    try:
        async with _httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                "https://dashscope.aliyuncs.com/api/v1/tasks",
                headers={"Authorization": f"Bearer {key}"},
                params={"page_no": 1, "page_size": 1},
            )
        body_snippet = (r.text or "")[:500]
        return {
            **info,
            "status": "ok" if r.is_success else "error",
            "http_status": r.status_code,
            "body_snippet": body_snippet,
        }
    except Exception as e:
        return {**info, "status": "exception", "detail": str(e)}


@app.get("/webhook/wecom")
async def wecom_verify(
    msg_signature: str = Query(..., alias="msg_signature"),
    timestamp: str = Query(...),
    nonce: str = Query(...),
    echostr: str = Query(...),
) -> PlainTextResponse:
    settings = get_settings()
    try:
        plain = _get_wecom(settings).verify_url(msg_signature, timestamp, nonce, echostr)
    except ValueError:
        raise HTTPException(status_code=403, detail="verify failed") from None
    return PlainTextResponse(content=plain, media_type="text/plain; charset=utf-8")


@app.post("/webhook/wecom")
async def wecom_message(
    request: Request,
    msg_signature: str = Query(..., alias="msg_signature"),
    timestamp: str = Query(...),
    nonce: str = Query(...),
) -> PlainTextResponse:
    settings = get_settings()
    body = await request.body()
    wecom = _get_wecom(settings)
    try:
        inner_xml = wecom.decrypt_body(body, msg_signature, timestamp, nonce)
    except ValueError:
        raise HTTPException(status_code=403, detail="decrypt failed") from None

    msg = parse_inbound_message(inner_xml)
    if msg is None:
        fu, tu = parse_sender_recipient(inner_xml)
        if not fu or not tu:
            return PlainTextResponse(content="success", media_type="text/plain")
        tip = "目前仅支持文字和图片咨询，请直接发送您的问题或照片。"
        reply = render_text_reply(to_user=fu, from_user=tu, content=tip)
        out = wecom.encrypt_reply(reply)
        return PlainTextResponse(content=out, media_type="application/xml; charset=utf-8")

    pipe = _get_pipeline(settings)
    text = await pipe.handle_message(msg)
    raw_content = msg.content if hasattr(msg, "content") else "[图片咨询]"
    await _auto_save_consult_ticket(
        settings,
        from_user=msg.from_user,
        content=raw_content,
        reply=text,
        channel="wecom",
        image_url=(msg.pic_url if hasattr(msg, "pic_url") else None),
        selected_asset_ids=None,
        session_phone=pipe.profile_phone(msg.from_user),
    )
    _schedule_consult_summary_refresh(settings, from_user=msg.from_user, channel="wecom")
    reply = render_text_reply(to_user=msg.from_user, from_user=msg.to_user, content=text)
    out = wecom.encrypt_reply(reply)
    return PlainTextResponse(content=out, media_type="application/xml; charset=utf-8")


@app.post("/internal/booking")
async def internal_booking(
    request: Request,
    draft: BookingDraft,
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
) -> dict[str, bool]:
    settings = get_settings()
    raw_auth = request.headers.get("authorization")
    logger.info(
        "internal_booking: raw_Authorization_present={} fastapi_Header_authorization_present={} X-Salon-Token_present={}",
        raw_auth is not None,
        authorization is not None,
        (x_salon_token or "").strip() != "",
    )
    _auth_internal(settings, authorization, x_salon_token)

    # 存储本轮图片 URL，供后续轮次生图接口跨轮补全（image_url 非空时才更新）
    cid = (draft.conversation_id or "").strip()
    if cid and draft.image_url:
        _conversation_images.save(cid, draft.image_url)

    # Session-based accumulation: merge fields from this turn into the
    # conversation session.  Only write to Feishu the first time all required
    # fields (phone + slot_text + store) are present.
    if cid:
        merged, newly_complete = _booking_sessions.merge_and_check(cid, draft)
        if not newly_complete:
            return {"ok": True, "dedup": False, "complete": False}
        draft = merged
    else:
        # No conversation_id: fall back to legacy single-turn idempotency.
        if not _idempotency.should_process(draft.idempotency_key):
            return {"ok": True, "dedup": True, "complete": True}

    sink = _get_sink(settings)
    try:
        await sink.append_booking(draft)
    except Exception as e:
        logger.exception("append_booking failed: {}", e)
        raise HTTPException(status_code=502, detail="sink failed") from e
    return {"ok": True, "dedup": False, "complete": True}


@app.post("/internal/conversation-image")
async def internal_conversation_image(
    body: ConversationImageSnap,
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
) -> dict[str, bool]:
    """缓存本轮会话的房间参考图 URL，供后续轮次 HTTP 节点 image_url 为空时补全。

    家居 Chatflow 首轮不经过 booking，需单独调用本接口写入 ConversationImageStore。
    鉴权与 POST /internal/booking 相同。
    """
    settings = get_settings()
    _auth_internal(settings, authorization, x_salon_token)
    cid = (body.conversation_id or "").strip()
    url = (body.image_url or "").strip()
    if cid and url:
        _conversation_images.save(cid, url)
    return {"ok": True}


@app.get("/internal/conversation-room-image")
async def internal_conversation_room_image(
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
    conversation_id: str = Query(
        default="",
        max_length=200,
        description="Dify 会话 ID；返回此前 POST /internal/conversation-image 写入的 room URL（无则空）",
    ),
) -> dict[str, object]:
    """供 Dify Code 前一轮拉取「已缓存的空间图 URL」，便于请求体里带上非空的 room_image_url。"""
    settings = get_settings()
    _auth_internal(settings, authorization, x_salon_token)
    cid = (conversation_id or "").strip()
    url = _conversation_images.get(cid) if cid else ""
    return {"image_url": url or ""}


@app.post("/internal/home-furnishing-preview")
async def internal_home_furnishing_preview(
    body: ImagePreviewRequest,
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
) -> ImagePreviewResponse:
    """根据已确认的软装方案，对房间参考图做「效果示意」重绘（通义万相）。

    请求体：image_url、style_prompt（整套方案中文描述）、conversation_id。
    image_url 为空时从会话缓存读取（须先由 POST /internal/conversation-image 或含图的首轮请求写入）。

    推荐使用 wan2.7-image / wan2.7-image-pro；wanx2.1-imageedit 走 description_edit。
    鉴权与 POST /internal/booking 相同；Dify HTTP 节点 read_timeout 建议 ≥ 90s。
    """
    settings = get_settings()
    _auth_internal(settings, authorization, x_salon_token)

    if not settings.dashscope_api_key:
        raise HTTPException(
            status_code=503,
            detail="home furnishing preview disabled: set SALON_DASHSCOPE_API_KEY to enable",
        )
    scheme = (body.style_prompt or "").strip()
    if not scheme:
        raise HTTPException(status_code=400, detail="style_prompt (confirmed scheme) is required")

    logger.info(
        "home_furnishing_preview: conversation_id={} scheme_len={}",
        body.conversation_id or "(none)",
        len(scheme),
    )
    effective_url = _conversation_images.resolve(body.conversation_id, body.image_url)
    if not effective_url:
        raise HTTPException(
            status_code=400,
            detail="image_url is required (no image in current or previous turns of this conversation)",
        )

    try:
        base_image = await resolve_base_image_for_dashscope(effective_url, settings)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    model_id = (settings.wanxiang_model or "").strip()
    use_wan27 = model_id.lower() in ("wan2.7-image", "wan2.7-image-pro")

    try:
        if use_wan27:
            full_prompt = build_home_furnishing_prompt(scheme)
            client27 = Wan27ImageClient(
                settings.dashscope_api_key,
                model_id,
                settings.dashscope_base_url,
            )
            result = await client27.edit_with_prompt(base_image, full_prompt)
        else:
            client = WanxiangClient(
                settings.dashscope_api_key,
                model_id,
                settings.dashscope_base_url,
            )
            result = await client.generate_interior_preview(base_image, scheme)
    except TimeoutError as e:
        logger.warning("home_furnishing_preview: timeout conversation_id={}: {}", body.conversation_id, e)
        raise HTTPException(status_code=504, detail="image generation timed out") from e
    except Exception as e:
        logger.exception("home_furnishing_preview: failed conversation_id={}: {}", body.conversation_id, e)
        raise HTTPException(status_code=502, detail="image generation failed") from e

    logger.info(
        "home_furnishing_preview: done task_id={} preview_url={}",
        result.task_id,
        result.preview_url,
    )
    return ImagePreviewResponse(preview_url=result.preview_url, task_id=result.task_id)


@app.get("/internal/furnishing-assets", response_model=FurnishingAssetsListResponse)
async def internal_furnishing_assets(
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
    q: str = Query(default="", description="名称 / 标签 / id 子串（不区分大小写）"),
    category: str = Query(default="", description="category 精确匹配；空=不限"),
    limit: int = Query(default=20, ge=1, le=100),
) -> FurnishingAssetsListResponse:
    """素材库检索（默认 JSON，可换路径见 SALON_FURNISHING_ASSETS_FILE）。鉴权同 internal/booking。"""
    settings = get_settings()
    _auth_internal(settings, authorization, x_salon_token)
    reg = _furnishing_registry_cached(settings.furnishing_assets_path.as_posix())
    items, total = reg.search(q=q, category=category, limit=limit)
    return FurnishingAssetsListResponse(items=items, total=total)


# 与生产反代路径一致（如 /salon/internal/...）；本地直连网关时也可少依赖 Vite rewrite
app.get("/salon/internal/furnishing-assets", include_in_schema=False)(internal_furnishing_assets)


@app.post("/internal/furnishing-compose-preview")
async def internal_furnishing_compose_preview(
    body: FurnishingComposePreviewRequest,
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
) -> ImagePreviewResponse:
    """空间参考图 + 多张产品参考图 → 万相 2.7 多图编辑效果图。

    图序：第 1 张为空间底图（room_image_url 或会话缓存）；其后为 product_image_urls。
    仅支持 ``wan2.7-image`` / ``wan2.7-image-pro``；read_timeout 建议 ≥ 90s。
    """
    settings = get_settings()
    _auth_internal(settings, authorization, x_salon_token)

    if not settings.dashscope_api_key:
        raise HTTPException(
            status_code=503,
            detail="compose preview disabled: set SALON_DASHSCOPE_API_KEY to enable",
        )
    model_id = (settings.wanxiang_model or "").strip()
    use_wan27 = model_id.lower() in ("wan2.7-image", "wan2.7-image-pro")
    if not use_wan27:
        raise HTTPException(
            status_code=400,
            detail="furnishing compose requires SALON_WANXIANG_MODEL=wan2.7-image or wan2.7-image-pro",
        )

    room_effective = _conversation_images.resolve(body.conversation_id, body.room_image_url)
    if not room_effective:
        raise HTTPException(
            status_code=400,
            detail=(
                "room_image_url is required (or use POST /internal/conversation-image first "
                "with the same conversation_id)"
            ),
        )

    all_urls = [room_effective, *body.product_image_urls]
    logger.info(
        "furnishing_compose_preview: conversation_id={} n_images={}",
        body.conversation_id or "(none)",
        len(all_urls),
    )
    try:
        refs = await asyncio.gather(
            *[resolve_base_image_for_dashscope(u, settings) for u in all_urls]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    prompt = build_furnishing_compose_prompt(
        n_product_images=len(body.product_image_urls),
        placement_hint=body.placement_hint,
        style_notes=body.style_notes,
    )
    client27 = Wan27ImageClient(
        settings.dashscope_api_key,
        model_id,
        settings.dashscope_base_url,
    )
    try:
        result = await client27.edit_with_images(list(refs), prompt)
    except TimeoutError as e:
        logger.warning("furnishing_compose_preview: timeout: {}", e)
        raise HTTPException(status_code=504, detail="image generation timed out") from e
    except Exception as e:
        logger.exception("furnishing_compose_preview: failed: {}", e)
        raise HTTPException(status_code=502, detail="image generation failed") from e

    logger.info(
        "furnishing_compose_preview: done task_id={} preview_url={}",
        result.task_id,
        result.preview_url,
    )
    return ImagePreviewResponse(preview_url=result.preview_url, task_id=result.task_id)


@app.get("/internal/booking-options")
async def internal_booking_options(
    request: Request,
    store_q: str = Query(default="", description="门店单选：按名称子串过滤（不区分大小写）"),
    service_q: str = Query(default="", description="项目多选：按名称子串过滤"),
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
) -> dict[str, object]:
    """飞书多维表中「门店」单选、「项目」多选的可选值，供前端/Dify 做下拉与搜索。

    列名来自 SALON_FEISHU_FIELD_MAP_JSON 的 store / service 键对应飞书列名。
    鉴权与 POST /internal/booking 相同（Bearer 或 X-Salon-Token）。
    """
    settings = get_settings()
    logger.info(
        "internal_booking_options: raw_Authorization_present={}",
        request.headers.get("authorization") is not None,
    )
    _auth_internal(settings, authorization, x_salon_token)
    sink = _get_sink(settings)
    if not isinstance(sink, FeishuBitableSink):
        raise HTTPException(status_code=404, detail="feishu not configured")
    try:
        return await sink.booking_field_options(store_search=store_q, service_search=service_q)
    except Exception as e:
        logger.exception("booking_field_options failed: {}", e)
        raise HTTPException(status_code=502, detail="feishu fields failed") from e


@app.get("/internal/perf-stats")
async def internal_perf_stats(
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
) -> dict[str, object]:
    """聚合对话性能：TTFB / 总耗时 / 快模型命中率（基于进程内最近样本）。"""
    settings = get_settings()
    _auth_internal(settings, authorization, x_salon_token)
    pipe = _get_pipeline(settings)
    return {
        **pipe.perf_stats(),
        "booking_flush_queue": _booking_flush_queue_stats(),
    }


@app.post("/internal/perf-stats/reset")
async def internal_perf_stats_reset(
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
) -> dict[str, object]:
    """重置进程内性能样本，便于对比调优前后数据窗口。"""
    settings = get_settings()
    _auth_internal(settings, authorization, x_salon_token)
    pipe = _get_pipeline(settings)
    out = pipe.reset_perf_stats()
    return {**out, "booking_flush_queue": _booking_flush_queue_stats()}


@app.get("/internal/perf-stats/prometheus")
async def internal_perf_stats_prometheus(
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
) -> Response:
    """Prometheus 文本指标（鉴权同 /internal/booking）。"""
    settings = get_settings()
    _auth_internal(settings, authorization, x_salon_token)
    pipe = _get_pipeline(settings)
    stats: dict[str, object] = {
        **pipe.perf_stats(),
        "booking_flush_queue": _booking_flush_queue_stats(),
    }
    return Response(
        content=_prometheus_perf_metrics(stats),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/internal/feishu/ensure-fields")
async def internal_feishu_ensure_fields(
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
) -> dict[str, object]:
    """根据 SALON_FEISHU_FIELD_MAP_JSON（+ 可选 SALON_FEISHU_FIELD_TYPES_JSON）自动补齐缺失列。"""
    settings = get_settings()
    _auth_internal(settings, authorization, x_salon_token)
    sink = _get_sink(settings)
    if not isinstance(sink, FeishuBitableSink):
        raise HTTPException(status_code=404, detail="feishu not configured")
    try:
        return await sink.ensure_fields_from_map()
    except Exception as e:
        logger.exception("ensure_fields_from_map failed: {}", e)
        raise HTTPException(status_code=502, detail="feishu ensure-fields failed") from e


@app.post("/simulate/wecom-text")
async def simulate_wecom_text(
    body: SimulateWecomTextIn,
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
) -> dict[str, object]:
    """Same pipeline as WeCom → LangGraph; supports optional image (image_url / upload_file_id)."""
    settings = get_settings()
    _auth_simulate(settings, authorization, x_salon_token)
    pipe = _get_pipeline(settings)
    if _is_handoff_action(body.action):
        reply = pipe.handoff_simulate_ack(body.from_user.strip(), body.action)
        fu = body.from_user.strip()
        await _auto_save_consult_ticket(
            settings,
            from_user=fu,
            content=body.content,
            reply=reply,
            channel="simulate",
            image_url=body.image_url,
            selected_asset_ids=body.selected_asset_ids,
            action=body.action,
            session_phone=pipe.profile_phone(fu),
        )
        _schedule_consult_summary_refresh(settings, from_user=fu, channel="simulate")
        return {"reply": reply, **pipe.ui_snapshot(body.from_user.strip())}
    reply = await pipe.handle_with_image(
        body.from_user.strip(),
        body.content,
        image_url=body.image_url,
        upload_file_id=body.upload_file_id,
        selected_asset_ids=body.selected_asset_ids,
        action=body.action,
    )
    fu = body.from_user.strip()
    await _auto_save_consult_ticket(
        settings,
        from_user=fu,
        content=body.content,
        reply=reply,
        channel="simulate",
        image_url=body.image_url,
        selected_asset_ids=body.selected_asset_ids,
        action=body.action,
        session_phone=pipe.profile_phone(fu),
    )
    _schedule_consult_summary_refresh(settings, from_user=fu, channel="simulate")
    return {"reply": reply, **pipe.ui_snapshot(body.from_user.strip())}


@app.post("/simulate/wecom-text-stream")
async def simulate_wecom_text_stream(
    body: SimulateWecomTextIn,
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
) -> StreamingResponse:
    """与 ``/simulate/wecom-text`` 相同入参；返回 ``text/event-stream``（与前端兼容的 JSON 行）。"""
    settings = get_settings()
    _auth_simulate(settings, authorization, x_salon_token)
    pipe = _get_pipeline(settings)

    async def gen():
        chunks: list[str] = []
        if _is_handoff_action(body.action):
            reply = pipe.handoff_simulate_ack(body.from_user.strip(), body.action)
            fu = body.from_user.strip()
            await _auto_save_consult_ticket(
                settings,
                from_user=fu,
                content=body.content,
                reply=reply,
                channel="simulate-stream",
                image_url=body.image_url,
                selected_asset_ids=body.selected_asset_ids,
                action=body.action,
                session_phone=pipe.profile_phone(fu),
            )
            _schedule_consult_summary_refresh(settings, from_user=fu, channel="simulate-stream")
            msg = json.dumps({"event": "message", "answer": reply}, ensure_ascii=False)
            yield f"data: {msg}\n\n".encode("utf-8")
            snap = pipe.ui_snapshot(body.from_user.strip())
            actions = snap.get("actions") or []
            if actions:
                obj = json.dumps({"event": "ui_actions", "actions": actions}, ensure_ascii=False)
                yield f"data: {obj}\n\n".encode("utf-8")
            yield b"data: {\"event\":\"message_end\"}\n\n"
            return
        try:
            async for chunk in pipe.handle_with_image_stream(
                body.from_user.strip(),
                body.content,
                image_url=body.image_url,
                upload_file_id=body.upload_file_id,
                selected_asset_ids=body.selected_asset_ids,
                action=body.action,
            ):
                try:
                    s = chunk.decode("utf-8", errors="ignore")
                    for line in s.splitlines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        obj = json.loads(payload)
                        if obj.get("event") == "message" and isinstance(obj.get("answer"), str):
                            chunks.append(obj["answer"])
                except Exception:
                    pass
                yield chunk
            if chunks:
                fu = body.from_user.strip()
                await _auto_save_consult_ticket(
                    settings,
                    from_user=fu,
                    content=body.content,
                    reply="".join(chunks),
                    channel="simulate-stream",
                    image_url=body.image_url,
                    selected_asset_ids=body.selected_asset_ids,
                    action=body.action,
                    session_phone=pipe.profile_phone(fu),
                )
                _schedule_consult_summary_refresh(settings, from_user=fu, channel="simulate-stream")
        except httpx.HTTPStatusError as e:
            logger.warning("simulate stream upstream HTTP {}", e.response.status_code)
            err = json.dumps(
                {"event": "error", "message": f"上游 HTTP {e.response.status_code}"},
                ensure_ascii=False,
            )
            yield f"data: {err}\n\n".encode("utf-8")
        except Exception as e:
            logger.exception("simulate stream failed: {}", e)
            err = json.dumps({"event": "error", "message": "对话流式失败"}, ensure_ascii=False)
            yield f"data: {err}\n\n".encode("utf-8")

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# 与生产反代路径一致（如 /salon/...）；本地直连网关时也可少依赖 Vite rewrite
app.post("/salon/simulate/wecom-text", include_in_schema=False)(simulate_wecom_text)
app.post("/salon/simulate/wecom-text-stream", include_in_schema=False)(simulate_wecom_text_stream)


@app.post("/simulate/upload-image")
async def simulate_upload_image(
    file: Annotated[UploadFile, File(description="Image file cached for simulate / LangGraph vision")],
    from_user: str = Query(default="sim-user-1", description="Must match from_user in subsequent simulate call"),
    authorization: Annotated[str | None, Header()] = None,
    x_salon_token: Annotated[str | None, Header(alias="X-Salon-Token")] = None,
) -> dict[str, str]:
    """缓存图片到进程内存储，返回 upload_file_id 供后续 ``/simulate/wecom-text`` 使用。"""
    settings = get_settings()
    _auth_simulate(settings, authorization, x_salon_token)
    prefix = (settings.dify_user_prefix or "wecom").strip()
    chat_user = f"{prefix}:{from_user.strip()}"
    content = await file.read()
    mime = file.content_type or "image/jpeg"
    fname = file.filename or "image.jpg"
    try:
        fid = SimulateUploadStore.instance().put(content, mime)
        _persist_simulate_upload(fid, content, mime)
    except Exception as e:
        logger.exception("simulate upload cache failed: {}", e)
        raise HTTPException(status_code=502, detail="upload cache failed") from e
    return {
        "upload_file_id": fid,
        "filename": fname,
        "dify_user": chat_user,
        # Keep /salon prefix so frontend dev proxy can forward it.
        "preview_url": f"/salon/simulate/upload-image/{fid}?token={settings.simulate_token}",
    }


@app.get("/simulate/upload-image/{upload_file_id}")
async def simulate_get_upload_image(
    upload_file_id: str,
    token: str = Query(default="", description="SALON_SIMULATE_TOKEN"),
) -> Response:
    settings = get_settings()
    expected = _normalize_secret(settings.simulate_token or "")
    if not expected or _normalize_secret(token) != expected:
        raise HTTPException(status_code=401, detail="unauthorized")
    hit = SimulateUploadStore.instance().get(upload_file_id.strip())
    if hit is not None:
        mime = (hit.mime or "image/jpeg").split(";", 1)[0].strip() or "image/jpeg"
        return Response(content=hit.data, media_type=mime)
    persisted = _load_persisted_simulate_upload(upload_file_id)
    if not persisted:
        raise HTTPException(status_code=404, detail="upload image not found")
    data, mime = persisted
    return Response(content=data, media_type=(mime or "image/jpeg"))


app.post("/salon/simulate/upload-image", include_in_schema=False)(simulate_upload_image)
app.get("/salon/simulate/upload-image/{upload_file_id}", include_in_schema=False)(simulate_get_upload_image)

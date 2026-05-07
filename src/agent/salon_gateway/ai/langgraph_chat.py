"""LangGraph 对话：替代 Dify chat-messages，会话用 MemorySaver + thread_id。"""

from __future__ import annotations

import base64
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from collections import deque
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger

from salon_gateway.ai.resolve_image import resolve_base_image_for_dashscope
from salon_gateway.ai.upload_cache import SimulateUploadStore
from salon_gateway.config import SalonGatewaySettings

try:
    from tools.llm import get_llm
except ImportError:  # pragma: no cover
    get_llm = None  # type: ignore[assignment]


FURNISHING_SYSTEM_PROMPT = """你是家居软装客服助理（企业微信场景）。

你的目标按优先级执行：
1) 先理解用户需求并提供简洁可执行的家居建议；
2) 只在用户出现预约、报价、购买、到店、转人工等明确意向时，再收集手机号与是否转人工；
3) 若用户有预约意向，主动确认时间与门店。

行为要求：
- 不要在第一次打招呼时就索要手机号或询问转人工；
- 用户只是打招呼时，先简短介绍能力，并引导其上传实景图或说明空间需求；
- 若手机号未明确给出，且用户出现预约/报价/购买/到店/联系意向时，再礼貌追问手机号；
- 若用户提到转人工、复杂报价、预约落地，再询问是否需要转人工；
- 如果用户问“什么信息/为什么要这些信息”，用1-2句解释用途并简短列出：手机号、是否转人工、是否预约；
- 即使追问，也要附上 1-3 句有价值的方案建议；
- 当用户只发图片时，先描述观察，再追问关键缺失信息；
- 推荐时优先基于“实景图 + 已选素材”给出落地建议，避免泛泛而谈；
- 若缺少实景图，明确提示“请先上传实景图”；
- 若缺少素材选择，明确提示“请先选择素材（可多选）”；
- 当需要推荐素材时，请给出 2-4 个具体素材建议（可用“素材ID/名称”表达）；
- 并明确告知：也可以选择推荐列表之外的其他素材；
- 若你没有拿到“素材清单”，不要输出具体素材ID或虚构名称，只能提示用户先选素材；
- 如果拿到了素材清单，只能从清单内引用素材ID/名称，严禁编造不存在的素材；
- 如果素材清单中没有某个品类，不要推荐该品类，只能说“当前素材库暂无该类素材”；
- 用户说“生成效果图”但未选择素材ID时，不能说已开始生成，必须先要求选择素材ID；
- 当实景图与素材都齐全时，明确提示可“生成效果图”并引导用户确认生成；
- 全程使用中文、简洁、客服口吻。
"""

_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_ASSET_ID_RE = re.compile(r"\bdemo-[a-z0-9-]+\b", re.IGNORECASE)
_VISION_HINTS = ("这张图", "图片", "看图", "图里", "空间", "配色", "摆放", "户型", "实景", "客厅", "卧室")
_TEXT_ONLY_HINTS = ("手机号", "电话", "预约", "转人工", "联系", "报价", "多少钱", "在吗", "你好", "谢谢")


class _FurnishingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _message_text(msg: BaseMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
        return "".join(parts)
    return str(c)


def _sse_line(obj: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


@dataclass(slots=True)
class _StructuredMemory:
    phone: str = ""
    handoff_requested: bool | None = None
    has_room_image: bool = False
    selected_asset_ids: list[str] = field(default_factory=list)

    def merge_inputs(self, inputs: dict[str, Any] | None) -> None:
        if not isinstance(inputs, dict):
            return
        phone = str(inputs.get("phone") or "").strip()
        if phone:
            self.phone = phone
        handoff = inputs.get("handoff_requested")
        if isinstance(handoff, bool):
            self.handoff_requested = handoff
        if bool(inputs.get("has_room_image")):
            self.has_room_image = True
        ids = inputs.get("selected_asset_ids")
        if isinstance(ids, list):
            self._merge_assets([str(x).strip() for x in ids if str(x).strip()])

    def observe_text(self, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        m = _PHONE_RE.search(t)
        if m:
            self.phone = m.group(1)
        lowered = t.lower()
        if "转人工" in t or "handoff_yes" in lowered:
            self.handoff_requested = True
        if "暂不转人工" in t or "handoff_no" in lowered:
            self.handoff_requested = False
        self._merge_assets(_ASSET_ID_RE.findall(t))

    def _merge_assets(self, ids: list[str]) -> None:
        for aid in ids:
            if aid not in self.selected_asset_ids:
                self.selected_asset_ids.append(aid)
        if len(self.selected_asset_ids) > 8:
            self.selected_asset_ids = self.selected_asset_ids[-8:]

    def prompt_block(self) -> str:
        parts: list[str] = []
        if self.phone:
            parts.append(f"- 已留资手机号: {self.phone}")
        if self.handoff_requested is True:
            parts.append("- 转人工意向: 是")
        elif self.handoff_requested is False:
            parts.append("- 转人工意向: 否")
        if self.has_room_image:
            parts.append("- 已提供实景图: 是")
        if self.selected_asset_ids:
            parts.append(f"- 已选素材ID: {', '.join(self.selected_asset_ids[:8])}")
        if not parts:
            return ""
        return "会话结构化记忆（仅供参考，若与用户本轮冲突以本轮为准）:\n" + "\n".join(parts)


@dataclass(slots=True)
class _ThreadMemory:
    summary: str = ""
    structured: _StructuredMemory = field(default_factory=_StructuredMemory)


class _MemoryManager:
    def __init__(
        self,
        recent_messages: int,
        summary_max_chars: int,
        recent_messages_short: int,
        keep_image_turns: int,
    ) -> None:
        self._recent_messages = max(4, recent_messages)
        self._recent_messages_short = max(4, min(recent_messages_short, self._recent_messages))
        self._summary_max_chars = max(200, summary_max_chars)
        self._keep_image_turns = max(0, keep_image_turns)
        self._threads: dict[str, _ThreadMemory] = {}

    def _thread(self, thread_id: str) -> _ThreadMemory:
        t = self._threads.get(thread_id)
        if t is None:
            t = _ThreadMemory()
            self._threads[thread_id] = t
        return t

    def build_messages(
        self,
        *,
        thread_id: str,
        prior: list[BaseMessage],
        human: HumanMessage,
        inputs: dict[str, Any] | None,
        compact: bool = False,
        include_image_history: bool = True,
    ) -> list[BaseMessage]:
        mem = self._thread(thread_id)
        mem.structured.merge_inputs(inputs)
        hist = [*prior, human]
        hist = self._normalize_image_history(hist, include_image_history=include_image_history)
        keep_n = self._recent_messages_short if compact else self._recent_messages
        dropped = hist[:-keep_n] if len(hist) > keep_n else []
        recent = hist[-keep_n:]
        if dropped:
            snippet = self._compress_messages(dropped)
            if snippet:
                mem.summary = self._merge_summary(mem.summary, snippet)
        memory_blocks: list[str] = []
        if mem.summary:
            memory_blocks.append(f"会话历史摘要:\n{mem.summary}")
        sm = mem.structured.prompt_block()
        if sm:
            memory_blocks.append(sm)
        memory_prompt = "\n\n".join(memory_blocks)
        sys = (
            f"{FURNISHING_SYSTEM_PROMPT}\n\n{memory_prompt}"
            if memory_prompt
            else FURNISHING_SYSTEM_PROMPT
        )
        return [SystemMessage(sys), *recent]

    @staticmethod
    def _human_has_image(msg: BaseMessage) -> bool:
        if not isinstance(msg, HumanMessage):
            return False
        c = msg.content
        if not isinstance(c, list):
            return False
        for p in c:
            if isinstance(p, dict) and p.get("type") == "image_url":
                return True
        return False

    @staticmethod
    def _strip_human_images(msg: BaseMessage) -> BaseMessage:
        if not isinstance(msg, HumanMessage):
            return msg
        c = msg.content
        if not isinstance(c, list):
            return msg
        text_parts: list[str] = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                text_parts.append(str(p.get("text", "")))
        merged = "".join(text_parts).strip() or "[图片消息]"
        return HumanMessage(content=merged)

    def _normalize_image_history(self, hist: list[BaseMessage], *, include_image_history: bool) -> list[BaseMessage]:
        if not hist:
            return hist
        if include_image_history:
            if self._keep_image_turns <= 0:
                return [self._strip_human_images(m) if self._human_has_image(m) else m for m in hist]
            image_idx = [i for i, m in enumerate(hist) if self._human_has_image(m)]
            keep_idx = set(image_idx[-self._keep_image_turns :])
            out: list[BaseMessage] = []
            for i, m in enumerate(hist):
                if i in keep_idx:
                    out.append(m)
                elif self._human_has_image(m):
                    out.append(self._strip_human_images(m))
                else:
                    out.append(m)
            return out
        return [self._strip_human_images(m) if self._human_has_image(m) else m for m in hist]

    def observe_turn(
        self,
        *,
        thread_id: str,
        user_text: str,
        assistant_text: str,
        had_image: bool,
        inputs: dict[str, Any] | None,
    ) -> None:
        mem = self._thread(thread_id)
        mem.structured.merge_inputs(inputs)
        mem.structured.observe_text(user_text)
        mem.structured.observe_text(assistant_text)
        if had_image:
            mem.structured.has_room_image = True

    def _compress_messages(self, messages: list[BaseMessage]) -> str:
        rows: list[str] = []
        for m in messages[-12:]:
            t = _message_text(m).strip()
            if not t:
                continue
            role = "用户" if isinstance(m, HumanMessage) else "助手" if isinstance(m, AIMessage) else "系统"
            rows.append(f"{role}: {t[:140]}")
        return "；".join(rows)[:700]

    def _merge_summary(self, existing: str, incoming: str) -> str:
        merged = f"{existing}\n{incoming}".strip() if existing else incoming
        return merged[-self._summary_max_chars :]


class LangGraphChatClient:
    """与旧 Dify 客户端对齐：``complete`` / ``stream_complete`` + 会话 id（thread_id）。"""

    def __init__(self, settings: SalonGatewaySettings) -> None:
        self._settings = settings
        self._memory = _MemoryManager(
            recent_messages=settings.memory_recent_messages,
            summary_max_chars=settings.memory_summary_max_chars,
            recent_messages_short=settings.memory_recent_messages_short,
            keep_image_turns=settings.memory_keep_image_turns,
        )
        self._checkpointer = MemorySaver()
        self._app = self._build_graph().compile(checkpointer=self._checkpointer)
        self._perf_turn_total = 0
        self._perf_turn_fast = 0
        self._perf_samples: deque[dict[str, Any]] = deque(maxlen=500)

    @property
    def upload_store(self) -> SimulateUploadStore:
        return SimulateUploadStore.instance()

    def _build_graph(self) -> StateGraph:
        settings = self._settings

        async def agent(state: _FurnishingState) -> dict[str, list[AIMessage]]:
            if get_llm is None:
                raise RuntimeError("tools.llm 不可用：请从 src/agent 目录设置 PYTHONPATH 后启动网关")
            llm = get_llm(
                model_name=(settings.llm_model or "").strip() or None,
                temperature=settings.llm_temperature,
            )
            msgs: list[BaseMessage] = [SystemMessage(FURNISHING_SYSTEM_PROMPT), *state["messages"]]
            resp = await llm.ainvoke(msgs)
            if not isinstance(resp, AIMessage):
                resp = AIMessage(content=getattr(resp, "content", str(resp)))
            return {"messages": [resp]}

        g: StateGraph = StateGraph(_FurnishingState)
        g.add_node("agent", agent)
        g.add_edge(START, "agent")
        g.add_edge("agent", END)
        return g

    async def _human_from_query(
        self,
        query: str,
        files: list[dict[str, Any]] | None,
    ) -> HumanMessage:
        if not files:
            return HumanMessage(content=query)
        parts: list[dict[str, Any]] = [{"type": "text", "text": query}]
        for f in files:
            if (f.get("type") or "") != "image":
                continue
            url = await self._resolve_image_part(f)
            parts.append({"type": "image_url", "image_url": {"url": url}})
        return HumanMessage(content=parts)

    async def _resolve_image_part(self, f: dict[str, Any]) -> str:
        method = f.get("transfer_method") or ""
        if method == "remote_url":
            raw = (f.get("url") or "").strip()
            if not raw:
                raise ValueError("remote_url 缺少 url")
            return await resolve_base_image_for_dashscope(raw, self._settings)
        if method == "local_file":
            uid = (f.get("upload_file_id") or "").strip()
            hit = SimulateUploadStore.instance().get(uid)
            if hit is None:
                raise ValueError("upload_file_id 无效或已过期，请重新上传图片")
            b64 = base64.standard_b64encode(hit.data).decode("ascii")
            mime = hit.mime.split(";")[0].strip() or "image/jpeg"
            return f"data:{mime};base64,{b64}"
        raise ValueError(f"不支持的图片 transfer_method: {method!r}")

    def _is_compact_turn(self, query: str, files: list[dict[str, Any]] | None) -> bool:
        q = (query or "").strip()
        if not q:
            return True
        if files:
            return False
        if len(q) > self._settings.llm_fast_query_chars:
            return False
        if _ASSET_ID_RE.search(q):
            return False
        if any(k in q for k in ("效果图", "生成", "预算", "风格", "搭配", "方案", "转人工")):
            return False
        return True

    def _pick_model_name(self, query: str, files: list[dict[str, Any]] | None) -> str | None:
        base = (self._settings.llm_model or "").strip() or None
        vision = (self._settings.llm_vision_model or "").strip() or "qwen/qwen-vl-plus"
        fast = (self._settings.llm_fast_model or "").strip() or None
        if files:
            return vision
        if fast and self._is_compact_turn(query, files):
            return fast
        return base

    @staticmethod
    def _routing_query(query: str, inputs: dict[str, Any] | None) -> str:
        if isinstance(inputs, dict):
            raw = str(inputs.get("__raw_query") or "").strip()
            if raw:
                return raw
        return (query or "").strip()

    def _route_intent(self, query: str, files: list[dict[str, Any]] | None) -> tuple[str, bool]:
        """Return (intent, include_image_history)."""
        if not self._settings.intent_router_enabled:
            return ("router_disabled", True)
        if files:
            return ("vision_required", True)
        q = (query or "").strip()
        if not q:
            return ("text_only", False)
        if any(k in q for k in _VISION_HINTS):
            return ("vision_related", True)
        if any(k in q for k in _TEXT_ONLY_HINTS):
            return ("text_only", False)
        return ("mixed", False)

    def _observe_model_pick(self, model_name: str | None) -> float:
        self._perf_turn_total += 1
        fast_name = (self._settings.llm_fast_model or "").strip()
        chosen = (model_name or "").strip()
        is_fast = bool(fast_name and chosen and chosen == fast_name)
        if is_fast:
            self._perf_turn_fast += 1
        if self._perf_turn_total <= 0:
            return 0.0
        return self._perf_turn_fast / self._perf_turn_total

    @staticmethod
    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        if p <= 0:
            return float(min(values))
        if p >= 1:
            return float(max(values))
        vals = sorted(values)
        idx = int(round((len(vals) - 1) * p))
        idx = max(0, min(idx, len(vals) - 1))
        return float(vals[idx])

    def _record_perf_sample(
        self,
        *,
        mode: str,
        model_name: str | None,
        compact: bool,
        has_files: bool,
        elapsed_ms: float,
        ttfb_ms: float,
        out_chars: int,
    ) -> None:
        self._perf_samples.append(
            {
                "mode": mode,
                "model": (model_name or "").strip() or "(default)",
                "compact": bool(compact),
                "has_files": bool(has_files),
                "elapsed_ms": float(elapsed_ms),
                "ttfb_ms": float(ttfb_ms),
                "out_chars": int(out_chars),
            }
        )

    def perf_stats(self) -> dict[str, Any]:
        samples = list(self._perf_samples)
        ttfb_vals = [float(x.get("ttfb_ms") or 0.0) for x in samples]
        elapsed_vals = [float(x.get("elapsed_ms") or 0.0) for x in samples]
        stream_samples = [x for x in samples if x.get("mode") == "stream"]
        complete_samples = [x for x in samples if x.get("mode") == "complete"]
        compact_count = sum(1 for x in samples if bool(x.get("compact")))
        with_files_count = sum(1 for x in samples if bool(x.get("has_files")))
        return {
            "total_turns": self._perf_turn_total,
            "fast_turns": self._perf_turn_fast,
            "fast_ratio": (self._perf_turn_fast / self._perf_turn_total) if self._perf_turn_total else 0.0,
            "sample_size": len(samples),
            "sample_window_max": self._perf_samples.maxlen or len(samples),
            "mode_counts": {"stream": len(stream_samples), "complete": len(complete_samples)},
            "compact_ratio": (compact_count / len(samples)) if samples else 0.0,
            "with_files_ratio": (with_files_count / len(samples)) if samples else 0.0,
            "ttfb_ms": {
                "p50": self._percentile(ttfb_vals, 0.50),
                "p95": self._percentile(ttfb_vals, 0.95),
                "max": self._percentile(ttfb_vals, 1.0),
            },
            "elapsed_ms": {
                "p50": self._percentile(elapsed_vals, 0.50),
                "p95": self._percentile(elapsed_vals, 0.95),
                "max": self._percentile(elapsed_vals, 1.0),
            },
            "recent_samples": samples[-20:],
        }

    def reset_perf_stats(self) -> dict[str, Any]:
        removed = len(self._perf_samples)
        self._perf_samples.clear()
        self._perf_turn_total = 0
        self._perf_turn_fast = 0
        return {"ok": True, "cleared_samples": removed}

    async def complete(
        self,
        *,
        user: str,
        query: str,
        conversation_id: str | None,
        files: list[dict[str, Any]] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> tuple[str, str | None]:
        del user
        if get_llm is None:
            return ("服务未配置 LLM（缺少 tools.llm / 环境变量）。", None)
        thread_id = (conversation_id or "").strip() or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self._app.aget_state(config)
        prior: list[BaseMessage] = []
        if snapshot and snapshot.values:
            prior = list(snapshot.values.get("messages") or [])
        routing_query = self._routing_query(query, inputs)
        compact = self._is_compact_turn(routing_query, files)
        intent, include_image_history = self._route_intent(routing_query, files)
        if intent == "text_only":
            compact = True
        model_name = self._pick_model_name(routing_query, files)
        start = time.perf_counter()
        try:
            human = await self._human_from_query(query, files)
            invoke_msgs = self._memory.build_messages(
                thread_id=thread_id,
                prior=prior,
                human=human,
                inputs=inputs,
                compact=compact,
                include_image_history=include_image_history,
            )
            llm = get_llm(
                model_name=model_name,
                temperature=self._settings.llm_temperature,
            )
            out = await llm.ainvoke(invoke_msgs)
        except httpx.HTTPStatusError as e:
            logger.error("langgraph upstream HTTP {} {}", e.response.status_code, e.response.text[:500])
            return ("抱歉，系统暂时繁忙，请稍后再试。", conversation_id)
        except Exception as e:
            logger.exception("langgraph ainvoke failed: {}", e)
            return ("抱歉，系统暂时繁忙，请稍后再试。", conversation_id)

        answer_text = _message_text(out).strip() if isinstance(out, BaseMessage) else str(out).strip()
        answer_text = answer_text or "（无回复）"
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        fast_ratio = self._observe_model_pick(model_name)
        logger.info(
            "chat_perf mode=complete model={} intent={} compact={} include_image_history={} has_files={} prior_msgs={} invoke_msgs={} elapsed_ms={:.1f} ttfb_ms={:.1f} fast_ratio={:.3f}",
            model_name or "(default)",
            intent,
            compact,
            include_image_history,
            bool(files),
            len(prior),
            len(invoke_msgs),
            elapsed_ms,
            elapsed_ms,
            fast_ratio,
        )
        self._record_perf_sample(
            mode="complete",
            model_name=model_name,
            compact=compact,
            has_files=bool(files),
            elapsed_ms=elapsed_ms,
            ttfb_ms=elapsed_ms,
            out_chars=len(answer_text),
        )
        await self._app.aupdate_state(
            config,
            {"messages": [human, AIMessage(content=answer_text)]},
            as_node="agent",
        )
        self._memory.observe_turn(
            thread_id=thread_id,
            user_text=query,
            assistant_text=answer_text,
            had_image=bool(files),
            inputs=inputs,
        )
        return (answer_text, thread_id)

    async def stream_complete(
        self,
        *,
        user: str,
        query: str,
        conversation_id: str | None,
        files: list[dict[str, Any]] | None = None,
        inputs: dict[str, Any] | None = None,
        conversation_id_holder: list[str | None],
    ) -> AsyncIterator[bytes]:
        del user
        conversation_id_holder.clear()
        conversation_id_holder.append(None)
        if get_llm is None:
            err = json.dumps({"event": "error", "message": "服务未配置 LLM。"}, ensure_ascii=False)
            yield f"data: {err}\n\n".encode("utf-8")
            return

        thread_id = (conversation_id or "").strip() or str(uuid.uuid4())
        conversation_id_holder[0] = thread_id
        yield _sse_line({"event": "workflow_started", "conversation_id": thread_id})

        try:
            human = await self._human_from_query(query, files)
        except Exception as e:
            logger.warning("stream: build human failed: {}", e)
            yield _sse_line({"event": "error", "message": str(e)})
            return

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self._app.aget_state(config)
        prior: list[BaseMessage] = []
        if snapshot and snapshot.values:
            prior = list(snapshot.values.get("messages") or [])

        routing_query = self._routing_query(query, inputs)
        compact = self._is_compact_turn(routing_query, files)
        intent, include_image_history = self._route_intent(routing_query, files)
        if intent == "text_only":
            compact = True
        model_name = self._pick_model_name(routing_query, files)
        llm = get_llm(model_name=model_name, temperature=self._settings.llm_temperature)
        invoke_msgs = self._memory.build_messages(
            thread_id=thread_id,
            prior=prior,
            human=human,
            inputs=inputs,
            compact=compact,
            include_image_history=include_image_history,
        )

        acc = ""
        start = time.perf_counter()
        first_token_ms: float | None = None
        try:
            async for chunk in llm.astream(invoke_msgs):
                piece = ""
                if hasattr(chunk, "content"):
                    raw = chunk.content
                    if isinstance(raw, str):
                        piece = raw
                    elif isinstance(raw, list):
                        for p in raw:
                            if isinstance(p, dict) and p.get("type") == "text":
                                piece += str(p.get("text", ""))
                if piece:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - start) * 1000.0
                    acc += piece
                    yield _sse_line(
                        {"event": "message", "answer": piece, "conversation_id": thread_id}
                    )
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            logger.exception("langgraph stream failed: {}", e)
            yield _sse_line({"event": "error", "message": "对话流式失败"})
            return

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        ttfb_ms = first_token_ms if first_token_ms is not None else elapsed_ms
        fast_ratio = self._observe_model_pick(model_name)
        logger.info(
            "chat_perf mode=stream model={} intent={} compact={} include_image_history={} has_files={} prior_msgs={} invoke_msgs={} elapsed_ms={:.1f} ttfb_ms={:.1f} out_chars={} fast_ratio={:.3f}",
            model_name or "(default)",
            intent,
            compact,
            include_image_history,
            bool(files),
            len(prior),
            len(invoke_msgs),
            elapsed_ms,
            ttfb_ms,
            len(acc),
            fast_ratio,
        )
        self._record_perf_sample(
            mode="stream",
            model_name=model_name,
            compact=compact,
            has_files=bool(files),
            elapsed_ms=elapsed_ms,
            ttfb_ms=ttfb_ms,
            out_chars=len(acc),
        )
        await self._app.aupdate_state(
            config,
            {"messages": [human, AIMessage(content=acc)]},
            as_node="agent",
        )
        self._memory.observe_turn(
            thread_id=thread_id,
            user_text=query,
            assistant_text=acc,
            had_image=bool(files),
            inputs=inputs,
        )
        yield _sse_line({"event": "message_end", "conversation_id": thread_id})

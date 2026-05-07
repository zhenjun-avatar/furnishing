from __future__ import annotations

import base64
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from salon_gateway.ai.furnishing_compose_prompt import build_furnishing_compose_prompt
from salon_gateway.ai.langgraph_chat import LangGraphChatClient
from salon_gateway.ai.resolve_image import resolve_base_image_for_dashscope
from salon_gateway.ai.store import ConversationStore
from salon_gateway.ai.upload_cache import SimulateUploadStore
from salon_gateway.ai.wan27_image import Wan27ImageClient
from salon_gateway.config import SalonGatewaySettings
from salon_gateway.furnishing.registry import FurnishingRegistry
from salon_gateway.models.messages import WecomImageInbound, WecomTextInbound

if TYPE_CHECKING:
    from salon_gateway.ai.protocol import ChatClient


def _remote_url_file(url: str) -> dict[str, Any]:
    return {"type": "image", "transfer_method": "remote_url", "url": url}


def _upload_file_ref(upload_file_id: str) -> dict[str, Any]:
    return {"type": "image", "transfer_method": "local_file", "upload_file_id": upload_file_id}


@lru_cache(maxsize=8)
def _furnishing_registry_cached(path_key: str) -> FurnishingRegistry:
    return FurnishingRegistry(Path(path_key))


_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_ASSET_ID_RE = re.compile(r"\bdemo-[a-z0-9-]+\b", re.IGNORECASE)

_REQ_SNIPPET_MAX = 6
_REQ_SNIPPET_CHAR_MAX = 160
_CONSULT_SUMMARY_CHAR_MAX = 1200
_ASSET_CACHE_TTL_SECONDS = 300.0
_ASSET_HINT_CACHE_TTL_SECONDS = 180.0

_MAJOR_CITIES = (
    "北京",
    "上海",
    "广州",
    "深圳",
    "杭州",
    "成都",
    "武汉",
    "西安",
    "南京",
    "苏州",
    "重庆",
    "天津",
    "郑州",
    "长沙",
    "东莞",
    "佛山",
    "宁波",
    "无锡",
    "青岛",
    "合肥",
    "厦门",
    "福州",
    "济南",
    "昆明",
    "沈阳",
    "石家庄",
    "太原",
    "南宁",
    "贵阳",
    "海口",
    "兰州",
    "乌鲁木齐",
)

_STYLE_TERMS_ORDERED = (
    "现代简约",
    "新中式",
    "奶油风",
    "工业风",
    "法式",
    "原木风",
    "北欧",
    "日式",
    "美式",
    "轻奢",
    "中式",
    "简约",
    "原木",
    "现代",
)

_BUDGET_RANGE_RE = re.compile(r"(?:预算|报价|价位|费用)[：:\s]*([^\n。]{1,56}?)(?:[。\n]|$)")
_BUDGET_WAN_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-~～至到]\s*(\d+(?:\.\d+)?)\s*万")
_BUDGET_SIMPLE_WAN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*万")

_BOILERPLATE_USER_PREFIXES = (
    "我选择了这些素材",
    "生成效果图",
    "需要转人工",
    "暂不转人工",
    "查看可选素材",
)
_IMAGE_QUERY_STUB = "[图片] 请根据这张照片"


@dataclass(slots=True)
class _ServiceProfile:
    phone: str = ""
    handoff_known: bool = False
    handoff_requested: bool = False
    has_room_image: bool = False
    latest_image_url: str = ""
    latest_upload_file_id: str = ""
    selected_asset_ids: list[str] = field(default_factory=list)
    turn_count: int = 0
    contact_prompted: bool = False
    ui_state: str = "INIT"
    ui_actions: list[str] = field(default_factory=list)
    requirement_snippets: list[str] = field(default_factory=list)
    slot_city: str = ""
    slot_budget: str = ""
    slot_style: str = ""


class SalonPipeline:
    def __init__(
        self,
        settings: SalonGatewaySettings,
        chat: ChatClient,
        store: ConversationStore,
    ) -> None:
        self._s = settings
        self._chat = chat
        self._store = store
        self._profiles: dict[str, _ServiceProfile] = {}
        self._assets_cache_at: float = 0.0
        self._assets_cache_items: list[Any] = []
        self._asset_hint_cache_at: float = 0.0
        self._asset_hint_cache_text: str = ""

    def _session_user(self, wecom_user: str) -> str:
        p = self._s.dify_user_prefix.strip() or "wecom"
        return f"{p}:{wecom_user}"

    def _profile(self, user: str) -> _ServiceProfile:
        p = self._profiles.get(user)
        if p is None:
            p = _ServiceProfile()
            self._profiles[user] = p
        return p

    def _update_profile_from_text(
        self,
        user: str,
        text: str,
        *,
        has_room_image: bool = False,
        image_url: str | None = None,
        upload_file_id: str | None = None,
        selected_asset_ids: list[str] | None = None,
        action: str | None = None,
    ) -> _ServiceProfile:
        t = (text or "").strip()
        act = (action or "").strip()
        p = self._profile(user)
        p.turn_count += 1
        if has_room_image:
            p.has_room_image = True
        if (image_url or "").strip():
            p.latest_image_url = (image_url or "").strip()
            p.latest_upload_file_id = ""
        if (upload_file_id or "").strip():
            p.latest_upload_file_id = (upload_file_id or "").strip()
            p.latest_image_url = ""
        m = _PHONE_RE.search(t)
        if m:
            p.phone = m.group(1)
        if act == "handoff_yes":
            p.handoff_known = True
            p.handoff_requested = True
        elif act == "handoff_no":
            p.handoff_known = True
            p.handoff_requested = False
        ids = self._valid_asset_ids([*(selected_asset_ids or []), *self._extract_asset_ids_from_text(t)])
        if ids:
            p.selected_asset_ids = self._merge_asset_ids(p.selected_asset_ids, ids)
        self._merge_slots_from_text(p, t)
        self._record_requirement_snippet(p, t)
        return p

    @staticmethod
    def _normalize_requirement_snippet(text: str) -> str | None:
        t = (text or "").strip().replace("\r\n", "\n").replace("\r", "\n")
        if len(t) < 3:
            return None
        for bp in _BOILERPLATE_USER_PREFIXES:
            if t.startswith(bp):
                return None
        if t.startswith(_IMAGE_QUERY_STUB):
            return "上传实景图，咨询软装与空间搭配"
        if len(t) > _REQ_SNIPPET_CHAR_MAX:
            return t[: _REQ_SNIPPET_CHAR_MAX - 1].rstrip() + "…"
        return t

    def _merge_slots_from_text(self, profile: _ServiceProfile, text: str) -> None:
        raw = (text or "").strip()
        if not raw:
            return
        if not profile.slot_city:
            for c in _MAJOR_CITIES:
                if c in raw:
                    profile.slot_city = c
                    break
        bud = ""
        mr = _BUDGET_WAN_RANGE_RE.search(raw)
        if mr:
            bud = f"{mr.group(1)}-{mr.group(2)}万"
        else:
            ms = _BUDGET_SIMPLE_WAN_RE.search(raw)
            if ms:
                bud = f"{ms.group(1)}万"
            else:
                mm = _BUDGET_RANGE_RE.search(raw)
                if mm:
                    bud = mm.group(1).strip()[:56]
        if bud:
            profile.slot_budget = bud
        found_styles = [s for s in _STYLE_TERMS_ORDERED if s in raw]
        if found_styles:
            existing = [x.strip() for x in profile.slot_style.split("、") if x.strip()] if profile.slot_style else []
            order: list[str] = []
            for x in [*existing, *found_styles]:
                if x not in order:
                    order.append(x)
            profile.slot_style = "、".join(order[:6])

    def _record_requirement_snippet(self, profile: _ServiceProfile, text: str) -> None:
        snip = self._normalize_requirement_snippet(text)
        if not snip:
            return
        if profile.requirement_snippets and profile.requirement_snippets[-1] == snip:
            return
        profile.requirement_snippets.append(snip)
        if len(profile.requirement_snippets) > _REQ_SNIPPET_MAX:
            profile.requirement_snippets = profile.requirement_snippets[-_REQ_SNIPPET_MAX :]

    def _all_assets_by_id(self) -> dict[str, Any]:
        return {x.id: x for x in self._cached_assets()}

    def _cached_assets(self) -> list[Any]:
        now = time.monotonic()
        if self._assets_cache_items and (now - self._assets_cache_at) < _ASSET_CACHE_TTL_SECONDS:
            return self._assets_cache_items
        reg = _furnishing_registry_cached(self._s.furnishing_assets_path.as_posix())
        items, _ = reg.search(q="", category="", limit=100)
        self._assets_cache_items = list(items)
        self._assets_cache_at = now
        return self._assets_cache_items

    def _valid_asset_ids(self, ids: list[str] | None) -> list[str]:
        raw = [str(x).strip() for x in (ids or []) if str(x).strip()]
        if not raw:
            return []
        try:
            by_id = self._all_assets_by_id()
        except Exception as e:
            logger.debug("valid asset id check skipped: {}", e)
            return []
        by_lower = {k.lower(): k for k in by_id}
        out: list[str] = []
        for x in raw:
            canonical = by_lower.get(x.lower()) or self._canonical_asset_id_by_numeric_suffix(x, by_id)
            if canonical and canonical not in out:
                out.append(canonical)
        return out[:4]

    @staticmethod
    def _canonical_asset_id_by_numeric_suffix(raw_id: str, by_id: dict[str, Any]) -> str | None:
        """Treat demo-foo-01 and demo-foo-001 as the same product if unambiguous."""
        m = re.match(r"^(?P<prefix>.+-)(?P<num>\d+)$", (raw_id or "").strip(), re.IGNORECASE)
        if not m:
            return None
        prefix = m.group("prefix").lower()
        try:
            n = int(m.group("num"))
        except ValueError:
            return None
        matches: list[str] = []
        for aid in by_id:
            am = re.match(r"^(?P<prefix>.+-)(?P<num>\d+)$", aid.strip(), re.IGNORECASE)
            if am and am.group("prefix").lower() == prefix and int(am.group("num")) == n:
                matches.append(aid)
        return matches[0] if len(matches) == 1 else None

    def _extract_asset_ids_from_text(self, text: str) -> list[str]:
        t = text or ""
        ids = _ASSET_ID_RE.findall(t)
        try:
            for aid in self._all_assets_by_id():
                if aid in t:
                    ids.append(aid)
        except Exception as e:
            logger.debug("asset id extraction by registry skipped: {}", e)
        return self._merge_asset_ids([], ids)

    @staticmethod
    def _merge_asset_ids(existing: list[str], incoming: list[str]) -> list[str]:
        out: list[str] = []
        for x in [*existing, *incoming]:
            s = str(x).strip()
            if s and s not in out:
                out.append(s)
        return out[:4]

    def _effective_asset_ids(self, profile: _ServiceProfile, selected_asset_ids: list[str] | None) -> list[str]:
        current = self._valid_asset_ids(selected_asset_ids)
        return self._merge_asset_ids(current, profile.selected_asset_ids)

    @staticmethod
    def _effective_image_refs(
        profile: _ServiceProfile,
        *,
        image_url: str | None,
        upload_file_id: str | None,
    ) -> tuple[str | None, str | None]:
        current_upload = (upload_file_id or "").strip()
        current_url = (image_url or "").strip()
        if current_upload:
            return None, current_upload
        if current_url:
            return current_url, None
        if profile.latest_upload_file_id:
            return None, profile.latest_upload_file_id
        if profile.latest_image_url:
            return profile.latest_image_url, None
        return None, None

    def _inject_customer_service_goal(self, query: str, profile: _ServiceProfile) -> str:
        return query

    @staticmethod
    def _memory_inputs(profile: _ServiceProfile, *, has_room_image: bool, selected_asset_ids: list[str]) -> dict[str, Any]:
        return {
            "phone": profile.phone or "",
            "handoff_requested": profile.handoff_requested if profile.handoff_known else None,
            "has_room_image": has_room_image,
            "selected_asset_ids": list(selected_asset_ids),
        }

    def ui_snapshot(self, wecom_user: str) -> dict[str, object]:
        p = self._profile(self._session_user(wecom_user))
        return {"state": p.ui_state, "actions": list(p.ui_actions)}

    def profile_phone(self, wecom_user: str) -> str:
        return (self._profile(self._session_user(wecom_user)).phone or "").strip()

    def perf_stats(self) -> dict[str, Any]:
        fn = getattr(self._chat, "perf_stats", None)
        if callable(fn):
            try:
                data = fn()
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning("pipeline perf_stats read failed: {}", e)
        return {
            "total_turns": 0,
            "fast_turns": 0,
            "fast_ratio": 0.0,
            "sample_size": 0,
            "sample_window_max": 0,
            "mode_counts": {"stream": 0, "complete": 0},
            "compact_ratio": 0.0,
            "with_files_ratio": 0.0,
            "ttfb_ms": {"p50": 0.0, "p95": 0.0, "max": 0.0},
            "elapsed_ms": {"p50": 0.0, "p95": 0.0, "max": 0.0},
            "recent_samples": [],
        }

    def reset_perf_stats(self) -> dict[str, Any]:
        fn = getattr(self._chat, "reset_perf_stats", None)
        if callable(fn):
            try:
                data = fn()
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning("pipeline reset_perf_stats failed: {}", e)
        return {"ok": False, "cleared_samples": 0}

    def _format_selected_assets_for_summary(self, ids: list[str]) -> str:
        if not ids:
            return "暂无"
        try:
            by_id = self._all_assets_by_id()
            parts: list[str] = []
            for aid in ids[:8]:
                it = by_id.get(aid)
                label = aid
                if it is not None:
                    name = getattr(it, "name", "") or ""
                    if name:
                        label = f"{aid}（{name}）"
                parts.append(label)
            return "、".join(parts)
        except Exception as e:
            logger.debug("asset labels for summary skipped: {}", e)
            return "、".join(ids[:8])

    def consult_content_summary(self, wecom_user: str, latest_user_text: str = "") -> str:
        """飞书「需求摘要」：会话画像 + 槽位 + 多轮诉求要点，避免仅截取末句。"""
        user = self._session_user(wecom_user)
        p = self._profile(user)
        snippets = list(p.requirement_snippets)
        extra = self._normalize_requirement_snippet(latest_user_text or "")
        if extra and (not snippets or snippets[-1] != extra):
            snippets.append(extra)

        lines: list[str] = []
        lines.append("【咨询会话摘要】")
        ph = (p.phone or "").strip()
        lines.append(f"• 联系方式：{'已记录手机号 ' + ph if ph else '待用户补充手机号'}")
        has_img = bool(p.latest_image_url or p.latest_upload_file_id or p.has_room_image)
        lines.append(f"• 实景图：{'已提供' if has_img else '未提供'}")
        lines.append(f"• 意向素材：{self._format_selected_assets_for_summary(p.selected_asset_ids)}")
        if p.handoff_known:
            ho = "需要人工跟进" if p.handoff_requested else "暂不转人工"
        else:
            ho = "用户尚未选择转人工与否"
        lines.append(f"• 转人工：{ho}")
        if (p.slot_city or "").strip():
            lines.append(f"• 城市/区域：{p.slot_city.strip()}")
        if (p.slot_budget or "").strip():
            lines.append(f"• 预算：{p.slot_budget.strip()}")
        if (p.slot_style or "").strip():
            lines.append(f"• 风格偏好：{p.slot_style.strip()}")
        show_snips = snippets[-6:] if len(snippets) > 6 else snippets
        if show_snips:
            lines.append("")
            lines.append("【用户诉求要点】")
            for i, s in enumerate(show_snips, 1):
                lines.append(f"{i}. {s}")
        out = "\n".join(lines).strip()
        if len(out) > _CONSULT_SUMMARY_CHAR_MAX:
            out = out[: _CONSULT_SUMMARY_CHAR_MAX - 1].rstrip() + "…"
        return out

    def handoff_simulate_ack(self, wecom_user: str, action: str | None) -> str:
        """仅 simulate 转人工控件：更新会话画像并返回给前端的固定提示（不走 LLM）。"""
        user = self._session_user(wecom_user)
        p0 = self._profile(user)
        act = (action or "").strip()
        kw: dict[str, Any] = dict(
            has_room_image=p0.has_room_image,
            selected_asset_ids=list(p0.selected_asset_ids),
            action=act or None,
        )
        if (p0.latest_upload_file_id or "").strip():
            kw["upload_file_id"] = p0.latest_upload_file_id.strip()
        elif (p0.latest_image_url or "").strip():
            kw["image_url"] = p0.latest_image_url.strip()
        self._update_profile_from_text(user, "", **kw)
        p = self._profile(user)
        has_room = bool(p.latest_image_url or p.latest_upload_file_id)
        has_assets = bool(p.selected_asset_ids)
        self._set_ui(p, has_room_image=has_room, has_assets=has_assets, action=act or None)
        if act == "handoff_no":
            return "已记录你的选择：暂不转人工。若之后需要人工协助，随时告诉我即可。"
        if act == "handoff_yes":
            base = "已为你登记「转人工」需求，我们会尽快安排客服查看您的会话与素材记录。"
            ph = (p.phone or "").strip()
            if ph:
                return f"{base}\n\n当前已留存联系电话 **{ph}**。如需更换，请直接发送新的 11 位手机号。"
            return f"{base}\n\n为便于客服回电或添加您，请直接在下方发送您的 **11 位手机号**（仅用于本次服务跟进）。"
        return ""

    def _set_ui(
        self,
        profile: _ServiceProfile,
        *,
        has_room_image: bool,
        has_assets: bool,
        action: str | None,
        fresh_generate_opportunity: bool = False,
        preview_ready: bool = False,
    ) -> None:
        act = (action or "").strip()
        # 已确认转人工且已留手机号：进入等待人工态，不再引导用户继续点选功能按钮。
        if profile.handoff_known and profile.handoff_requested and (profile.phone or "").strip():
            profile.ui_state = "WAITING_HUMAN"
            profile.ui_actions = []
            return
        actions: list[str] = []
        if preview_ready:
            state = "PREVIEW_READY"
        elif not has_room_image:
            state = "NEED_ROOM_IMAGE"
        elif not has_assets:
            state = "NEED_ASSETS"
        else:
            state = "READY_TO_GENERATE"
            if fresh_generate_opportunity and act != "generate_preview":
                actions.append("generate_preview")
        if profile.turn_count >= 3 and not profile.handoff_known and act not in {"handoff_yes", "handoff_no"}:
            actions.extend(["handoff_yes", "handoff_no"])
        profile.ui_state = state
        profile.ui_actions = actions

    def _should_prompt_contact(self, profile: _ServiceProfile) -> bool:
        return not profile.phone and not profile.contact_prompted and profile.turn_count >= 4

    def _append_contact_prompt(self, reply: str, profile: _ServiceProfile) -> str:
        if not self._should_prompt_contact(profile):
            return reply
        profile.contact_prompted = True
        prompt = (
            "如果方便，也可以留下手机号；我会把当前方案和素材选择一起记录，"
            "便于后续报价或客服跟进。"
        )
        r = (reply or "").strip()
        return f"{r}\n\n{prompt}" if r else prompt

    def _inject_furnishing_flow_hint(self, query: str, *, has_room_image: bool, has_assets: bool) -> str:
        q = (query or "").strip()
        asset_hint = self._build_asset_suggestion_hint()
        if has_room_image and has_assets:
            hint = (
                "【家装流程】当前已具备实景图和素材，可基于两者给出定制建议，并提醒可直接生成效果图。"
            )
        elif has_room_image and not has_assets:
            hint = (
                "【家装流程】当前已有实景图，但未选择素材。请先给简短建议，并引导用户从素材库选择产品后再生成效果图。"
            )
        elif (not has_room_image) and has_assets:
            hint = (
                "【家装流程】当前已选素材，但缺少实景图。请提示用户先上传实景图，再进行合成效果图。"
            )
        else:
            hint = (
                "【家装流程】当前缺少实景图与素材。请先引导上传实景图，再引导选择素材，最后再生成效果图。"
            )
        if asset_hint:
            hint = f"{hint}\n{asset_hint}"
        return f"{q}\n\n{hint}" if q else hint

    def _build_asset_suggestion_hint(self) -> str:
        now = time.monotonic()
        if self._asset_hint_cache_text and (now - self._asset_hint_cache_at) < _ASSET_HINT_CACHE_TTL_SECONDS:
            return self._asset_hint_cache_text
        try:
            items = self._cached_assets()[:20]
        except Exception as e:
            logger.debug("build asset suggestion hint skipped: {}", e)
            return ""
        if not items:
            self._asset_hint_cache_text = ""
            self._asset_hint_cache_at = now
            return ""
        lines = [f"- {it.id}：{(it.name or it.id)}" for it in items]
        text = (
            "【真实素材库】可用素材如下，请仅从这些素材中推荐或生成：\n"
            + "\n".join(lines)
            + "\n【严格约束】\n"
            "- 产品推荐必须引用上方真实素材ID；禁止推荐素材库不存在的品类。\n"
            "- 若用户要求生成效果图，但未选择素材ID，必须先让用户从上方素材ID中选择，不能承诺已经开始生成。\n"
            "- 用户也可以选择列表之外的素材，但必须提供明确素材ID或图片。"
        )
        self._asset_hint_cache_text = text
        self._asset_hint_cache_at = now
        return text

    def _catalog_items_for_query(self, query: str) -> list:
        reg = _furnishing_registry_cached(self._s.furnishing_assets_path.as_posix())
        items, _ = reg.search(q="", category="", limit=6)
        return items

    def _catalog_recommend_reply_for_action(self, action: str | None) -> str | None:
        if (action or "").strip() != "show_assets":
            return None
        items = self._catalog_items_for_query("")
        if not items:
            return "当前素材库没有匹配这类产品的条目，你可以换一个品类，或先从左侧素材库选择已有素材。"
        lines = [f"- `{it.id}`：{it.name or it.id}（{it.category or '未分类'}）" for it in items]
        return (
            "我从当前真实素材库里为你筛选了这些可选产品：\n"
            + "\n".join(lines)
            + "\n你可以回复其中的素材ID，或在左侧素材库点选；选好后我可以结合实景图生成效果图。"
        )

    @staticmethod
    def _is_generate_action(action: str | None) -> bool:
        return (action or "").strip() == "generate_preview"

    @staticmethod
    def _is_asset_selection_action(action: str | None) -> bool:
        act = (action or "").strip()
        return act not in {"generate_preview", "handoff_yes", "handoff_no", "show_assets"}

    def _asset_selection_fast_reply(self, *, has_room_image: bool, selected_asset_ids: list[str]) -> str:
        n = len(selected_asset_ids)
        if has_room_image:
            return f"已记录你选择的 {n} 个素材，可直接点击下方“生成效果图”。"
        return (
            f"已记录你选择的 {n} 个素材。"
            "下一步请先上传实景图，上传后可直接点击“生成效果图”。"
        )

    @staticmethod
    def _room_image_fast_reply() -> str:
        return "已收到实景图，且你已选好素材，可直接点击下方“生成效果图”。"

    def _handoff_phone_fast_reply(self, profile: _ServiceProfile, query: str, action: str | None) -> str | None:
        """已登记转人工后，手机号消息直接收口，避免重复调用 LLM。"""
        if (action or "").strip() in {"handoff_yes", "handoff_no"}:
            return None
        if not (profile.handoff_known and profile.handoff_requested):
            return None
        text = (query or "").strip()
        if not text:
            return None
        m = _PHONE_RE.search(text)
        if not m:
            return None
        return (
            f"已收到你的手机号 **{m.group(1)}**，并已完成转人工登记。"
            "客服会尽快联系你并跟进本次空间搭配需求。"
        )

    def _missing_asset_for_generation_reply(self) -> str:
        asset_hint = self._build_asset_suggestion_hint()
        return (
            "还不能直接生成效果图：我已经有实景图，但还缺少要放入空间的素材。\n\n"
            "请先从素材库选择 1-4 个素材ID（也可以在左侧点选素材），我再基于“实景图 + 素材”生成效果图。\n\n"
            f"{asset_hint}"
        )

    def _missing_room_for_generation_reply(self) -> str:
        return (
            "还不能直接生成效果图：我已经记录了你选择的素材，但还缺少实景图。\n\n"
            "请先上传客厅或卧室的实景照片，上传后回复“生成”即可直接生成效果图。"
        )

    async def _try_compose_preview(
        self,
        *,
        conversation_id: str,
        content: str,
        image_url: str | None,
        upload_file_id: str | None,
        selected_asset_ids: list[str] | None,
    ) -> str | None:
        ids = [x.strip() for x in (selected_asset_ids or []) if str(x).strip()]
        if not ids:
            return None
        if not (image_url or upload_file_id):
            return None

        if not self._s.dashscope_api_key:
            return "已识别到素材选择，但未配置 SALON_DASHSCOPE_API_KEY，暂无法生成合成效果图。"
        model_id = (self._s.wanxiang_model or "").strip().lower()
        if model_id not in ("wan2.7-image", "wan2.7-image-pro"):
            return "已识别到素材选择，请将 SALON_WANXIANG_MODEL 设为 wan2.7-image 或 wan2.7-image-pro 后再生成效果图。"

        reg = _furnishing_registry_cached(self._s.furnishing_assets_path.as_posix())
        all_items, _ = reg.search(q="", category="", limit=100)
        by_id = {x.id: x for x in all_items}
        chosen = [by_id[i] for i in ids if i in by_id and (by_id[i].image_url or "").strip()]
        if not chosen:
            return "已识别到素材选择，但所选素材不存在或缺少图片 URL，请重新选择后再试。"

        if upload_file_id:
            hit = SimulateUploadStore.instance().get(upload_file_id)
            if hit is None:
                return "上传图片已过期，请重新上传实景图再生成效果图。"
            mime = (hit.mime or "image/jpeg").split(";", 1)[0].strip() or "image/jpeg"
            room_ref = f"data:{mime};base64,{base64.standard_b64encode(hit.data).decode('ascii')}"
        else:
            try:
                room_ref = await resolve_base_image_for_dashscope(image_url or "", self._s)
            except Exception as e:
                logger.warning("compose room image resolve failed: {}", e)
                return "实景图读取失败，请更换图片或稍后重试。"

        product_refs = [x.image_url.strip() for x in chosen]
        prompt = build_furnishing_compose_prompt(
            n_product_images=len(product_refs),
            placement_hint="",
            style_notes=content,
        )
        try:
            client = Wan27ImageClient(
                self._s.dashscope_api_key,
                (self._s.wanxiang_model or "").strip(),
                self._s.dashscope_base_url,
            )
            result = await client.edit_with_images([room_ref, *product_refs], prompt)
        except Exception as e:
            logger.exception("compose preview generation failed: {}", e)
            return "效果图生成失败，请稍后重试。"

        selected_line = "、".join([x.name or x.id for x in chosen])
        return (
            f"已基于实景图和素材生成效果图（{selected_line}）。\n\n"
            f"![效果图]({result.preview_url})\n\n"
            f"预览链接：{result.preview_url}"
        )

    async def _complete(
        self,
        wecom_user: str,
        query: str,
        files: list[dict[str, Any]] | None = None,
        selected_asset_ids: list[str] | None = None,
        image_url: str | None = None,
        upload_file_id: str | None = None,
        action: str | None = None,
    ) -> str:
        user = self._session_user(wecom_user)
        current_has_room_image = bool((image_url or "").strip() or (upload_file_id or "").strip())
        profile = self._update_profile_from_text(
            user,
            query,
            has_room_image=current_has_room_image,
            image_url=image_url,
            upload_file_id=upload_file_id,
            selected_asset_ids=selected_asset_ids,
            action=action,
        )
        effective_image_url, effective_upload_file_id = self._effective_image_refs(
            profile,
            image_url=image_url,
            upload_file_id=upload_file_id,
        )
        effective_asset_ids = self._effective_asset_ids(profile, selected_asset_ids)
        current_asset_ids = self._valid_asset_ids(selected_asset_ids)
        act = (action or "").strip()
        fresh_generate_opportunity = current_has_room_image or bool(current_asset_ids)
        query_eff = self._inject_customer_service_goal(query, profile)
        has_room_image = bool(effective_image_url or effective_upload_file_id)
        has_assets = bool(effective_asset_ids)
        handoff_phone_reply = self._handoff_phone_fast_reply(profile, query, action)
        if handoff_phone_reply:
            self._set_ui(
                profile,
                has_room_image=has_room_image,
                has_assets=has_assets,
                action=action,
            )
            return handoff_phone_reply
        if current_asset_ids and self._is_asset_selection_action(act):
            self._set_ui(
                profile,
                has_room_image=has_room_image,
                has_assets=has_assets,
                action=action,
                fresh_generate_opportunity=True,
            )
            return self._asset_selection_fast_reply(
                has_room_image=has_room_image,
                selected_asset_ids=current_asset_ids,
            )
        if current_has_room_image and has_assets and self._is_asset_selection_action(act):
            self._set_ui(
                profile,
                has_room_image=has_room_image,
                has_assets=has_assets,
                action=action,
                fresh_generate_opportunity=True,
            )
            return self._room_image_fast_reply()
        query_eff = self._inject_furnishing_flow_hint(
            query_eff,
            has_room_image=has_room_image,
            has_assets=has_assets,
        )
        if self._is_generate_action(action):
            if has_room_image and has_assets:
                cid = await self._store.get(user)
                compose_reply = await self._try_compose_preview(
                    conversation_id=cid or "",
                    content=query_eff,
                    image_url=effective_image_url,
                    upload_file_id=effective_upload_file_id,
                    selected_asset_ids=effective_asset_ids,
                )
                if compose_reply:
                    self._set_ui(
                        profile,
                        has_room_image=has_room_image,
                        has_assets=has_assets,
                        action=action,
                        preview_ready=True,
                    )
                    return self._append_contact_prompt(compose_reply, profile)
            if has_room_image and not has_assets:
                self._set_ui(profile, has_room_image=has_room_image, has_assets=has_assets, action=action)
                return self._append_contact_prompt(self._missing_asset_for_generation_reply(), profile)
            if has_assets and not has_room_image:
                self._set_ui(profile, has_room_image=has_room_image, has_assets=has_assets, action=action)
                return self._append_contact_prompt(self._missing_room_for_generation_reply(), profile)
        if not has_assets:
            catalog_reply = self._catalog_recommend_reply_for_action(action)
            if catalog_reply:
                self._set_ui(profile, has_room_image=has_room_image, has_assets=has_assets, action=action)
                return self._append_contact_prompt(catalog_reply, profile)
        cid = await self._store.get(user)
        try:
            answer, new_cid = await self._chat.complete(
                user=user,
                query=query_eff,
                conversation_id=cid,
                files=files,
                inputs={
                    **self._memory_inputs(
                        profile,
                        has_room_image=has_room_image,
                        selected_asset_ids=effective_asset_ids,
                    ),
                    "__raw_query": query,
                },
            )
        except httpx.HTTPStatusError as e:
            try:
                snippet = (e.response.text or "")[:4000]
            except Exception:
                snippet = ""
            logger.error(
                "chat upstream HTTP {} body: {}",
                e.response.status_code,
                snippet,
            )
            return "抱歉，系统暂时繁忙，请稍后再试。"
        except Exception as e:
            logger.exception("chat failed: {}", e)
            return "抱歉，系统暂时繁忙，请稍后再试。"
        if new_cid:
            await self._store.set(user, new_cid)
        self._set_ui(
            profile,
            has_room_image=has_room_image,
            has_assets=has_assets,
            action=action,
            fresh_generate_opportunity=fresh_generate_opportunity,
        )
        return self._append_contact_prompt(answer or "（无回复）", profile)

    async def handle_text(self, msg: WecomTextInbound) -> str:
        return await self._complete(msg.from_user, msg.content, image_url=None, upload_file_id=None)

    async def handle_image(self, msg: WecomImageInbound) -> str:
        files = [_remote_url_file(msg.pic_url)]
        return await self._complete(
            msg.from_user,
            query="[图片] 请根据这张照片做软装与空间搭配方面的分析与建议。",
            files=files,
            image_url=msg.pic_url,
            upload_file_id=None,
        )

    async def handle_message(self, msg: WecomTextInbound | WecomImageInbound) -> str:
        if isinstance(msg, WecomImageInbound):
            return await self.handle_image(msg)
        return await self.handle_text(msg)

    async def handle_with_image(
        self,
        wecom_user: str,
        content: str,
        *,
        image_url: str | None = None,
        upload_file_id: str | None = None,
        selected_asset_ids: list[str] | None = None,
        action: str | None = None,
    ) -> str:
        """Used by /simulate endpoint: text + optional image."""
        files: list[dict[str, Any]] | None = None
        if upload_file_id:
            files = [_upload_file_ref(upload_file_id)]
        elif image_url:
            files = [_remote_url_file(image_url)]
        query = content or "[图片] 请根据这张照片做软装与空间搭配方面的分析与建议。"
        return await self._complete(
            wecom_user,
            query,
            files,
            selected_asset_ids=selected_asset_ids,
            image_url=image_url,
            upload_file_id=upload_file_id,
            action=action,
        )

    async def handle_with_image_stream(
        self,
        wecom_user: str,
        content: str,
        *,
        image_url: str | None = None,
        upload_file_id: str | None = None,
        selected_asset_ids: list[str] | None = None,
        action: str | None = None,
    ) -> AsyncIterator[bytes]:
        files: list[dict[str, Any]] | None = None
        if upload_file_id:
            files = [_upload_file_ref(upload_file_id)]
        elif image_url:
            files = [_remote_url_file(image_url)]
        query = content or "[图片] 请根据这张照片做软装与空间搭配方面的分析与建议。"
        user = self._session_user(wecom_user)
        current_has_room_image = bool((image_url or "").strip() or (upload_file_id or "").strip())
        profile = self._update_profile_from_text(
            user,
            query,
            has_room_image=current_has_room_image,
            image_url=image_url,
            upload_file_id=upload_file_id,
            selected_asset_ids=selected_asset_ids,
            action=action,
        )
        effective_image_url, effective_upload_file_id = self._effective_image_refs(
            profile,
            image_url=image_url,
            upload_file_id=upload_file_id,
        )
        effective_asset_ids = self._effective_asset_ids(profile, selected_asset_ids)
        current_asset_ids = self._valid_asset_ids(selected_asset_ids)
        act = (action or "").strip()
        fresh_generate_opportunity = current_has_room_image or bool(current_asset_ids)
        has_room_image = bool(effective_image_url or effective_upload_file_id)
        has_assets = bool(effective_asset_ids)
        handoff_phone_reply = self._handoff_phone_fast_reply(profile, query, action)
        if handoff_phone_reply:
            self._set_ui(
                profile,
                has_room_image=has_room_image,
                has_assets=has_assets,
                action=action,
            )
            payload = json.dumps({"event": "message", "answer": handoff_phone_reply}, ensure_ascii=False)
            ui_payload = json.dumps({"event": "ui_actions", **self.ui_snapshot(wecom_user)}, ensure_ascii=False)
            yield f"data: {payload}\n\n".encode("utf-8")
            yield f"data: {ui_payload}\n\n".encode("utf-8")
            yield b"data: {\"event\":\"message_end\"}\n\n"
            return
        if current_asset_ids and self._is_asset_selection_action(act):
            self._set_ui(
                profile,
                has_room_image=has_room_image,
                has_assets=has_assets,
                action=action,
                fresh_generate_opportunity=True,
            )
            reply = self._asset_selection_fast_reply(
                has_room_image=has_room_image,
                selected_asset_ids=current_asset_ids,
            )
            payload = json.dumps({"event": "message", "answer": reply}, ensure_ascii=False)
            ui_payload = json.dumps({"event": "ui_actions", **self.ui_snapshot(wecom_user)}, ensure_ascii=False)
            yield f"data: {payload}\n\n".encode("utf-8")
            yield f"data: {ui_payload}\n\n".encode("utf-8")
            yield b"data: {\"event\":\"message_end\"}\n\n"
            return
        if current_has_room_image and has_assets and self._is_asset_selection_action(act):
            self._set_ui(
                profile,
                has_room_image=has_room_image,
                has_assets=has_assets,
                action=action,
                fresh_generate_opportunity=True,
            )
            reply = self._room_image_fast_reply()
            payload = json.dumps({"event": "message", "answer": reply}, ensure_ascii=False)
            ui_payload = json.dumps({"event": "ui_actions", **self.ui_snapshot(wecom_user)}, ensure_ascii=False)
            yield f"data: {payload}\n\n".encode("utf-8")
            yield f"data: {ui_payload}\n\n".encode("utf-8")
            yield b"data: {\"event\":\"message_end\"}\n\n"
            return
        query_eff = self._inject_customer_service_goal(query, profile)
        query_eff = self._inject_furnishing_flow_hint(
            query_eff,
            has_room_image=has_room_image,
            has_assets=has_assets,
        )
        if self._is_generate_action(action):
            if has_room_image and has_assets:
                compose_reply = await self._try_compose_preview(
                    conversation_id="",
                    content=query_eff,
                    image_url=effective_image_url,
                    upload_file_id=effective_upload_file_id,
                    selected_asset_ids=effective_asset_ids,
                )
                if compose_reply is not None:
                    self._set_ui(
                        profile,
                        has_room_image=has_room_image,
                        has_assets=has_assets,
                        action=action,
                        preview_ready=True,
                    )
                    compose_reply = self._append_contact_prompt(compose_reply, profile)
                    payload = json.dumps({"event": "message", "answer": compose_reply}, ensure_ascii=False)
                    ui_payload = json.dumps({"event": "ui_actions", **self.ui_snapshot(wecom_user)}, ensure_ascii=False)
                    yield f"data: {payload}\n\n".encode("utf-8")
                    yield f"data: {ui_payload}\n\n".encode("utf-8")
                    yield b"data: {\"event\":\"message_end\"}\n\n"
                    return
            if has_room_image and not has_assets:
                self._set_ui(profile, has_room_image=has_room_image, has_assets=has_assets, action=action)
                reply = self._append_contact_prompt(self._missing_asset_for_generation_reply(), profile)
                payload = json.dumps({"event": "message", "answer": reply}, ensure_ascii=False)
                ui_payload = json.dumps({"event": "ui_actions", **self.ui_snapshot(wecom_user)}, ensure_ascii=False)
                yield f"data: {payload}\n\n".encode("utf-8")
                yield f"data: {ui_payload}\n\n".encode("utf-8")
                yield b"data: {\"event\":\"message_end\"}\n\n"
                return
            if has_assets and not has_room_image:
                self._set_ui(profile, has_room_image=has_room_image, has_assets=has_assets, action=action)
                reply = self._append_contact_prompt(self._missing_room_for_generation_reply(), profile)
                payload = json.dumps({"event": "message", "answer": reply}, ensure_ascii=False)
                ui_payload = json.dumps({"event": "ui_actions", **self.ui_snapshot(wecom_user)}, ensure_ascii=False)
                yield f"data: {payload}\n\n".encode("utf-8")
                yield f"data: {ui_payload}\n\n".encode("utf-8")
                yield b"data: {\"event\":\"message_end\"}\n\n"
                return
        if not has_assets:
            catalog_reply = self._catalog_recommend_reply_for_action(action)
            if catalog_reply:
                self._set_ui(profile, has_room_image=has_room_image, has_assets=has_assets, action=action)
                catalog_reply = self._append_contact_prompt(catalog_reply, profile)
                payload = json.dumps({"event": "message", "answer": catalog_reply}, ensure_ascii=False)
                ui_payload = json.dumps({"event": "ui_actions", **self.ui_snapshot(wecom_user)}, ensure_ascii=False)
                yield f"data: {payload}\n\n".encode("utf-8")
                yield f"data: {ui_payload}\n\n".encode("utf-8")
                yield b"data: {\"event\":\"message_end\"}\n\n"
                return
        cid = await self._store.get(user)
        holder: list[str | None] = []
        try:
            async for chunk in self._chat.stream_complete(
                user=user,
                query=query_eff,
                conversation_id=cid,
                files=files,
                inputs={
                    **self._memory_inputs(
                        profile,
                        has_room_image=has_room_image,
                        selected_asset_ids=effective_asset_ids,
                    ),
                    "__raw_query": query,
                },
                conversation_id_holder=holder,
            ):
                yield chunk
            contact_prompt = self._append_contact_prompt("", profile)
            if contact_prompt:
                payload = json.dumps({"event": "message", "answer": f"\n\n{contact_prompt}"}, ensure_ascii=False)
                yield f"data: {payload}\n\n".encode("utf-8")
            self._set_ui(
                profile,
                has_room_image=has_room_image,
                has_assets=has_assets,
                action=action,
                fresh_generate_opportunity=fresh_generate_opportunity,
            )
            ui_payload = json.dumps({"event": "ui_actions", **self.ui_snapshot(wecom_user)}, ensure_ascii=False)
            yield f"data: {ui_payload}\n\n".encode("utf-8")
        except httpx.HTTPStatusError:
            raise
        except Exception:
            logger.exception("handle_with_image_stream failed user={}", wecom_user)
            raise
        finally:
            new_cid = holder[0] if holder else None
            if new_cid:
                await self._store.set(user, new_cid)


def default_pipeline(settings: SalonGatewaySettings) -> SalonPipeline:
    return SalonPipeline(
        settings=settings,
        chat=LangGraphChatClient(settings),
        store=ConversationStore.instance(),
    )

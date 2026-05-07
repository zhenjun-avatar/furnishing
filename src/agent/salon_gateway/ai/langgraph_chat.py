"""LangGraph 对话：替代 Dify chat-messages，会话用 MemorySaver + thread_id。"""

from __future__ import annotations

import base64
import json
import re
import uuid
from collections.abc import AsyncIterator
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
    def __init__(self, recent_messages: int, summary_max_chars: int) -> None:
        self._recent_messages = max(4, recent_messages)
        self._summary_max_chars = max(200, summary_max_chars)
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
    ) -> list[BaseMessage]:
        mem = self._thread(thread_id)
        mem.structured.merge_inputs(inputs)
        hist = [*prior, human]
        dropped = hist[:-self._recent_messages] if len(hist) > self._recent_messages else []
        recent = hist[-self._recent_messages:]
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
        )
        self._checkpointer = MemorySaver()
        self._app = self._build_graph().compile(checkpointer=self._checkpointer)

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
        try:
            human = await self._human_from_query(query, files)
            invoke_msgs = self._memory.build_messages(
                thread_id=thread_id,
                prior=prior,
                human=human,
                inputs=inputs,
            )
            llm = get_llm(
                model_name=(self._settings.llm_model or "").strip() or None,
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

        llm = get_llm(
            model_name=(self._settings.llm_model or "").strip() or None,
            temperature=self._settings.llm_temperature,
        )
        invoke_msgs = self._memory.build_messages(
            thread_id=thread_id,
            prior=prior,
            human=human,
            inputs=inputs,
        )

        acc = ""
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

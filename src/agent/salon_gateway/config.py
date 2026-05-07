from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_AGENT_ROOT = Path(__file__).resolve().parents[1]


def _strip_env_secret_wrapping(s: str) -> str:
    """BOM/whitespace + optional outer matching quotes from .env paste mistakes."""
    t = (s or "").strip().strip("\ufeff")
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t


class SalonGatewaySettings(BaseSettings):
    """Loads `SALON_*` from环境变量与 `src/agent/.env`。"""

    model_config = SettingsConfigDict(
        env_prefix="SALON_",
        env_file=str(_AGENT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = Field(default="INFO", description="loguru level")

    # --- 企业微信（自建应用「接收消息」回调）---
    wecom_token: str = Field(default="", description="回调 Token")
    wecom_encoding_aes_key: str = Field(default="", description="EncodingAESKey")
    wecom_corp_id: str = Field(default="", description="企业 CorpId")
    wecom_plaintext: bool = Field(
        default=False,
        description="True=明文模式（仅本地调试；生产请用密文）",
    )

    # --- Dify Chatflow / 对话应用 API ---
    dify_api_base: str = Field(
        default="http://localhost:5001/v1",
        description="Dify API 根，如 https://api.dify.ai/v1",
    )
    dify_api_key: str = Field(default="", description="应用 API Key")
    dify_user_prefix: str = Field(
        default="wecom",
        description="传给 Dify 的 user id 前缀，如 wecom:userid",
    )
    # Chatflow 开始节点必填变量：与控制台变量名一致，合并进 chat-messages 的 inputs
    dify_default_inputs_json: str = Field(default="{}", description='JSON 对象，如 {"lang":"zh"}')

    # --- LangGraph / LLM（对话；复用 tools.llm + core.config 的 DEFAULT_MODEL 与各厂商 Key）---
    llm_model: str = Field(
        default="",
        description="非空则覆盖 DEFAULT_MODEL；空则读环境变量 DEFAULT_MODEL",
    )
    llm_vision_model: str = Field(
        default="qwen/qwen-vl-plus",
        description="带图轮次强制使用的多模态模型",
    )
    llm_fast_model: str = Field(
        default="",
        description="快模型（可选）。短消息优先使用；为空则始终使用 llm_model/DEFAULT_MODEL",
    )
    llm_fast_query_chars: int = Field(
        default=80,
        ge=20,
        le=500,
        description="短消息路由阈值（字符数）；<= 阈值且无图片时可走快模型",
    )
    intent_router_enabled: bool = Field(
        default=True,
        description="是否启用规则意图路由（图像相关/文本事务/混合）",
    )
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="聊天温度")
    memory_recent_messages: int = Field(
        default=16,
        ge=4,
        le=60,
        description="每轮发送给模型的最近消息条数（不含系统提示/摘要）",
    )
    memory_recent_messages_short: int = Field(
        default=8,
        ge=4,
        le=30,
        description="短消息时使用的最近消息条数（进一步降低首 token 延迟）",
    )
    memory_summary_max_chars: int = Field(
        default=1200,
        ge=200,
        le=8000,
        description="滚动摘要最大字符数（超出会截断）",
    )
    memory_keep_image_turns: int = Field(
        default=1,
        ge=0,
        le=8,
        description="历史中最多保留多少个含图用户轮次（其余仅保留文本）",
    )

    # --- 内部：Dify HTTP 工具回调写预约 ---
    internal_booking_token: str = Field(
        default="",
        description="Bearer / X-Salon-Token；多个等价值用 | 分隔（Dify 与 .env 长度不一致时可各填一份）",
    )
    booking_flush_queue_maxsize: int = Field(
        default=2000,
        ge=200,
        le=20000,
        description="异步写飞书队列容量；满时会丢弃最旧消息保护主链路",
    )
    booking_flush_queue_warn_ratio: float = Field(
        default=0.7,
        ge=0.1,
        le=0.98,
        description="队列占用告警阈值（size/maxsize）",
    )
    booking_flush_retry_max_attempts: int = Field(
        default=3,
        ge=1,
        le=8,
        description="异步写飞书失败重试次数（含首次）",
    )
    booking_flush_retry_base_delay_ms: int = Field(
        default=150,
        ge=20,
        le=5000,
        description="异步写飞书重试基础退避毫秒（指数退避）",
    )

    # --- 模拟企微文本（仅调试；生产勿设置）---
    simulate_token: str = Field(
        default="",
        description="非空则开启 POST /simulate/wecom-text，需 Bearer / X-Salon-Token 与此相同",
    )

    # --- 通义万相（家居效果图等，可选）---
    dashscope_api_key: str = Field(
        default="",
        description="DashScope API Key；为空则关闭家装预览与 compose 等图像接口",
    )
    wanxiang_model: str = Field(
        default="wanx2.1-imageedit",
        description=(
            "万相模型：wanx2.1-imageedit（image-synthesis description_edit）；"
            "或 wan2.7-image / wan2.7-image-pro（multimodal-generation）"
        ),
    )
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/api/v1",
        description=(
            "DashScope 原生 API 根；国内账号用 https://dashscope.aliyuncs.com/api/v1，"
            "国际站账号用 https://dashscope-intl.aliyuncs.com/api/v1"
        ),
    )

    furnishing_assets_file: str = Field(
        default="",
        description="家居素材库 JSON；空则使用 salon_gateway/data/furnishing_assets.json",
    )

    @field_validator("dashscope_api_key", mode="before")
    @classmethod
    def normalize_dashscope_api_key(cls, v: object) -> str:
        if v is None:
            return ""
        return _strip_env_secret_wrapping(str(v))

    # --- 飞书多维表（可选；未配置则只打日志）---
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_bitable_app_token: str = ""
    feishu_bitable_table_id: str = Field(
        default="",
        description="数据表 table_id，仅 tbl 开头一段；勿把 URL 里的 &view= 粘进来",
    )
    # JSON: {"phone":"手机号","store":"门店",...} 将 BookingDraft 字段映射到飞书列名
    feishu_field_map_json: str = "{}"
    # JSON: {"手机号":"Phone","咨询类型":"SingleSelect",...} 覆盖自动类型推断
    feishu_field_types_json: str = "{}"

    @property
    def dify_default_inputs(self) -> dict[str, Any]:
        raw = self.dify_default_inputs_json.strip() or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): v for k, v in data.items()}

    @property
    def internal_booking_tokens_accepted(self) -> frozenset[str]:
        """Any of these secrets accepts POST /internal/booking (after Bearer / X-Salon-Token parse)."""
        raw = _strip_env_secret_wrapping(self.internal_booking_token)
        if not raw:
            return frozenset()
        out: set[str] = set()
        for part in raw.replace("\n", "|").split("|"):
            p = _strip_env_secret_wrapping(part)
            if p:
                out.add(p)
        return frozenset(out)

    @property
    def feishu_field_map(self) -> dict[str, str]:
        raw = self.feishu_field_map_json.strip() or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    @property
    def feishu_field_types(self) -> dict[str, str]:
        raw = self.feishu_field_types_json.strip() or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    @field_validator("internal_booking_token", mode="before")
    @classmethod
    def normalize_internal_booking_token(cls, v: object) -> str:
        if v is None:
            return ""
        return _strip_env_secret_wrapping(str(v))

    @field_validator("feishu_bitable_table_id")
    @classmethod
    def normalize_feishu_table_id(cls, v: str) -> str:
        """Wiki URL 常为 table=tblXXX&view=...，只保留 tbl 段。"""
        v = (v or "").strip()
        if not v:
            return v
        if "&" in v:
            v = v.split("&", 1)[0].strip()
        if "?" in v:
            v = v.split("?", 1)[0].strip()
        if v.lower().startswith("table="):
            v = v[6:].strip()
        return v

    @field_validator("dify_api_base")
    @classmethod
    def normalize_dify_base(cls, v: str) -> str:
        v = (v or "").strip().rstrip("/")
        if not v:
            return "http://localhost:5001/v1"
        return v if v.endswith("/v1") else f"{v}/v1"

    @property
    def furnishing_assets_path(self) -> Path:
        raw = (self.furnishing_assets_file or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return (_AGENT_ROOT / "salon_gateway" / "data" / "furnishing_assets.json").resolve()


@lru_cache
def get_settings() -> SalonGatewaySettings:
    return SalonGatewaySettings()

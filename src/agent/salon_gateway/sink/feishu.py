from __future__ import annotations

import re
import time
from typing import Any

import httpx
from loguru import logger

from salon_gateway.config import SalonGatewaySettings
from salon_gateway.models.booking import BookingDraft

_SINGLE_SELECT_UI = frozenset({"SingleSelect"})
_MULTI_SELECT_UI = frozenset({"MultiSelect"})
_TYPE_NAME_TO_CODE: dict[str, int] = {
    "text": 1,
    "number": 2,
    "singleselect": 3,
    "multiselect": 4,
    "datetime": 5,
    "checkbox": 7,
    "user": 11,
    "phone": 13,
    "url": 15,
}
_CODE_TO_UI: dict[int, str] = {
    1: "Text",
    2: "Number",
    3: "SingleSelect",
    4: "MultiSelect",
    5: "DateTime",
    7: "Checkbox",
    11: "User",
    13: "Phone",
    15: "Url",
}
_DEFAULT_SELECT_OPTIONS: dict[str, list[str]] = {
    "预算区间": ["<5w", "5-10w", "10-20w", "20w+", "未知"],
    "客户状态": ["新线索", "跟进中", "已预约", "已成交", "无效"],
    "咨询类型": ["预约咨询", "产品咨询", "方案咨询"],
    "工单状态": ["待跟进", "沟通中", "待到店", "已完成", "已关闭"],
    "结果": ["待定", "已预约", "已成交", "未成交"],
    "渠道": ["wecom", "simulate", "simulate-stream", "api"],
}


class FeishuBitableSink:
    """飞书多维表新增一行记录。"""

    _token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    # class-level: avoid hammering list-fields API (20/s limit)
    _fields_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
    _fields_cache_ttl_sec = 300.0

    def __init__(self, settings: SalonGatewaySettings) -> None:
        self._s = settings
        self._token: str | None = None
        self._token_deadline: float = 0.0

    def _fields_base_url(self) -> str:
        app = self._s.feishu_bitable_app_token
        tid = self._s.feishu_bitable_table_id
        return f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app}/tables/{tid}/fields"

    def _records_base_url(self) -> str:
        app = self._s.feishu_bitable_app_token
        tid = self._s.feishu_bitable_table_id
        return f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app}/tables/{tid}/records"

    async def _tenant_token(self, client: httpx.AsyncClient) -> str:
        now = time.monotonic()
        if self._token and now < self._token_deadline - 60:
            return self._token
        body = {"app_id": self._s.feishu_app_id, "app_secret": self._s.feishu_app_secret}
        r = await client.post(self._token_url, json=body)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != 0:
            raise RuntimeError(f"feishu token error: {data}")
        self._token = data["tenant_access_token"]
        expire = int(data.get("expire", 7200))
        self._token_deadline = now + float(expire)
        return self._token

    def _fields_cache_key(self) -> str:
        return f"{self._s.feishu_bitable_app_token}|{self._s.feishu_bitable_table_id}"

    @staticmethod
    def _normalize_type_name(type_name: str) -> str:
        return (type_name or "").strip().replace("_", "").replace("-", "").lower()

    @staticmethod
    def infer_field_type(field_name: str) -> str:
        n = (field_name or "").strip()
        if any(k in n for k in ("URL", "链接")):
            return "Url"
        if "手机号" in n:
            return "Phone"
        if n in _DEFAULT_SELECT_OPTIONS:
            return "SingleSelect"
        if "时间" in n:
            return "DateTime"
        return "Text"

    def _resolve_target_type(self, field_name: str) -> tuple[int, str]:
        override = self._s.feishu_field_types.get(field_name, "")
        chosen = override or self.infer_field_type(field_name)
        code = _TYPE_NAME_TO_CODE.get(self._normalize_type_name(chosen))
        if code is None:
            code = 1
            chosen = "Text"
        return code, chosen

    @staticmethod
    def _field_ui_name(fdef: dict[str, Any]) -> str:
        ui = str(fdef.get("ui_type") or "").strip()
        if ui:
            return ui
        t = int(fdef.get("type") or 0)
        return _CODE_TO_UI.get(t, str(t))

    @staticmethod
    def _looks_like_http_url(s: str) -> bool:
        return bool(re.match(r"^https?://", (s or "").strip(), flags=re.IGNORECASE))

    @classmethod
    def _coerce_field_value_for_type(cls, ui_type: str, value: Any) -> Any:
        """Coerce payload value to Feishu Bitable expected wire format."""
        ui = (ui_type or "").strip().lower()
        if ui == "url":
            if isinstance(value, str):
                v = value.strip()
                if not v:
                    return None
                if not cls._looks_like_http_url(v):
                    return None
                # Bitable Link/URL field expects an object, not plain string.
                return {"text": v, "link": v}
            return None
        return value

    def _normalize_fields_by_schema(
        self,
        fields_payload: dict[str, Any],
        table_fields: list[dict[str, Any]],
    ) -> dict[str, Any]:
        by_name = {str(f.get("field_name") or "").strip(): f for f in table_fields}
        out: dict[str, Any] = {}
        for col, val in fields_payload.items():
            fdef = by_name.get(str(col).strip())
            if not fdef:
                out[col] = val
                continue
            ui = self._field_ui_name(fdef)
            coerced = self._coerce_field_value_for_type(ui, val)
            if coerced is None and val not in (None, "", [], {}):
                logger.warning("drop incompatible field value col={} ui_type={} value={!r}", col, ui, val)
                continue
            out[col] = coerced
        return out

    async def list_table_fields(self) -> list[dict[str, Any]]:
        """All bitable field definitions (paginated)."""
        key = self._fields_cache_key()
        now = time.monotonic()
        hit = FeishuBitableSink._fields_cache.get(key)
        if hit and now - hit[0] < FeishuBitableSink._fields_cache_ttl_sec:
            return list(hit[1])

        items: list[dict[str, Any]] = []
        page_token: str | None = None
        async with httpx.AsyncClient(timeout=60.0) as client:
            token = await self._tenant_token(client)
            headers = {"Authorization": f"Bearer {token}"}
            while True:
                params: dict[str, str | int] = {"page_size": 100}
                if page_token:
                    params["page_token"] = page_token
                r = await client.get(self._fields_base_url(), headers=headers, params=params)
                r.raise_for_status()
                data = r.json()
                if data.get("code") != 0:
                    raise RuntimeError(f"feishu list fields error: {data}")
                block = data.get("data") or {}
                batch = block.get("items") or []
                items.extend(batch)
                if not block.get("has_more"):
                    break
                page_token = block.get("page_token")
                if not page_token:
                    break
        FeishuBitableSink._fields_cache[key] = (now, list(items))
        return items

    async def ensure_fields_from_map(self) -> dict[str, Any]:
        """Create missing fields declared in feishu_field_map using inferred/override types."""
        fmap = self._s.feishu_field_map
        wanted_names = sorted({v.strip() for v in fmap.values() if str(v).strip()})
        if not wanted_names:
            return {"ok": True, "created": [], "skipped": [], "warning": "field map empty"}

        existing = await self.list_table_fields()
        by_name = {str(f.get("field_name") or "").strip(): f for f in existing}
        created: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            token = await self._tenant_token(client)
            headers = {"Authorization": f"Bearer {token}"}
            for col in wanted_names:
                current = by_name.get(col)
                if current:
                    skipped.append(
                        {
                            "field_name": col,
                            "reason": "exists",
                            "current_type": self._field_ui_name(current),
                        }
                    )
                    continue

                t_code, t_name = self._resolve_target_type(col)
                payload: dict[str, Any] = {
                    "field_name": col,
                    "type": t_code,
                }
                if t_code in (3, 4):
                    opts = [{"name": x} for x in _DEFAULT_SELECT_OPTIONS.get(col, [])]
                    payload["property"] = {"options": opts}

                r = await client.post(self._fields_base_url(), json=payload, headers=headers)
                try:
                    data = r.json()
                except Exception:
                    data = {"_parse_error": (r.text or "")[:1200]}
                if r.status_code >= 400 or data.get("code") != 0:
                    raise RuntimeError(
                        f"create field failed field={col} type={t_name} status={r.status_code} body={data}"
                    )
                created.append({"field_name": col, "type": t_name})

        # force refresh cache on next call
        FeishuBitableSink._fields_cache.pop(self._fields_cache_key(), None)
        return {"ok": True, "created": created, "skipped": skipped}

    @staticmethod
    def _filter_option_names(
        options: list[dict[str, Any]],
        q: str,
    ) -> list[dict[str, Any]]:
        qn = (q or "").strip().lower()
        out: list[dict[str, Any]] = []
        for opt in options:
            name = str(opt.get("name") or "").strip()
            if not name:
                continue
            oid = str(opt.get("id") or "")
            if not qn or qn in name.lower():
                out.append({"id": oid, "name": name})
        return out

    async def booking_field_options(
        self,
        *,
        store_search: str = "",
        service_search: str = "",
    ) -> dict[str, Any]:
        """Options for store (SingleSelect) and service (MultiSelect) per feishu_field_map column names."""
        fmap = self._s.feishu_field_map
        store_col = (fmap.get("store") or "").strip()
        service_col = (fmap.get("service") or "").strip()
        result: dict[str, Any] = {
            "store": {"field_name": store_col, "ui_type": None, "options": []},
            "service": {"field_name": service_col, "ui_type": None, "options": []},
        }
        if not store_col and not service_col:
            result["warning"] = "feishu_field_map 缺少 store 或 service 列名映射"
            return result

        fields = await self.list_table_fields()
        by_name = {str(f.get("field_name") or ""): f for f in fields}

        def _resolve_ui(fdef: dict[str, Any]) -> str:
            t = int(fdef.get("type") or 0)
            if t == 3:
                return "SingleSelect"
            if t == 4:
                return "MultiSelect"
            return self._field_ui_name(fdef)

        def fill(key: str, col: str, search: str, allowed_ui: frozenset[str]) -> None:
            if not col:
                return
            fdef = by_name.get(col)
            if not fdef:
                result[key]["error"] = f"表中未找到名为「{col}」的列"
                return
            ui = _resolve_ui(fdef)
            result[key]["ui_type"] = ui
            if ui not in allowed_ui:
                result[key]["error"] = f"列「{col}」类型为 {ui}，需要 {allowed_ui}"
                return
            prop = fdef.get("property") or {}
            raw_opts = prop.get("options") or []
            if not isinstance(raw_opts, list):
                raw_opts = []
            result[key]["options"] = self._filter_option_names(raw_opts, search)

        fill("store", store_col, store_search, _SINGLE_SELECT_UI)
        fill("service", service_col, service_search, _MULTI_SELECT_UI)
        return result

    async def append_booking(self, draft: BookingDraft) -> None:
        fields = draft.to_feishu_fields(self._s.feishu_field_map)
        if not fields:
            logger.warning("feishu_field_map 为空，跳过写入；请配置 SALON_FEISHU_FIELD_MAP_JSON")
            return
        table_fields = await self.list_table_fields()
        fields = self._normalize_fields_by_schema(fields, table_fields)
        if not fields:
            logger.warning("all mapped fields were dropped by schema coercion; skip append")
            return
        url = self._records_base_url()
        conv_col = (self._s.feishu_field_map.get("conversation_id") or "").strip()
        conv_val = str(fields.get(conv_col) or "").strip() if conv_col else ""
        async with httpx.AsyncClient(timeout=60.0) as client:
            token = await self._tenant_token(client)
            headers = {"Authorization": f"Bearer {token}"}
            payload: dict[str, Any] = {"fields": fields}
            record_id = ""
            if conv_col and conv_val:
                record_id = await self._find_record_id_by_field(
                    client,
                    headers=headers,
                    field_name=conv_col,
                    expected_value=conv_val,
                )
            if record_id:
                r = await client.put(f"{url}/{record_id}", json=payload, headers=headers)
            else:
                r = await client.post(url, json=payload, headers=headers)
            try:
                data = r.json()
            except Exception:
                data = {"_parse_error": (r.text or "")[:2000]}
            if r.status_code >= 400:
                logger.error("feishu bitable HTTP {}: {}", r.status_code, data)
                raise RuntimeError(f"feishu HTTP {r.status_code}: {data}") from None
        if data.get("code") != 0:
            logger.error("feishu bitable business error: {}", data)
            raise RuntimeError(f"feishu bitable error: {data}")
        logger.info(
            "feishu_bitable_record_upserted mode={} {}",
            "update" if (conv_col and conv_val) else "create",
            data.get("data", {}),
        )

    async def _find_record_id_by_field(
        self,
        client: httpx.AsyncClient,
        *,
        headers: dict[str, str],
        field_name: str,
        expected_value: str,
    ) -> str:
        """Linear scan records to find record_id by exact field value."""
        page_token: str | None = None
        url = self._records_base_url()
        while True:
            params: dict[str, str | int] = {"page_size": 200}
            if page_token:
                params["page_token"] = page_token
            r = await client.get(url, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != 0:
                raise RuntimeError(f"feishu list records error: {data}")
            block = data.get("data") or {}
            items = block.get("items") or []
            for it in items:
                fields = it.get("fields") or {}
                got = str(fields.get(field_name) or "").strip()
                if got == expected_value:
                    rid = str(it.get("record_id") or "").strip()
                    if rid:
                        return rid
            if not block.get("has_more"):
                return ""
            page_token = block.get("page_token")
            if not page_token:
                return ""

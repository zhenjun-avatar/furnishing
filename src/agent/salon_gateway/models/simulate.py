from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SimulateWecomTextIn(BaseModel):
    """Dev-only: same pipeline as WeCom → Dify; JSON in/out.

    At least one of *content* or *image_url* / *upload_file_id* must be present.
    """

    content: str = Field(default="", description="User message text (may be empty when image only)")
    from_user: str = Field(default="sim-user-1", max_length=128)
    to_user: str = Field(default="corp", max_length=128)
    msg_id: str | None = Field(default=None, description="Optional; for logging only")

    # Image fields (mutually exclusive; upload_file_id takes priority)
    image_url: str | None = Field(
        default=None,
        description="Publicly accessible image URL (used as Dify remote_url)",
    )
    upload_file_id: str | None = Field(
        default=None,
        description="upload_file_id returned by POST /simulate/upload-image",
    )
    selected_asset_ids: list[str] = Field(
        default_factory=list,
        description="对话里勾选的素材 id（最多 4 个），用于实景+素材合成效果图",
    )
    action: str | None = Field(
        default=None,
        description="用户通过聊天控件确认的结构化动作，如 generate_preview / handoff_yes / handoff_no",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_asset_ids(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        raw = data.get("selected_asset_ids")
        if not isinstance(raw, list):
            data["selected_asset_ids"] = []
            return data
        out: list[str] = []
        for x in raw:
            s = str(x).strip()
            if s and s not in out:
                out.append(s)
            if len(out) >= 4:
                break
        data["selected_asset_ids"] = out
        return data

    @model_validator(mode="after")
    def _require_content_or_image(self) -> "SimulateWecomTextIn":
        if not self.content.strip() and not self.image_url and not self.upload_file_id and not self.action:
            raise ValueError("content 或 image_url / upload_file_id 至少提供一个")
        return self

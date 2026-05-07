from __future__ import annotations

from pydantic import BaseModel, Field


class ImagePreviewRequest(BaseModel):
    image_url: str = Field(description="可公开访问的 HTTPS 图片 URL（或经网关解析的 Dify 图链）")
    style_prompt: str = Field(
        default="",
        description="软装方案等文本描述；家居预览中为已确认的方案全文",
    )
    conversation_id: str = Field(default="", description="Dify 会话 ID，用于跨轮次缓存参考图")


class ImagePreviewResponse(BaseModel):
    preview_url: str = Field(description="通义万相生成的效果图 URL")
    task_id: str = Field(description="DashScope 任务 ID")

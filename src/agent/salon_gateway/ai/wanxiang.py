"""DashScope 通义万相图像编辑客户端（wanx2.1-imageedit）。

调用流程（异步轮询）：
    1. POST /services/aigc/image2image/image-synthesis  →  task_id
    2. GET  /tasks/{task_id}  轮询，直至 SUCCEEDED / FAILED
    3. 返回 results[0].url

家居场景使用 ``description_edit``（无 mask）。

base_image_url：
    公网 HTTPS URL 或 data:{mime};base64,...（网关对 Dify CDN 图片会代为下载转 data URI）。
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

import httpx
from loguru import logger

from salon_gateway.ai.home_furnishing_prompt import build_home_furnishing_prompt


def _key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12]


_DEFAULT_BASE = "https://dashscope.aliyuncs.com/api/v1"
_GENERATION_PATH = "/services/aigc/image2image/image-synthesis"
_TASK_PATH = "/tasks/{task_id}"

_POLL_INTERVAL_S = 3.0
_MAX_POLLS = 20  # 最多等待约 60 秒


@dataclass(slots=True)
class ImageEditResult:
    preview_url: str
    task_id: str


class WanxiangClient:
    """通义万相图像编辑（img2img）异步客户端（家居类 ``description_edit``）。"""

    def __init__(
        self,
        api_key: str,
        model: str = "wanx2.1-imageedit",
        base_url: str = _DEFAULT_BASE,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base = base_url.rstrip("/")

    async def generate_interior_preview(self, image_url: str, scheme_description: str) -> ImageEditResult:
        """家居空间效果示意：description_edit。"""
        prompt = build_home_furnishing_prompt(scheme_description)
        task_id = await self._submit(image_url, prompt)
        preview_url = await self._poll(task_id)
        return ImageEditResult(preview_url=preview_url, task_id=task_id)

    async def _submit(self, image_url: str, style_prompt: str) -> str:
        inp = {
            "function": "description_edit",
            "prompt": style_prompt,
            "base_image_url": image_url,
        }
        payload = {
            "model": self._model,
            "input": inp,
            "parameters": {"n": 1},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self._base}{_GENERATION_PATH}",
                headers=headers,
                json=payload,
            )
            if not resp.is_success:
                snippet = (resp.text or "")[:2000]
                logger.error(
                    "wanxiang _submit HTTP {} function=description_edit key_sha256_12={} body={}",
                    resp.status_code,
                    _key_fingerprint(self._api_key),
                    snippet,
                )
            resp.raise_for_status()
            data = resp.json()

        task_id: str = data["output"]["task_id"]
        logger.info(
            "wanxiang: task submitted task_id={} model={} function=description_edit",
            task_id,
            self._model,
        )
        return task_id

    async def _poll(self, task_id: str) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = f"{self._base}{_TASK_PATH.format(task_id=task_id)}"

        async with httpx.AsyncClient(timeout=30) as client:
            for attempt in range(1, _MAX_POLLS + 1):
                await asyncio.sleep(_POLL_INTERVAL_S)
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                status: str = data["output"]["task_status"]
                logger.debug(
                    "wanxiang: poll {}/{} task_id={} status={}",
                    attempt,
                    _MAX_POLLS,
                    task_id,
                    status,
                )
                if status == "SUCCEEDED":
                    return data["output"]["results"][0]["url"]
                if status in ("FAILED", "CANCELED"):
                    raise RuntimeError(
                        f"wanxiang task {task_id} ended with status={status}: {data}"
                    )

        raise TimeoutError(
            f"wanxiang task {task_id} did not complete within "
            f"{_MAX_POLLS * _POLL_INTERVAL_S:.0f}s"
        )

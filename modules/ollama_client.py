"""
modules/ollama_client.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Async HTTP client for Ollama /api/generate (translation + preload).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import List, Union
from urllib.error import URLError
from urllib.request import Request, urlopen

from config import TranslationConfig
from utils.logger import get_logger

log = get_logger("OllamaClient")

KeepAlive = Union[int, float, str]


class OllamaClient:
    """Ollama generate + preload + health check."""

    def __init__(self, cfg: TranslationConfig):
        self._cfg = cfg
        self._url = f"{cfg.ollama_base_url.rstrip('/')}/api/generate"
        self._tags_url = f"{cfg.ollama_base_url.rstrip('/')}/api/tags"

    async def generate(
        self,
        prompt: str,
        system: str,
        temperature: float,
        top_p: float,
        num_predict: int,
    ) -> str:
        payload = self._build_payload(
            prompt=prompt,
            system=system,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
        )
        try:
            import aiohttp  # type: ignore
            return await self._post_aiohttp(payload, aiohttp)
        except ImportError:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._post_blocking, payload
            )

    async def preload(self) -> None:
        """Load model into GPU/RAM (empty prompt + keep_alive)."""
        payload = {
            "model": self._cfg.model,
            "prompt": " ",
            "stream": False,
            "keep_alive": self._cfg.keep_alive,
        }
        t0 = time.monotonic()
        try:
            import aiohttp  # type: ignore
            await self._post_aiohttp(payload, aiohttp, expect_response=False)
        except ImportError:
            await asyncio.get_event_loop().run_in_executor(
                None, self._post_blocking, payload, False
            )
        elapsed = time.monotonic() - t0
        log.info(f"Ollama preload finished in {elapsed:.1f}s (model={self._cfg.model!r})")

    def check_reachable(self) -> None:
        """Raise if Ollama server is not reachable."""
        try:
            with urlopen(self._tags_url, timeout=5) as r:
                if r.status != 200:
                    raise RuntimeError(f"Ollama /api/tags returned {r.status}")
        except URLError as exc:
            raise RuntimeError(
                f"Ollama not reachable at {self._cfg.ollama_base_url}. "
                "Run: ollama serve"
            ) from exc

    def list_models(self) -> List[str]:
        """Return installed model names from /api/tags."""
        try:
            with urlopen(self._tags_url, timeout=5) as r:
                data = json.loads(r.read())
                models = [m.get("name", "") for m in data.get("models", [])]
                return [n for n in models if n]
        except Exception as exc:
            log.warning(f"Could not list Ollama models: {exc}")
            return []

    def warn_if_model_missing(self) -> None:
        names = self.list_models()
        if not names:
            log.warning("No Ollama models found. Run: ollama pull <model>")
            return
        target = self._cfg.model
        if target not in names and not any(target in n for n in names):
            log.warning(
                f"Model {target!r} not in Ollama. Available: {', '.join(names[:5])}. "
                f"Run: ollama pull {target}"
            )

    def _build_payload(
        self,
        prompt: str,
        system: str,
        temperature: float,
        top_p: float,
        num_predict: int,
    ) -> dict:
        return {
            "model": self._cfg.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": self._cfg.keep_alive,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
                "stop": ["\n\n", "---"],
            },
        }

    async def _post_aiohttp(
        self, payload: dict, aiohttp, expect_response: bool = True
    ) -> str:
        timeout = aiohttp.ClientTimeout(total=self._cfg.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self._url, json=payload) as resp:
                resp.raise_for_status()
                if not expect_response:
                    return ""
                data = await resp.json(content_type=None)
                return data.get("response", "").strip()

    def _post_blocking(self, payload: dict, expect_response: bool = True) -> str:
        body = json.dumps(payload).encode()
        req = Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self._cfg.timeout_s) as r:
            if not expect_response:
                return ""
            data = json.loads(r.read())
            return data.get("response", "").strip()

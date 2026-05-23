"""
modules/translator_factory.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Returns the correct LLM translator module based on AppConfig.active_provider.
Pipeline only imports this — never imports Groq/Gemini modules directly.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

from typing import Union

from config import AppConfig
from modules.groq_translator import GroqTranslatorModule
from modules.gemini_translator import GeminiTranslatorModule

TranslatorModule = Union[GroqTranslatorModule, GeminiTranslatorModule]


class TranslatorFactory:

    @staticmethod
    def create(cfg: AppConfig) -> TranslatorModule:
        """
        Instantiate and return the correct translator module.
        Raises ValueError for unknown provider.
        """
        provider = cfg.active_provider.lower().strip()

        if provider == "groq":
            return GroqTranslatorModule(cfg.groq)
        elif provider == "gemini":
            return GeminiTranslatorModule(cfg.gemini)
        else:
            raise ValueError(
                f"Unknown provider: {provider!r}. Must be 'groq' or 'gemini'."
            )

    @staticmethod
    def validate(cfg: AppConfig) -> None:
        """Validate API key + SDK availability for the active provider."""
        module = TranslatorFactory.create(cfg)
        module.validate()

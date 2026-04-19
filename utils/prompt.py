"""Backward-compatible prompt helpers."""

from __future__ import annotations

from prompts.medical_prompt import build_general_prompt, build_medical_prompt

__all__ = ["build_general_prompt", "build_medical_prompt"]
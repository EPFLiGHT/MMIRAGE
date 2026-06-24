"""Shared text normalization for dedup passes."""

from __future__ import annotations


def normalize(text: str) -> str:
    return " ".join(text.lower().split())

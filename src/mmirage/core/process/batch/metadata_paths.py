"""Helpers for resolving batch metadata receipt paths."""

from __future__ import annotations

import glob
from typing import List, Sequence

_METADATA_SUFFIXES = ("text", "multimodal")


def _base_path_to_patterns(base_path: str) -> List[str]:
    trimmed = base_path[:-6] if base_path.endswith(".jsonl") else base_path
    return [f"{trimmed}.{suffix}.*.jsonl" for suffix in _METADATA_SUFFIXES]


def resolve_metadata_paths_from_config(metadata_output_paths: Sequence[str]) -> List[str]:
    """Return metadata receipt paths for config-provided base paths.

    Submission writes suffixed receipts using .text.<run>.jsonl and
    .multimodal.<run>.jsonl, so we expand base paths into matching globs.
    """
    patterns: List[str] = []
    resolved: List[str] = []

    for base_path in metadata_output_paths:
        for pattern in _base_path_to_patterns(base_path):
            patterns.append(pattern)
            matches = sorted(glob.glob(pattern))
            if matches:
                resolved.extend(matches)

    resolved = list(dict.fromkeys(resolved))
    if not resolved:
        pattern_list = ", ".join(patterns) if patterns else "<none>"
        raise ValueError(
            "No metadata receipts matched config metadata_output_path patterns. "
            f"Tried: {pattern_list}. Expected suffixed files like "
            "'<base>.text.<run>.jsonl' or '<base>.multimodal.<run>.jsonl'."
        )

    return resolved

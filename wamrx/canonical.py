"""Canonical serialization and hashing shared by every milestone-1 component."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_json(value: Any) -> str:
    """Return the one JSON representation used for hashes and durable records."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require_string_map(value: Mapping[str, Any], label: str) -> dict[str, str]:
    result = {str(key): str(item) for key, item in value.items()}
    if not result or any(not key or not item for key, item in result.items()):
        raise ValueError(f"{label} must be a non-empty map of non-empty strings")
    return dict(sorted(result.items()))

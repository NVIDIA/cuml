# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonicalize JSON-compatible values and compute content hashes."""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from typing import Any, Mapping

MAX_SAFE_INTEGER = 2**53 - 1


class CanonicalizationError(ValueError):
    """Indicate a value that cannot be represented as canonical JSON."""

    pass


def _utf16_key(value: str) -> bytes:
    """Encode a string as a validated UTF-16 sorting key."""
    try:
        return value.encode("utf-16-be")
    except UnicodeEncodeError as exc:
        raise CanonicalizationError(
            "JSON strings must not contain lone surrogates"
        ) from exc


def _quote(value: str) -> str:
    """Validate and quote a JSON string."""
    _utf16_key(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _float_text(value: float) -> str:
    """Render a finite float using canonical JSON number formatting."""
    if not math.isfinite(value):
        raise CanonicalizationError(
            "RFC 8785 does not permit non-finite numbers"
        )
    if value == 0:
        return "0"
    raw = repr(value).lower()
    if "e" not in raw:
        return raw.removesuffix(".0")
    coefficient, exponent_text = raw.split("e", 1)
    exponent = int(exponent_text)
    negative = coefficient.startswith("-")
    coefficient = coefficient.removeprefix("-")
    whole, _, fraction = coefficient.partition(".")
    digits = whole + fraction
    decimal_position = len(whole) + exponent
    if -6 < decimal_position <= 21:
        if decimal_position <= 0:
            rendered = "0." + "0" * (-decimal_position) + digits
        elif decimal_position >= len(digits):
            rendered = digits + "0" * (decimal_position - len(digits))
        else:
            rendered = (
                digits[:decimal_position] + "." + digits[decimal_position:]
            )
        return ("-" if negative else "") + rendered
    mantissa = digits[0]
    if len(digits) > 1:
        mantissa += "." + digits[1:]
    normalized_exponent = decimal_position - 1
    sign = "+" if normalized_exponent >= 0 else ""
    return (
        ("-" if negative else "")
        + mantissa
        + "e"
        + sign
        + str(normalized_exponent)
    )


def canonical_json(value: Any) -> str:
    """Serialize a value using canonical JSON ordering and formatting.

    Parameters
    ----------
    value : Any
        JSON-compatible value to canonicalize.
    """
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise CanonicalizationError(
                f"integer {value} is outside the interoperable IEEE-754 range"
            )
        return str(value)
    if isinstance(value, float):
        return _float_text(value)
    if isinstance(value, Decimal):
        return _float_text(float(value))
    if isinstance(value, str):
        return _quote(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, Mapping):
        keys = list(value)
        if not all(isinstance(key, str) for key in keys):
            raise CanonicalizationError("JSON object keys must be strings")
        keys.sort(key=_utf16_key)
        return (
            "{"
            + ",".join(
                _quote(key) + ":" + canonical_json(value[key]) for key in keys
            )
            + "}"
        )
    raise CanonicalizationError(
        f"unsupported JSON value {type(value).__name__}"
    )


def content_hash(value: Any) -> str:
    """Return an algorithm-prefixed SHA-256 digest of canonical JSON.

    Parameters
    ----------
    value : Any
        JSON-compatible value to hash.
    """
    payload = canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()

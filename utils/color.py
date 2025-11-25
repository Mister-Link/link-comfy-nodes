from __future__ import annotations

import re
from typing import Iterable, Tuple

HEX_PATTERN = re.compile(r"^#?[0-9a-fA-F]{1,6}$")


def parse_hex_color(
    hex_color: str, fallback: Iterable[int] = (0, 0, 0)
) -> Tuple[int, int, int]:
    """Convert a hex string into an RGB tuple (0-255)."""
    hex_color = (hex_color or "").strip()
    if not HEX_PATTERN.match(hex_color):
        return tuple(fallback)  # type: ignore[return-value]

    normalized = hex_color[1:] if hex_color.startswith("#") else hex_color
    padded = normalized.zfill(6)
    try:
        r = int(padded[0:2], 16)
        g = int(padded[2:4], 16)
        b = int(padded[4:6], 16)
        return (r, g, b)
    except ValueError:
        return tuple(fallback)  # type: ignore[return-value]


def rgb_to_int(r: int, g: int, b: int) -> int:
    """Convert an RGB triplet into a 24-bit integer."""
    return (int(r) << 16) | (int(g) << 8) | int(b)


def int_to_rgb(value_int: int) -> Tuple[int, int, int]:
    """Convert a 24-bit integer into an RGB tuple."""
    value_int = ensure_24bit(value_int)
    r = (value_int >> 16) & 0xFF
    g = (value_int >> 8) & 0xFF
    b = value_int & 0xFF
    return (r, g, b)


def ensure_24bit(value_int: int) -> int:
    """Validate and clamp a 24-bit integer."""
    if not 0 <= value_int <= 0xFFFFFF:
        raise ValueError("Color value must be within 24-bit range (0-16777215)")
    return value_int


def format_color_outputs(value_int: int) -> Tuple[int, str, str]:
    """Return integer, hex string, and human-readable RGB string."""
    value_int = ensure_24bit(value_int)
    r, g, b = int_to_rgb(value_int)
    return value_int, f"#{value_int:06X}", f"{r}, {g}, {b}"


def parse_color_value(value: str | int) -> Tuple[int, str, str]:
    """Parse user-supplied color into standard outputs."""
    value_str = str(value).strip()
    try:
        if HEX_PATTERN.match(value_str):
            normalized = value_str[1:] if value_str.startswith("#") else value_str
            value_int = int(normalized, 16)
        else:
            value_int = int(value_str)
        ensure_24bit(value_int)
    except Exception:
        return 0, "Invalid", "0, 0, 0"

    return format_color_outputs(value_int)


def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Convert RGB (0-255) to HSV (H: 0-360, S/V: 0-1)."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c

    if max_c == min_c:
        h = 0.0
    elif max_c == r:
        h = (60.0 * ((g - b) / diff) + 360.0) % 360.0
    elif max_c == g:
        h = (60.0 * ((b - r) / diff) + 120.0) % 360.0
    else:
        h = (60.0 * ((r - g) / diff) + 240.0) % 360.0

    s = 0.0 if max_c == 0 else diff / max_c
    v = max_c

    return h, s, v

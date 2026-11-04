"""Images for multimodal tasks: encoded image bytes as data URIs."""
from __future__ import annotations

import base64

# Leading bytes of the image formats the benchmark datasets store.
_SIGNATURES = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"),
               (b"GIF87a", "image/gif"), (b"GIF89a", "image/gif"))


def image_mime_type(data: bytes) -> str:
    """The MIME type of encoded image ``data`` (PNG, JPEG, GIF or WebP), from its signature."""
    for signature, mime in _SIGNATURES:
        if data.startswith(signature):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError(f"unrecognised image format (leading bytes {data[:12]!r})")


def image_data_uri(data: bytes) -> str:
    """``data`` as a ``data:`` URI, unchanged (no re-encoding)."""
    return f"data:{image_mime_type(data)};base64,{base64.b64encode(data).decode('ascii')}"

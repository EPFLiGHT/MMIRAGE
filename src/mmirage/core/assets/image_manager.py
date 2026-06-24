"""Canonical image resolution for local/remote/inline inputs."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageOps


@dataclass(frozen=True)
class ResolvedImage:
    """Canonical image descriptor returned by ImageAssetManager."""

    source_type: str
    source_id: str
    canonical_path: str
    width: int
    height: int
    mode: str
    sha256: str


class ImageAssetError(RuntimeError):
    """Base class for image asset errors."""


class ImageResolveError(ImageAssetError):
    """Raised when image source cannot be resolved."""


class ImageDecodeError(ImageAssetError):
    """Raised when image bytes cannot be decoded by Pillow."""


class ImageDownloadError(ImageAssetError):
    """Raised when remote image download fails."""


class ImageTooLargeError(ImageAssetError):
    """Raised when downloaded payload exceeds max size."""


class ImageAssetManager:
    """Resolve and canonicalize image inputs."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        apply_exif_orientation: bool = True,
        convert_mode: Optional[str] = "RGB",
        max_side: Optional[int] = None,
        canonical_format: str = "png",
        jpeg_quality: int = 95,
        remote_timeout_s: int = 20,
        remote_max_bytes: int = 20_000_000,
        remote_user_agent: str = "mmirage/0.x",
    ) -> None:
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir)) if root_dir else None
        self.apply_exif_orientation = apply_exif_orientation
        self.convert_mode = convert_mode
        self.max_side = max_side
        self.canonical_format = canonical_format.lower().strip()
        if self.canonical_format == "jpg":
            self.canonical_format = "jpeg"
        self.jpeg_quality = jpeg_quality
        self.remote_timeout_s = remote_timeout_s
        self.remote_max_bytes = remote_max_bytes
        self.remote_user_agent = remote_user_agent

        if self.canonical_format not in {"png", "jpeg"}:
            raise ValueError(f"Invalid canonical format: {self.canonical_format!r}")
        if self.max_side is not None and self.max_side <= 0:
            raise ValueError("max_side must be > 0 when set")

        self._stats: Dict[str, int] = {
            "downloads": 0,
            "resolved": 0,
        }

    def resolve_image(self, value: Any, root_dir: Optional[str] = None) -> ResolvedImage:
        """Resolve image from local path, URL, PIL image, or raw bytes."""
        source_type, source_id, source_bytes = self._resolve_source(value, root_dir=root_dir)
        canonical_bytes, width, height, mode, sha256 = self._canonicalize(source_bytes, source_id)
        resolved = self._write_temp(source_type, source_id, canonical_bytes, width, height, mode, sha256)

        self._stats["resolved"] += 1
        return resolved

    def get_stats(self) -> Dict[str, int]:
        """Get in-process image resolution statistics."""
        return dict(self._stats)

    def _resolve_source(self, value: Any, root_dir: Optional[str] = None) -> tuple[str, str, bytes]:
        if isinstance(value, Image.Image):
            source_id = "inline:PIL"
            source_bytes = self._pil_to_png_bytes(value)
            return "inline", source_id, source_bytes

        if isinstance(value, (bytes, bytearray)):
            source_bytes = bytes(value)
            source_id = f"inline:bytes:{hashlib.sha256(source_bytes).hexdigest()}"
            return "inline", source_id, source_bytes

        if not isinstance(value, str):
            raise ImageResolveError(
                f"Unsupported image value type: {type(value).__name__}. Expected str, bytes, or PIL.Image.Image"
            )

        value = value.removeprefix("file://")

        if self._is_remote_url(value):
            return "remote", value, self._download_remote(value)

        local_path = self._resolve_local_path(value, root_dir=root_dir)
        try:
            with open(local_path, "rb") as f:
                return "local", local_path, f.read()
        except OSError as e:
            raise ImageResolveError(f"Failed to read local image file '{local_path}': {e}") from e

    def _resolve_local_path(self, value: str, root_dir: Optional[str] = None) -> str:
        path = value
        if not os.path.isabs(path):
            base_dir = self.root_dir or root_dir
            if base_dir is None:
                raise ImageResolveError(
                    f"Relative image path '{value}' cannot be resolved without assets.images.root_dir"
                )
            path = os.path.join(base_dir, value)

        abs_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(abs_path):
            raise ImageResolveError(f"Local image path does not exist: {abs_path}")
        if not os.path.isfile(abs_path):
            raise ImageResolveError(f"Local image path exists but is not a file: {abs_path}")
        return abs_path

    def _download_remote(self, url: str) -> bytes:
        req = Request(url=url, headers={"User-Agent": self.remote_user_agent})
        try:
            with urlopen(req, timeout=self.remote_timeout_s) as resp:
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self.remote_max_bytes:
                        raise ImageTooLargeError(
                            f"Downloaded payload exceeds remote.max_bytes ({self.remote_max_bytes}) for URL '{url}'"
                        )
                    chunks.append(chunk)
        except ImageTooLargeError:
            raise
        except Exception as e:
            raise ImageDownloadError(f"Failed to download image from '{url}': {e}") from e

        self._stats["downloads"] += 1
        return b"".join(chunks)

    def _canonicalize(self, source_bytes: bytes, source_id: str) -> tuple[bytes, int, int, str, str]:
        try:
            with Image.open(io.BytesIO(source_bytes)) as img:
                image = img.copy()
        except Exception as e:
            raise ImageDecodeError(f"Cannot decode image from '{source_id}': {e}") from e

        if self.apply_exif_orientation:
            image = ImageOps.exif_transpose(image)

        if self.convert_mode is not None and image.mode != self.convert_mode:
            image = image.convert(self.convert_mode)

        if self.max_side is not None:
            width, height = image.size
            longest = max(width, height)
            if longest > self.max_side:
                scale = self.max_side / float(longest)
                new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

        out = io.BytesIO()
        save_format = "PNG" if self.canonical_format == "png" else "JPEG"
        save_kwargs: Dict[str, Any] = {}
        if save_format == "JPEG":
            save_kwargs["quality"] = self.jpeg_quality
            save_kwargs["optimize"] = True

            # JPEG cannot store alpha channels.
            if image.mode in {"RGBA", "LA"}:
                image = image.convert("RGB")

        image.save(out, format=save_format, **save_kwargs)
        canonical_bytes = out.getvalue()
        width, height = image.size
        mode = image.mode
        sha256 = hashlib.sha256(canonical_bytes).hexdigest()
        return canonical_bytes, width, height, mode, sha256

    def _write_temp(
        self,
        source_type: str,
        source_id: str,
        canonical_bytes: bytes,
        width: int,
        height: int,
        mode: str,
        sha256: str,
    ) -> ResolvedImage:
        ext = ".png" if self.canonical_format == "png" else ".jpg"
        temp_dir = tempfile.mkdtemp(prefix="mmirage-image-")
        temp_path = os.path.join(temp_dir, f"{sha256}{ext}")
        with open(temp_path, "wb") as f:
            f.write(canonical_bytes)

        return ResolvedImage(
            source_type=source_type,
            source_id=source_id,
            canonical_path=temp_path,
            width=width,
            height=height,
            mode=mode,
            sha256=sha256,
        )

    @staticmethod
    def _is_remote_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _pil_to_png_bytes(img: Image.Image) -> bytes:
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from mmirage.core.assets.image_manager import ImageAssetManager, ImageResolveError


@contextmanager
def _serve_directory(directory: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format, *args):  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _write_image(path: Path, size=(64, 32), mode="RGBA"):
    img = Image.new(mode, size=size, color=(255, 0, 0, 128) if "A" in mode else 128)
    img.save(path)


def test_relative_path_requires_root_dir(tmp_path: Path):
    image_path = tmp_path / "image.png"
    _write_image(image_path)

    manager = ImageAssetManager()

    with pytest.raises(ImageResolveError):
        manager.resolve_image("image.png")


def test_relative_path_resolves_and_canonicalizes(tmp_path: Path):
    image_path = tmp_path / "image.png"
    _write_image(image_path, size=(200, 100), mode="RGBA")

    manager = ImageAssetManager(root_dir=str(tmp_path), max_side=100)
    resolved = manager.resolve_image("image.png")

    assert os.path.exists(resolved.canonical_path)
    assert resolved.width == 100
    assert resolved.height == 50
    assert resolved.mode == "RGB"


def test_separate_resolves_create_canonical_files(tmp_path: Path):
    image_path = tmp_path / "image.png"
    _write_image(image_path, mode="RGB")

    manager = ImageAssetManager(root_dir=str(tmp_path))

    first = manager.resolve_image("image.png")
    second = manager.resolve_image("image.png")

    assert first.canonical_path != second.canonical_path
    assert os.path.exists(first.canonical_path)
    assert os.path.exists(second.canonical_path)
    stats = manager.get_stats()
    assert stats["resolved"] == 2


def test_remote_url_resolution(tmp_path: Path):
    image_path = tmp_path / "remote.png"
    _write_image(image_path, mode="RGB")

    with _serve_directory(tmp_path) as base_url:
        manager = ImageAssetManager()
        resolved = manager.resolve_image(f"{base_url}/remote.png")

    assert os.path.exists(resolved.canonical_path)
    assert resolved.source_type == "remote"


def test_accepts_jpg_alias_for_canonical_format(tmp_path: Path):
    image_path = tmp_path / "image.png"
    _write_image(image_path, mode="RGB")

    manager = ImageAssetManager(root_dir=str(tmp_path), canonical_format="jpg")
    resolved = manager.resolve_image("image.png")

    assert resolved.canonical_path.endswith(".jpg")

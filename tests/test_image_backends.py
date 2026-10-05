"""The cloud (Gemini) and remote (HTTP) image backends, with their transports faked."""

import io
from types import SimpleNamespace

import pytest
from PIL import Image

from generated_adventures.core.config import settings
from generated_adventures.core.image_backends import (
    GeminiImageGenerator,
    HttpImageGenerator,
)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "player.png"
    path.write_bytes(_png())
    return path


# --- Gemini -------------------------------------------------------------------


@pytest.fixture
def gemini(monkeypatch):
    """A Gemini backend whose client records each request and returns one image."""
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        part = SimpleNamespace(inline_data=SimpleNamespace(data=_png()))
        return SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))]
        )

    backend = GeminiImageGenerator.__new__(GeminiImageGenerator)
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    monkeypatch.setattr(backend, "client", fake_client, raising=False)
    return backend, calls


def test_gemini_portrait_never_uploads_photos_by_default(
    gemini, photo, tmp_path, monkeypatch
):
    backend, calls = gemini
    monkeypatch.setattr(settings, "ALLOW_CLOUD_IMAGE_UPLOAD", False)
    out = tmp_path / "portrait.png"

    assert (
        backend.generate_character_image("A dwarf", out, reference_images=[photo])
        == out
    )

    (call,) = calls
    assert call["contents"][0].startswith("A dwarf")
    assert (
        len(call["contents"]) == 1
    )  # the prompt only: the photo stayed on this machine
    assert out.exists()


def test_gemini_scene_uploads_references_when_allowed(
    gemini, photo, tmp_path, monkeypatch
):
    backend, calls = gemini
    monkeypatch.setattr(settings, "ALLOW_CLOUD_IMAGE_UPLOAD", True)
    out = tmp_path / "scene.png"

    assert (
        backend.generate_scene_image("A battle", out, reference_images=[photo, photo])
        == out
    )

    (call,) = calls
    assert len(call["contents"]) == 3  # prompt + two references
    assert call["config"].response_modalities == ["image"]


def test_gemini_failure_returns_none(gemini, tmp_path):
    backend, _ = gemini

    def boom(**kwargs):
        raise RuntimeError("quota exceeded")

    backend.client.models.generate_content = boom

    assert backend.generate_scene_image("A battle", tmp_path / "x.png") is None


# --- HTTP ---------------------------------------------------------------------


@pytest.fixture
def http(monkeypatch):
    """An HTTP backend whose transport records each request and returns one image."""
    calls = []

    def post(url, *, data, files, headers, timeout):
        calls.append({"url": url, "data": data, "fields": [f for f, _ in files or []]})
        return SimpleNamespace(status_code=200, content=_png(), text="")

    backend = HttpImageGenerator("http://gpu-box:8000/", api_key="secret")
    monkeypatch.setattr(backend, "_httpx", SimpleNamespace(post=post))
    return backend, calls


def test_http_portrait_sends_only_the_first_reference(http, photo, tmp_path):
    backend, calls = http
    out = tmp_path / "portrait.png"

    result = backend.generate_character_image(
        "A dwarf", out, reference_images=[photo, photo], art_style="painterly_hero"
    )

    assert result == out and out.exists()
    (call,) = calls
    assert call["url"] == "http://gpu-box:8000/generate/character"
    assert call["data"] == {"prompt": "A dwarf", "art_style": "painterly_hero"}
    assert call["fields"] == ["photo"]


def test_http_scene_sends_every_reference(http, photo, tmp_path):
    backend, calls = http

    backend.generate_scene_image(
        "A battle", tmp_path / "scene.png", reference_images=[photo, photo]
    )

    (call,) = calls
    assert call["url"] == "http://gpu-box:8000/generate/scene"
    assert call["fields"] == ["references", "references"]


def test_http_server_error_returns_none(http, tmp_path, monkeypatch):
    backend, _ = http
    error = SimpleNamespace(status_code=500, content=b"", text="CUDA out of memory")
    monkeypatch.setattr(backend, "_httpx", SimpleNamespace(post=lambda *a, **k: error))

    assert backend.generate_scene_image("A battle", tmp_path / "scene.png") is None


def test_missing_reference_files_are_skipped(http, tmp_path):
    backend, calls = http

    backend.generate_scene_image(
        "A battle", tmp_path / "scene.png", reference_images=[tmp_path / "gone.png"]
    )

    assert calls[0]["fields"] == []

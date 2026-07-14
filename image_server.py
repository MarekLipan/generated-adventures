#!/usr/bin/env python3
"""Local image-generation inference service.

Loads a local image backend (FLUX.2 Klein / FLUX.1 Kontext) ONCE at startup,
keeps it warm in GPU memory, and serves generation over HTTP. The game (or any
device on the same LAN) becomes a thin client via IMAGE_PROVIDER="http".

Run it:
    uv run python image_server.py

Key design points for a single-GPU box:
  * ONE model instance, ONE worker. Never run with multiple uvicorn workers — each
    would load its own copy of the model and fight over the GPU.
  * A global lock serializes generation: the GPU can only do one image at a time,
    so concurrent requests queue rather than overlap (which would OOM/crash).
  * Startup warmup runs one throwaway generation so the first real request doesn't
    pay the one-time quantization / CUDA-autotune cost.

LAN access: IMAGE_SERVER_HOST defaults to 0.0.0.0 (all interfaces). Point a client
at http://<this-machine-ip>:<port>. See README_image_server.md for finding your
address and opening the Windows firewall port.
"""

import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

from core.config import settings
from core.image_backends import _local_backend_for_server

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("image_server")

# Holds the warm model and the lock that serializes GPU access.
STATE: dict = {"generator": None, "lock": None}


def _check_api_key(x_api_key: Optional[str]) -> None:
    """Enforce the shared secret if one is configured."""
    expected = settings.IMAGE_SERVER_API_KEY
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading image backend (this may take ~30-60s)...")
    generator = _local_backend_for_server()
    STATE["generator"] = generator
    STATE["lock"] = asyncio.Lock()

    # Warmup: trigger quantization / kernel autotune so request #1 is fast.
    try:
        logger.info("Warming up with a throwaway generation...")
        with tempfile.TemporaryDirectory() as td:
            await asyncio.to_thread(
                generator.generate_character_image,
                "a small grey stone, plain background",
                Path(td) / "warmup.png",
            )
        logger.info("✓ Warmup complete — server ready")
    except Exception as e:  # non-fatal: serve anyway, first request just pays the cost
        logger.warning(f"Warmup failed ({e}); serving without warmup")

    yield
    STATE.clear()


app = FastAPI(title="Generated Adventures Image Server", lifespan=lifespan)


@app.get("/health")
async def health():
    gen = STATE.get("generator")
    return {
        "status": "ready" if gen is not None else "loading",
        "backend": type(gen).__name__ if gen else None,
        "provider": settings.IMAGE_PROVIDER,
    }


async def _run_generation(call) -> bytes:
    """Serialize GPU access, run a blocking `call(out_path)` in a thread, return PNG bytes.

    `call` is a closure that takes the output Path and invokes the backend with the
    right arguments, returning the saved path (or None on failure).
    """
    gen = STATE.get("generator")
    lock = STATE.get("lock")
    if gen is None or lock is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    async with lock:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.png"
            result = await asyncio.to_thread(call, out)
            if result is None or not out.exists():
                raise HTTPException(status_code=500, detail="Generation failed")
            return out.read_bytes()


@app.post("/generate/character")
async def generate_character(
    prompt: str = Form(...),
    art_style: Optional[str] = Form(default=None),
    photo: Optional[UploadFile] = File(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_api_key(x_api_key)
    gen = STATE["generator"]

    # An optional profile photo drives the hero's likeness; persist it so the
    # backend (which takes file paths) can use it as a reference.
    with tempfile.TemporaryDirectory() as td:
        refs: Optional[List[Path]] = None
        if photo is not None:
            data = await photo.read()
            if data:
                p = Path(td) / f"photo_{Path(photo.filename or 'photo').name}"
                p.write_bytes(data)
                refs = [p]

        png = await _run_generation(
            lambda out: gen.generate_character_image(
                prompt, out, reference_images=refs, art_style=art_style
            )
        )
    return Response(content=png, media_type="image/png")


@app.post("/generate/scene")
async def generate_scene(
    prompt: str = Form(...),
    art_style: Optional[str] = Form(default=None),
    references: List[UploadFile] = File(default=[]),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_api_key(x_api_key)
    gen = STATE["generator"]

    # Persist uploaded reference images to a temp dir; the backend takes file paths.
    with tempfile.TemporaryDirectory() as td:
        ref_paths: List[Path] = []
        for i, uf in enumerate(references or []):
            data = await uf.read()
            if not data:
                continue
            p = Path(td) / f"ref_{i}_{Path(uf.filename or 'ref').name}"
            p.write_bytes(data)
            ref_paths.append(p)

        png = await _run_generation(
            lambda out: gen.generate_scene_image(
                prompt, out, reference_images=ref_paths or None, art_style=art_style
            )
        )
    return Response(content=png, media_type="image/png")


def main():
    import uvicorn

    logger.info(
        f"Starting image server on {settings.IMAGE_SERVER_HOST}:{settings.IMAGE_SERVER_PORT} "
        f"(hosting provider '{settings.IMAGE_PROVIDER}')"
    )
    # workers=1 is critical: a single model instance owns the GPU.
    uvicorn.run(
        app,
        host=settings.IMAGE_SERVER_HOST,
        port=settings.IMAGE_SERVER_PORT,
        workers=1,
    )


if __name__ == "__main__":
    main()

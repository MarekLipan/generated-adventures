# Scene Narration (Text-to-Speech)

The game narrates each scene with a text-to-speech backend, selected by
`TTS_PROVIDER` in `.env` / `core/config.py`:

| Provider | Cost | Runs on | Notes |
|----------|------|---------|-------|
| `kokoro` *(default)* | Free, local | **CPU** | Kokoro-82M via onnxruntime. ~5× realtime. Off-GPU, so it narrates in parallel with local image generation. |
| `gemini` | Paid, cloud | API | Google Gemini TTS. Expressive; requires `GOOGLE_API_KEY`. |
| `none` | — | — | Disables narration. |

Backends live in [`core/tts_backends.py`](core/tts_backends.py) and all emit a
24 kHz / 16-bit / mono WAV, which the web UI plays back.

## Kokoro setup (default)

`kokoro-onnx` is a project dependency (installed via `uv sync`). It bundles its
own espeak-ng, so there is **no system install** needed.

You do need the model files once (~340 MB total), placed in `models/kokoro/`:

```bash
mkdir -p models/kokoro && cd models/kokoro
curl -L -o kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Paths are configurable via `KOKORO_MODEL_PATH` / `KOKORO_VOICES_PATH`.

### Choosing the narrator voice

Set `KOKORO_VOICE` (and match `KOKORO_LANG` to the voice's region). The game
default is **`bm_lewis`** (British male, deep/gravelly DM read).

| Voice | Region | `KOKORO_LANG` |
|-------|--------|---------------|
| `bm_lewis` *(default)* | British male | `en-gb` |
| `bm_george` | British male | `en-gb` |
| `bf_emma` | British female | `en-gb` |
| `am_michael` | American male | `en-us` |
| `am_adam` | American male | `en-us` |
| `af_heart` | American female | `en-us` |

`KOKORO_SPEED` (default `1.0`) tweaks pace; slightly below 1.0 reads more
deliberately.

## Notes

- Narration is **optional**: if the model files are missing or a backend fails
  to initialize, the game logs a warning and continues silently (no audio).
- The model loads once (lazy singleton in `core/generator.py`) and stays warm.
- On a local-GPU image setup (`IMAGE_PROVIDER=flux-klein`), Kokoro deliberately
  stays on the CPU so it doesn't contend with FLUX for VRAM.

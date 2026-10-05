# generated-adventures
A Python app for generating short D&D like adventures with generated story, voice narration and imagery using Multimodal LLMs.

## Image generation

Images come from **FLUX.2 Klein** running locally, or from Gemini in the cloud.
Choose with `IMAGE_PROVIDER` in `.env` (start from `.env.example`):

| Provider | Runs on | Notes |
|---|---|---|
| `flux-klein` (default) | Your NVIDIA GPU or Apple Silicon | Free. Takes multiple reference images natively, so characters stay consistent across scenes. |
| `gemini` | Google's API | No GPU needed, but image generation requires a billing-enabled key. |
| `http` | Another machine running `image_server.py` | Lets a laptop use a desktop GPU over the LAN. See [README_image_server.md](README_image_server.md). |

**First run with `flux-klein`:** the model (~32 GB) is gated on Hugging Face.
Accept its license on the [model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B),
set `HUGGING_FACE_HUB_TOKEN`, and set `IMAGE_HF_OFFLINE=false` until the download
finishes. After that the app loads it from the local cache without network access.

### Hardware

- **NVIDIA GPU:** nothing to configure. Klein loads with bitsandbytes NF4
  quantization, about 12–20 s per image on a 10 GB RTX 3080.
- **Apple Silicon (32 GB):** set `FLUX_KLEIN_GGUF_TRANSFORMER` as shown in
  `.env.example`. It loads a 4 GB quantized transformer so the whole pipeline fits
  in memory: about 2 minutes per image on an M1 Max, versus about 9 without it.

### Checking speed and quality

```bash
uv run python scripts/benchmark_images.py
```

Renders portraits, a one-character scene and a multi-character scene with your
configured backend, and prints how long each step took. Output goes to `test_images/`.

### How to Run
1. `uv run uvicorn webapp.main:app --reload` for playing the game.
2. `SHOW_DM_NOTES=1 uv run uvicorn webapp.main:app --reload` for playing the game with DM notes visible.

### App Flow
1. User is prompted to enter a how many players will be playing the adventure.
2. User is prompted to select out of 3 generated fantasy adventure scenarios.
3. The story of the adventure is generated including the setting, plot, main quest, important NPCs and locations.
4. User is prompted to select a character for each player out of 6 generated characters.
5. Party overview is displayed with selected characters.
6. The opening scene is generated and narrated.
7. Image is generated to represent the opening scene.
8. One of the characters is prompted for an action.
9. The action is processed and the next scene is generated and narrated.
10. The process continues until the adventure is completed.

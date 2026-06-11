# generated-adventures
A Python app for generating short D&D like adventures with generated story, voice narration and imagery using Multimodal LLMs.

## Image Generation Backends

This project supports **two image generation backends** with easy switching via environment variables:

- **🖥️ FLUX.1 Kontext [dev]** (default): Local, free, supports reference images for character consistency
- **🌐 Google Gemini**: Cloud-based, supports reference images for character consistency

### Quick Start

#### Using Gemini (Cloud API)
```bash
# .env file
IMAGE_PROVIDER=gemini
GOOGLE_API_KEY=your_api_key_here
```

#### Using FLUX.1 Kontext [dev] (Local - Free, Default)
```bash
# 1. Install local image generation dependencies
uv pip install -e ".[local-image]"

# 2. Configure .env
IMAGE_PROVIDER=flux-kontext
FLUX_KONTEXT_MODEL=black-forest-labs/FLUX.1-Kontext-dev
FLUX_KONTEXT_GUIDANCE_SCALE=2.5
IMAGE_ENABLE_CPU_OFFLOAD=true
IMAGE_DEVICE=auto  # auto, mps (Mac), cuda (NVIDIA), cpu
IMAGE_NUM_INFERENCE_STEPS=30  # Lower=faster, Higher=better quality

# 3. First run will download ~24GB model (one-time)
```

**⚠️ Note**: FLUX.1 Kontext [dev] requires accepting the model license on Hugging Face and logging in with `huggingface-cli login` before first use.

### Performance Notes

**Mac (Apple Silicon)**:
- FLUX.1 Kontext [dev]: ~2-4 min/image (M1/M2 with 16GB RAM, CPU offload enabled)
- Uses Metal Performance Shaders (MPS) automatically

**NVIDIA GPU**:
- FLUX.1 Kontext [dev]: ~30-90 sec/image (RTX 3080+)
- Requires CUDA support

**CPU Only** (not recommended):
- Very slow, 10-20 min/image

### Character Consistency

**Reference Image Support** (maintains character appearance across scenes):
- ✅ **FLUX.1 Kontext [dev]**: Local and free
- ✅ **Gemini**: Cloud-based, costs per image

For best results:
- **Character portraits**: FLUX.1 Kontext [dev]
- **Scene images with character consistency**: FLUX.1 Kontext [dev] (local) or Gemini (cloud)

### Testing Image Quality

Use the included test script to preview image quality before generating full adventures:

```bash
python test_image_generation.py
```

This will generate sample character portraits and scenes using your configured backend, helping you compare quality and choose the best option for your needs.

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

# Running FLUX.2 Klein on a 32 GB Mac

The game renders a portrait per hero and an illustration per scene with FLUX.2
Klein 9B. On an NVIDIA GPU that takes seconds. On an Apple Silicon Mac with 32 GB of
unified memory it first took **8 to 10 minutes per image**, and got slower the longer the
game ran. This page shows how that went to about two minutes.

All measurements: an Apple Silicon Mac with 32 GB of unified memory, PyTorch on
Metal (MPS), 4 denoising steps. Absolute times vary by chip; the ratios are the
point.

## The problem: 34 GB of weights in 32 GB of memory

Klein has two large parts: a 9B transformer (17 GB in bfloat16) and a Qwen3-8B
text encoder (15 GB). Together they don't fit, so the pipeline used model-level
CPU offload, moving each module onto the "GPU" when needed. On unified memory the
CPU and GPU share the same RAM, so each move was a copy into memory that was
already full. macOS paged to disk:

| Configuration | First image | Second image |
|---|---|---|
| bfloat16, 768×1024, CPU offload | 476 s | 630 s |

The second image being *slower* was the giveaway: memory pressure was building
up (swap reached 29 GB), so every image paged more than the one before.

## What didn't work

- **fp8 weights.** Black Forest Labs publishes a 9.4 GB fp8 transformer, but
  PyTorch can't even allocate a float8 tensor on MPS. It would have to be
  converted back to bfloat16 on load, which brings the 17 GB back.
- **FLUX.1 Kontext with a GGUF transformer**, the project's previous image model:
  about 56 to 90 s per denoising step on this machine, so 14 minutes or more per
  image. Stopped after 6 of 15 steps.

## What did

**1. A smaller canvas on MPS.** Scenes render at 768×576 and portraits at 576×768
instead of 1024×768 / 768×1024: 56% of the tokens. This helped (476 s → 375 s)
but couldn't fix the problem, because about 114 s of every image was spent
outside the denoising loop, paging the text encoder in and out. That cost doesn't
depend on resolution.

**2. A quantized transformer that fits, so nothing is offloaded.** The community
GGUF builds quantize only the transformer, which is enough: a Q2_K transformer is
4 GB, so the whole pipeline is about 19 GB and stays in memory. Offloading
switches off entirely and the paging stops:

| Configuration | First image | Second image |
|---|---|---|
| bfloat16, 576×768, CPU offload | 375 s | 503 s |
| **GGUF Q2_K, 576×768, no offload** | **116 s** | **128 s** |

That's 4.5× faster than where it started, and the drift between images dropped
from about +33% to +11%. Q2_K is aggressive quantization, but portraits and scenes
stayed clean, and prompt adherence was no worse. Q4_K_M (5.9 GB) and Q6_K
(7.9 GB) also fit, if you'd rather trade some speed for quality.

**3. Single-pass multi-character scenes on MPS.** To stop characters' looks
bleeding into each other, multi-character scenes are rendered in passes: a base
render, then one masked repaint per character using only that character's
portrait. On this Mac the base render leaves the memory pool at its limit (36.1
of 36.3 GiB), so building the repaint pipeline ran out of memory on every scene.
MPS now keeps the base render, whose prompt still places each character in its
own spot in the frame. A two-character scene went from 320 s to 227 s with an
identical image (about 53 s of that was the failed attempt; the rest was
run-to-run variation). NVIDIA GPUs keep the repaint passes
(`IMAGE_COMPOSITE_INPAINT_ON_MPS` turns them on for MPS too).

## Current numbers

From `scripts/benchmark_images.py --party 2`:

| Step | Time |
|---|---|
| Load the model | 59 s |
| Portrait | 130–146 s |
| Scene, one character | 225 s |
| Scene, two characters | 242 s |

Scenes take longer than portraits because the reference portraits add to what
the model attends to.

## Turning it on

Add this to `.env`:

```bash
FLUX_KLEIN_GGUF_TRANSFORMER=unsloth/FLUX.2-klein-9B-GGUF:flux-2-klein-9b-Q2_K.gguf
```

and download the file once:

```bash
uv run hf download unsloth/FLUX.2-klein-9B-GGUF flux-2-klein-9b-Q2_K.gguf
```

The text encoder, VAE and tokenizer still come from the main Klein repository,
which also needs to be in the Hugging Face cache (see the README). NVIDIA GPUs
ignore this setting and use the bitsandbytes path.

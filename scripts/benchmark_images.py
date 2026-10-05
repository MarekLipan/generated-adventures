#!/usr/bin/env python3
"""Time and eyeball the configured image backend.

Renders party portraits, a single-character scene and a multi-character scene,
then prints how long each step took. It uses the IMAGE_PROVIDER and device
settings from .env, so the same script benchmarks the Mac (GGUF on MPS) and the
CUDA desktop (bitsandbytes).

    uv run python scripts/benchmark_images.py             # two characters
    uv run python scripts/benchmark_images.py --party 3   # three-character scene

Images are written to test_images/ (gitignored) for side-by-side comparison.
"""

import argparse
import logging
import pathlib
import sys
import time

# Run from a checkout without installing the project.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402
from core.image_backends import get_image_generator  # noqa: E402

PARTY = [
    (
        "Elara Moonwhisper",
        "tall elf mage, long silver hair, piercing blue eyes, midnight blue robes "
        "with glowing silver runes, ornate staff topped with a crackling blue crystal",
        "raises her staff, casting a bolt of blue lightning",
    ),
    (
        "Gareth Ironwall",
        "stocky dwarf warrior, enormous braided red beard, dented iron plate armor, "
        "massive battle-axe resting on his shoulder",
        "charges forward, swinging his battle-axe",
    ),
    (
        "Lyra Swiftarrow",
        "agile human ranger, short auburn hair, sharp green eyes, dark green leather "
        "coat with silver buckles, recurve bow",
        "draws her bow from a crumbling balcony",
    ),
]

SOLO_SCENE = (
    "Elara Moonwhisper stands at the frozen altar of a glacier temple, "
    "blue light spilling from the ice around her"
)
PARTY_SCENE = (
    "The party fights a towering stone golem in a torchlit underground dungeon, "
    "sparks and stone chips flying"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--party", type=int, choices=(1, 2, 3), default=2, help="characters to render"
    )
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("test_images"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    timings: list[tuple[str, float, bool]] = []

    def timed(label, fn):
        start = time.perf_counter()
        result = fn()
        timings.append((label, time.perf_counter() - start, result is not None))
        return result

    generator = timed("load model", get_image_generator)
    party = PARTY[: args.party]

    portraits = []
    for name, look, _ in party:
        path = args.out / f"bench_portrait_{name.split()[0].lower()}.png"
        result = timed(
            f"portrait: {name}",
            lambda n=name, lk=look, p=path: generator.generate_character_image(f"{n}, {lk}", p),
        )
        if result:
            portraits.append(result)

    if portraits:
        timed(
            "scene: 1 character",
            lambda: generator.generate_scene_image(
                SOLO_SCENE, args.out / "bench_scene_solo.png", reference_images=portraits[:1]
            ),
        )
    if len(portraits) > 1:
        timed(
            f"scene: {len(portraits)} characters",
            lambda: generator.generate_scene_image(
                PARTY_SCENE,
                args.out / "bench_scene_party.png",
                reference_images=portraits,
                subject_directions=[f"{name} {action}" for name, _, action in party],
            ),
        )

    device = getattr(generator, "device", "n/a")
    gguf = settings.FLUX_KLEIN_GGUF_TRANSFORMER.split(":")[-1] or "none"
    print(f"\nprovider={settings.IMAGE_PROVIDER}  device={device}  gguf={gguf}")
    for label, seconds, ok in timings:
        print(f"  {label:32s} {seconds:7.1f}s  {'ok' if ok else 'FAILED'}")
    print(f"  output: {args.out}/")
    return 0 if all(ok for _, _, ok in timings) else 1


if __name__ == "__main__":
    raise SystemExit(main())

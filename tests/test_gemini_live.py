"""Live smoke test for the Gemini image backend.

Opt-in: costs a real API call. Run with `pytest -m live` and GOOGLE_API_KEY set.
"""

import pytest
from PIL import Image

from core.config import settings
from core.image_backends import GeminiImageGenerator

pytestmark = pytest.mark.live


@pytest.fixture
def gemini():
    if not settings.GOOGLE_API_KEY:
        pytest.skip("GOOGLE_API_KEY is not set")
    return GeminiImageGenerator(api_key=settings.GOOGLE_API_KEY)


def test_character_portrait(gemini, tmp_path):
    out = tmp_path / "portrait.png"

    result = gemini.generate_character_image(
        "A weathered dwarven blacksmith with a braided grey beard", out
    )

    assert result == out
    with Image.open(out) as image:
        assert min(image.size) >= 256

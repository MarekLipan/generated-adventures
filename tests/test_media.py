"""Saved games store media as URLs; rendering needs the files behind them."""

from generated_adventures.core import media


def test_media_path_maps_saved_urls_to_files():
    url = "/static/scenes/g1/scene_001.png"

    assert media.media_path(url) == media.MEDIA_DIR / "scenes" / "g1" / "scene_001.png"


def test_media_lives_under_the_data_dir():
    from generated_adventures.core.config import settings

    assert media.MEDIA_DIR.is_relative_to(settings.DATA_DIR)
    assert media.SCENE_IMAGE_DIR.parent == media.MEDIA_DIR

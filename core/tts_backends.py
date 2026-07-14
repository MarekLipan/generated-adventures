"""Text-to-speech backends for scene narration.

Mirrors image_backends.py: an abstract TTSGenerator with swappable
implementations selected by settings.TTS_PROVIDER.

- KokoroTTSGenerator: FREE, fully local. Kokoro-82M via onnxruntime on CPU. Fast
  (~5x realtime) and runs entirely off the GPU, so scene narration generates in
  parallel with local image generation without fighting for VRAM.
- GeminiTTSGenerator: Google Gemini TTS (cloud, requires GOOGLE_API_KEY).

All backends emit a 24 kHz / 16-bit / mono WAV, matching what the web UI plays.
"""

import logging
import pathlib
import re
import wave
from abc import ABC, abstractmethod
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)

# All backends produce this WAV format so the player and files stay uniform.
TTS_SAMPLE_RATE = 24000


def _clean_for_tts(text: str) -> str:
    """Strip markdown/formatting so it isn't read aloud as literal symbols.

    Scene text is authored as markdown (bold, italics, headings, quotes). Left in,
    a narrator would voice the asterisks and hashes, so remove the syntax while
    keeping the words and sentence flow.
    """
    # Drop code fences / inline backticks.
    text = text.replace("`", "")
    # Bold/italic emphasis markers.
    text = re.sub(r"(\*\*|\*|__|_)", "", text)
    # Leading heading hashes and blockquote markers, per line.
    text = re.sub(r"^[ \t]*[#>]+[ \t]*", "", text, flags=re.MULTILINE)
    # List bullets at line start.
    text = re.sub(r"^[ \t]*[-•]\s+", "", text, flags=re.MULTILINE)
    # Collapse excess whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _write_wav_int16(path: pathlib.Path, samples, sample_rate: int) -> None:
    """Write a float [-1, 1] numpy array to a 16-bit mono WAV."""
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


class TTSGenerator(ABC):
    """Abstract base class for text-to-speech narration backends."""

    # Whether this backend can voice different segments with different voices.
    # Backends that can't (single-voice) fall back to reading the whole script
    # in one voice via the default synthesize_segments below.
    supports_multivoice: bool = False

    @abstractmethod
    def synthesize(
        self, text: str, output_path: pathlib.Path
    ) -> Optional[pathlib.Path]:
        """Synthesize narration for `text` and save a WAV to `output_path`.

        Returns the path on success, or None on failure.
        """
        pass

    def synthesize_segments(
        self, segments: list, output_path: pathlib.Path
    ) -> Optional[pathlib.Path]:
        """Synthesize an ordered list of (voice_id, text) segments to one WAV.

        Default (single-voice) implementation ignores the per-segment voices and
        reads the concatenated text in this backend's single voice. Multi-voice
        backends override this.
        """
        text = " ".join(t for _voice, t in segments if t and t.strip())
        return self.synthesize(text, output_path)


class KokoroTTSGenerator(TTSGenerator):
    """Local narration via Kokoro-82M (ONNX, CPU). Loads the model once."""

    supports_multivoice = True

    def __init__(self):
        from kokoro_onnx import Kokoro

        model_path = pathlib.Path(settings.KOKORO_MODEL_PATH)
        voices_path = pathlib.Path(settings.KOKORO_VOICES_PATH)
        if not model_path.exists() or not voices_path.exists():
            raise FileNotFoundError(
                f"Kokoro model files not found: {model_path} / {voices_path}. "
                "Download them into models/kokoro/ (see README_tts.md)."
            )

        logger.info(f"Loading Kokoro TTS (voice='{settings.KOKORO_VOICE}')...")
        self.kokoro = Kokoro(str(model_path), str(voices_path))
        self.voice = settings.KOKORO_VOICE
        self.lang = settings.KOKORO_LANG
        self.speed = settings.KOKORO_SPEED
        logger.info("✓ Kokoro TTS ready")

    def synthesize(
        self, text: str, output_path: pathlib.Path
    ) -> Optional[pathlib.Path]:
        try:
            clean = _clean_for_tts(text)
            if not clean:
                logger.warning("Kokoro TTS: empty text after cleanup, skipping")
                return None
            samples, sample_rate = self.kokoro.create(
                clean, voice=self.voice, speed=self.speed, lang=self.lang
            )
            _write_wav_int16(output_path, samples, sample_rate)
            logger.info(f"✓ Kokoro TTS: saved narration to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Kokoro TTS synthesis failed: {e}")
            return None

    def synthesize_segments(
        self, segments: list, output_path: pathlib.Path
    ) -> Optional[pathlib.Path]:
        """Render each (voice_id, text) segment in its own voice and concatenate."""
        import numpy as np

        from .voice_casting import lang_for_voice

        try:
            chunks = []
            sample_rate = TTS_SAMPLE_RATE
            spoken = 0
            for voice, text in segments:
                clean = _clean_for_tts(text)
                if not clean:
                    continue
                samples, sample_rate = self.kokoro.create(
                    clean, voice=voice, speed=self.speed, lang=lang_for_voice(voice)
                )
                chunks.append(np.asarray(samples, dtype=np.float32))
                # Short pause between segments so speaker changes read naturally.
                chunks.append(np.zeros(int(sample_rate * 0.3), dtype=np.float32))
                spoken += 1

            if not chunks:
                logger.warning("Kokoro multi-voice: nothing to synthesize")
                return None

            _write_wav_int16(output_path, np.concatenate(chunks), sample_rate)
            logger.info(
                f"✓ Kokoro TTS: saved {spoken}-segment narration to {output_path}"
            )
            return output_path
        except Exception as e:
            logger.error(f"Kokoro multi-voice synthesis failed: {e}")
            return None


class GeminiTTSGenerator(TTSGenerator):
    """Cloud narration via Google Gemini TTS."""

    def __init__(self, api_key: str):
        from google import genai as genai_client  # type: ignore

        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required when TTS_PROVIDER='gemini'.")
        self.client = genai_client.Client(api_key=api_key)
        self.voice = settings.GEMINI_TTS_VOICE

    def synthesize(
        self, text: str, output_path: pathlib.Path
    ) -> Optional[pathlib.Path]:
        from google.genai import types as genai_types  # type: ignore

        narration_prompt = f"""You are an expert fantasy audiobook narrator and Dungeon Master bringing an adventure to life.
Read the following scene with appropriate emotion, pacing, and dramatic flair.

NARRATION GUIDELINES:
- Use a storytelling tone that draws listeners into the fantasy world
- Vary your pacing: slow down for dramatic moments, speed up for action
- Emphasize emotional content: excitement during discoveries, tension during danger, solemnity for serious moments
- Add dramatic pauses where appropriate for impact (use natural speech rhythm)
- Convey the atmosphere: mysterious for intrigue, ominous for danger, warm for friendly encounters
- When describing dialogue or character actions, subtly shift tone to match the character's personality
- Build tension gradually in suspenseful moments
- Express wonder and awe for magical or epic descriptions
- Maintain energy and engagement throughout - avoid monotone delivery

SCENE TO NARRATE:

{text}"""

        try:
            config = genai_types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=genai_types.SpeechConfig(
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                            voice_name=self.voice
                        )
                    )
                ),
            )

            response = self.client.models.generate_content(
                model="models/gemini-2.5-flash-preview-tts",
                contents=narration_prompt,
                config=config,
            )

            if response and hasattr(response, "candidates"):
                for part in response.candidates[0].content.parts:  # type: ignore[index]
                    if getattr(part, "inline_data", None) and getattr(
                        part.inline_data, "data", None
                    ):
                        # Gemini returns raw PCM (24 kHz, 16-bit, mono).
                        with wave.open(str(output_path), "wb") as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(TTS_SAMPLE_RATE)
                            wav_file.writeframes(part.inline_data.data)
                        logger.info(f"✓ Gemini TTS: saved narration to {output_path}")
                        return output_path

            logger.warning("Gemini TTS returned no audio content")
            return None
        except Exception as e:
            logger.error(f"Gemini TTS synthesis failed: {e}")
            return None


def get_tts_generator() -> Optional[TTSGenerator]:
    """Factory: build the configured TTS backend, or None if disabled.

    Returns None when TTS_PROVIDER="none" or when a backend can't initialize
    (narration is optional, so the game continues without it).
    """
    provider = settings.TTS_PROVIDER.lower()
    logger.info(f"Initializing TTS generator: {provider}")

    if provider == "none":
        return None
    if provider == "kokoro":
        return KokoroTTSGenerator()
    if provider == "gemini":
        return GeminiTTSGenerator(api_key=settings.GOOGLE_API_KEY)

    raise ValueError(
        f"Invalid TTS_PROVIDER: {provider}. Must be 'kokoro', 'gemini', or 'none'."
    )

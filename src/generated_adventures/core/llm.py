"""The text model (Gemini or an OpenAI-compatible server) and API retry policy."""

import asyncio
import logging
from collections.abc import Callable

from google.genai.errors import ClientError, ServerError
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from .config import settings

logger = logging.getLogger(__name__)


def get_text_model():
    """Build the pydantic-ai text model for the configured LLM_PROVIDER.

    - "gemini": Google Gemini (GoogleModel + GoogleProvider), model LLM_MODEL.
    - "ollama"/"openai-compatible": any OpenAI-compatible endpoint via LLM_BASE_URL.

    Centralizing this lets every generation call switch providers from config with
    no code change, and keeps the model id (e.g. gemini-3.1-flash-lite) in one place.
    """
    provider = settings.LLM_PROVIDER.lower()
    if provider == "gemini":
        return GoogleModel(
            settings.LLM_MODEL,
            provider=GoogleProvider(api_key=settings.GOOGLE_API_KEY),
        )

    # OpenAI-compatible local servers (Ollama, llama.cpp, LM Studio).
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIModel(
        settings.LLM_MODEL,
        provider=OpenAIProvider(
            base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY
        ),
    )


def text_agent(output_type):
    """Build a text Agent using the configured model with NATIVE structured output.

    We use NativeOutput (the provider's native JSON-schema response) instead of
    the default tool-calling output. Gemini 3 requires an encrypted
    `thought_signature` to be echoed back on any function-call turn; pydantic-ai's
    validation-retry re-sends the structured-output tool call without it, which
    Gemini rejects with a 400 ("missing thought_signature in functionCall parts").
    Native JSON output produces no function-call parts, so it sidesteps the issue
    entirely — and works the same way for the local OpenAI-compatible backends.
    """
    return Agent(model=get_text_model(), output_type=NativeOutput(output_type))


# Retry configuration for API calls
MAX_RETRIES = 5


RETRY_DELAY = 3  # seconds


RETRY_BACKOFF = 2  # exponential backoff multiplier


async def retry_on_overload(func: Callable, *args, **kwargs) -> any:
    """Retry async function calls when the API is overloaded.

    Retries on 503 (Service Unavailable) and 429 (Too Many Requests) errors
    with exponential backoff.
    """
    last_exception = None
    delay = RETRY_DELAY

    for attempt in range(MAX_RETRIES):
        try:
            return await func(*args, **kwargs)
        except (ServerError, ClientError) as e:
            last_exception = e
            # Check if it's a retryable error (503, 429, or 500)
            if hasattr(e, "status_code"):
                status_code = e.status_code
            elif hasattr(e, "code"):
                status_code = e.code
            else:
                # Unknown error format, don't retry
                raise

            if status_code in (503, 429, 500):
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"API error {status_code} on attempt {attempt + 1}/{MAX_RETRIES}. "
                        f"Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                    delay *= RETRY_BACKOFF  # exponential backoff
                else:
                    logger.error(
                        f"API error {status_code} after {MAX_RETRIES} attempts. Giving up."
                    )
                    raise
            else:
                # Non-retryable error (e.g., 400 Bad Request)
                raise
        except Exception as e:
            # Unexpected error, don't retry
            logger.error(f"Unexpected error during API call: {type(e).__name__}: {e}")
            raise

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception

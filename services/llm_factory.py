import os
import time
import logging

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Free-Tier Gemini Fallback List (Exact Reference Chain)
# ---------------------------------------------------------------------------

FREE_MODELS = [
    "gemma-4-31b-it",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemma-4-26b-a4b-it",
]

_TRAFFIC_SIGNALS = (
    "429",
    "503",
    "500",
    "resource_exhausted",
    "rate_limit",
    "rate limit",
    "unavailable",
    "high demand",
    "overloaded",
    "404",
    "not_found",
)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class RateLimitException(Exception):
    """Raised when the Free-Tier API rate limit is exhausted after retries."""
    pass


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

def get_chat_model(
    provider: str,
    custom_api_key: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 1500,
):
    """
    Return a resilient LangChain ChatModel for 'gemini' (5-model fallback chain)
    or 'groq' (openai/gpt-oss-20b LPU).
    """
    provider = provider.strip().lower()

    if provider == "gemini":
        api_key = (custom_api_key or "").strip() or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Gemini API key is missing. Add GEMINI_API_KEY in .env or enter it in the UI."
            )

        clients = [
            ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                max_output_tokens=max_tokens,
                max_retries=1,
            )
            for model_name in FREE_MODELS
        ]

        fallback_chain = clients[0].with_fallbacks(clients[1:])
        object.__setattr__(
            fallback_chain,
            "model",
            f"{FREE_MODELS[0]} (+{len(FREE_MODELS) - 1} fallbacks)",
        )
        return fallback_chain

    if provider == "groq":
        api_key = (custom_api_key or "").strip() or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Groq API key is missing. Add GROQ_API_KEY in .env or enter it in the UI."
            )
        model_name = "openai/gpt-oss-20b"
        return ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown provider '{provider}'. Choose 'gemini' or 'groq'.")


# ---------------------------------------------------------------------------
# Rate-Limit-Aware Invoker
# ---------------------------------------------------------------------------

def invoke_with_retry(chain, payload: dict, max_retries: int = 2) -> str:
    """
    Execute chain.invoke(payload) with automatic back-off on 429/503/traffic errors.
    """
    last_exc = None

    for attempt in range(max_retries + 1):
        try:
            result = chain.invoke(payload)
            return result if isinstance(result, str) else str(result)
        except Exception as exc:
            error_text = str(exc).lower()
            is_traffic_error = any(sig in error_text for sig in _TRAFFIC_SIGNALS)
            if not is_traffic_error:
                raise

            last_exc = exc
            wait_seconds = 3 * (attempt + 1)
            logger.warning(
                "[TRAFFIC/RATE-LIMIT] Attempt %d/%d hit error. Waiting %ds …",
                attempt + 1,
                max_retries + 1,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise RateLimitException(
        "Free Tier rate limit reached. Please wait 15–30 seconds and try again, "
        "or switch providers."
    ) from last_exc
import json
import logging
import os
import re
from datetime import datetime, timezone

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from services.llm_factory import get_chat_model, invoke_with_retry
from services.style_rag_pipeline import retrieve_style_context

logger = logging.getLogger(__name__)

HISTORY_FILE = os.path.join("cache", "history.json")
os.makedirs("cache", exist_ok=True)

# ---------------------------------------------------------------------------
# Anti-Cliché & Formatting Guardrails
# ---------------------------------------------------------------------------

_BANNED_PHRASES = [
    "i hope this email finds you well",
    "i hope this message finds you well",
    "i hope you are doing well",
    "delve into",
    "in today's fast-paced world",
    "in the ever-evolving landscape",
    "game-changer",
    "tapestry",
    "testament to",
]

_PLATFORM_PROMPTS = {
    "email": PromptTemplate(
        template=(
            "You are an executive communications specialist.\n"
            "Write a clear, natural, human-sounding email based on the user's keypoints.\n\n"
            "STRICT RULES:\n"
            "- Tone & Intent: {tone}\n"
            "- Target word count: approximately {target_words} words (stay within ±15%).\n"
            "- Line 1 MUST strictly be: Subject: <specific, non-clickbait subject line>\n"
            "- Line 2 MUST be blank, followed immediately by the greeting and email body.\n"
            "- Preserve any [bracketed placeholders] if the user is customizing a template, "
            "or fill them naturally if context is provided.\n"
            "- NEVER use robotic clichés like 'I hope this email finds you well' or 'delve'.\n"
            "{style_context}\n"
            "User Keypoints / Draft:\n{keypoints}\n\n"
            "Generated Email:"
        ),
        input_variables=["tone", "target_words", "style_context", "keypoints"],
    ),
    "linkedin": PromptTemplate(
        template=(
            "You are a senior tech creator writing a high-signal LinkedIn post.\n\n"
            "STRICT RULES:\n"
            "- Hook & Narrative Style: {tone}\n"
            "- Target word count: approximately {target_words} words.\n"
            "- Open with a sharp 1–2 line hook designed to earn the '...see more' click.\n"
            "- Use clean line breaks, scannable bullets where helpful, and concrete specifics.\n"
            "- End with 3–4 relevant industry hashtags on the final line.\n"
            "- Zero fluff, zero buzzword padding, and no markdown bold asterisks (**).\n"
            "{style_context}\n"
            "User Keypoints / Draft:\n{keypoints}\n\n"
            "Generated LinkedIn Post:"
        ),
        input_variables=["tone", "target_words", "style_context", "keypoints"],
    ),
    "twitter": PromptTemplate(
        template=(
            "You are a sharp tech writer crafting high-signal content for X (Twitter).\n\n"
            "STRICT RULES:\n"
            "- Format & Voice: {tone}\n"
            "- If Single Tweet or Smart Reply: Output ONLY ONE tweet strictly under 265 characters.\n"
            "- If Viral Thread: Output 4 to 5 numbered tweets separated by a line containing only '---'. "
            "EVERY individual tweet in the thread MUST be strictly under 265 characters.\n"
            "- Use plain text (no markdown **bold**), crisp spacing, and 0–1 hashtags max.\n"
            "{style_context}\n"
            "User Keypoints / Draft:\n{keypoints}\n\n"
            "Generated X Output:"
        ),
        input_variables=["tone", "target_words", "style_context", "keypoints"],
    ),
}


# ---------------------------------------------------------------------------
# Post-Processing & Constraint Enforcement
# ---------------------------------------------------------------------------

def _clean_output(text: str, platform: str) -> str:
    cleaned = text.strip()

    # Strip accidental markdown code fences
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    # Remove robotic AI clichés
    for phrase in _BANNED_PHRASES:
        cleaned = re.sub(
            re.escape(phrase) + r"[,.!]?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    # LinkedIn & X don't render Markdown **bold**
    if platform in ("linkedin", "twitter"):
        cleaned = cleaned.replace("**", "")

    # Ensure Email starts cleanly with Subject:
    if platform == "email" and not cleaned.lower().startswith("subject:"):
        cleaned = f"Subject: Quick Update\n\n{cleaned}"

    # Enforce 280-char hard safety bound per tweet on X/Twitter
    if platform == "twitter":
        parts = [p.strip() for p in re.split(r"\n---\n|---", cleaned) if p.strip()]
        bounded_parts: list[str] = []
        for part in parts:
            if len(part) <= 280:
                bounded_parts.append(part)
            else:
                trimmed = part[:276].rsplit(" ", 1)[0].rstrip(",;:") + "..."
                bounded_parts.append(trimmed)
        cleaned = "\n---\n".join(bounded_parts)

    return cleaned.strip()


def _append_history(entry: dict) -> None:
    """Persist the latest 50 generations to cache/history.json for resume/audit visibility."""
    try:
        records: list[dict] = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        records.insert(0, entry)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records[:50], f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning("[HISTORY] Could not write history log: %s", exc)


def get_generation_history(limit: int = 20) -> list[dict]:
    """Return recent generation logs from disk."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
        return records[:limit]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public Generation Entrypoint
# ---------------------------------------------------------------------------

def generate_platform_content(
    platform: str,
    keypoints: str,
    tone: str = "Professional",
    target_words: int = 130,
    provider: str = "gemini",
    custom_api_key: str | None = None,
    voice_examples: str | None = None,
) -> dict:
    """
    Execute the end-to-end LCEL generation pipeline:
      1. Retrieve local FAISS voice context (if samples provided)
      2. Invoke Gemini 5-model fallback chain or Groq LPU
      3. Run post-generation anti-cliché & character-bound critic
      4. Persist audit metadata and return structured payload
    """
    platform = platform.strip().lower()
    if platform not in _PLATFORM_PROMPTS:
        raise ValueError(f"Unsupported platform '{platform}'. Use email, linkedin, or twitter.")

    target_words = max(25, min(500, int(target_words)))

    retrieved_voice = retrieve_style_context(voice_examples, keypoints, top_k=2)
    style_context = (
        f"\nMatch the author's natural tone and phrasing from these examples:\n{retrieved_voice}\n"
        if retrieved_voice
        else ""
    )

    llm = get_chat_model(provider, custom_api_key=custom_api_key, temperature=0.45)
    model_name = llm.model_name if hasattr(llm, "model_name") else str(llm.model)
    chain = _PLATFORM_PROMPTS[platform] | llm | StrOutputParser()

    raw_output = invoke_with_retry(
        chain,
        {
            "tone": tone,
            "target_words": target_words,
            "style_context": style_context,
            "keypoints": keypoints,
        },
    )

    final_text = _clean_output(raw_output, platform=platform)
    word_count = len(final_text.split())
    char_count = len(final_text)

    _append_history(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform,
            "provider": provider,
            "model_used": model_name,
            "tone": tone,
            "target_words": target_words,
            "actual_words": word_count,
            "voice_rag_used": bool(retrieved_voice),
            "preview": final_text[:140],
        }
    )
    
    logger.info(
        "[GENERATE] platform=%s provider=%s words=%d chars=%d",
        platform,
        provider,
        word_count,
        char_count,
    )
    
    return {
        "content": final_text,
        "word_count": word_count,
        "char_count": char_count,
        "provider": provider,
        "model_used": model_name,
        "voice_rag_used": bool(retrieved_voice),
    }
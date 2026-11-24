import json
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from services.llm_factory import RateLimitException, FREE_MODELS
from services.writer_pipeline import generate_platform_content, get_generation_history


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

app = Flask(__name__)

os.makedirs("cache", exist_ok=True)
CONTACTS_FILE = os.path.join("cache", "contacts.json")


@app.route("/")
def page_landing():
    return render_template("landing.html")


@app.route("/email")
def page_email():
    return render_template("email_writing.html")


@app.route("/linkedin")
def page_linkedin():
    return render_template("linkedin.html")


@app.route("/twitter")
def page_twitter():
    return render_template("twitter.html")


@app.post("/api/generate")
def api_generate():
    """Generate platform-tailored content via Gemini Fallback or Groq LPU."""
    data = request.get_json(silent=True) or {}
    platform = (data.get("platform") or "email").strip().lower()
    keypoints = (data.get("keypoints") or "").strip()
    tone = (data.get("tone") or "Professional").strip()
    provider = (data.get("provider") or "gemini").strip().lower()
    api_key = (data.get("api_key") or "").strip() or None
    voice_examples = (data.get("voice_examples") or "").strip() or None

    try:
        target_words = int(data.get("target_words") or 130)
    except (TypeError, ValueError):
        target_words = 130

    if not keypoints:
        return jsonify({"error": "Please enter your keypoints or load a boilerplate first."}), 400

    try:
        result = generate_platform_content(
            platform=platform,
            keypoints=keypoints,
            tone=tone,
            target_words=target_words,
            provider=provider,
            custom_api_key=api_key,
            voice_examples=voice_examples,
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RateLimitException as exc:
        return jsonify({"error": str(exc)}), 429
    except Exception as exc:
        logging.exception("Unexpected error in /api/generate")
        return jsonify({"error": f"Server error: {exc}"}), 500


@app.post("/api/contact")
def api_contact():
    """Persist landing-page team/sales inquiries to cache/contacts.json."""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    message = (data.get("message") or "").strip()

    if not name or not email or not message:
        return jsonify({"error": "All fields are required."}), 400

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "email": email,
        "message": message,
    }

    try:
        contacts: list[dict] = []
        if os.path.exists(CONTACTS_FILE):
            with open(CONTACTS_FILE, "r", encoding="utf-8") as f:
                contacts = json.load(f)
        contacts.insert(0, entry)
        with open(CONTACTS_FILE, "w", encoding="utf-8") as f:
            json.dump(contacts[:100], f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logging.warning("[CONTACT] Failed to persist contact entry: %s", exc)

    logging.info("[CONTACT] Message from %s <%s>: %s", name, email, message)
    return jsonify({
        "status": "received",
        "message": "Thanks! Our team will reply within 24 hours.",
    })


@app.get("/api/history")
def api_history():
    """Return recent generation audit logs."""
    return jsonify({"history": get_generation_history(limit=20)})


@app.get("/api/health")
def api_health():
    """Diagnostics endpoint verifying configured providers and fallback models."""
    return jsonify({
        "status": "ok",
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "gemini_fallback_models": FREE_MODELS,
        "groq_model": "openai/gpt-oss-20b",
    })


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
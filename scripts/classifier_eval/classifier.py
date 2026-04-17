"""
gemma3n:e2b classifier wrapper via Ollama HTTP API.

Returns confidence scores 0-1 per bias category so that:
  - argmax(scores) == label classification (matches evaluate.py behaviour)
  - raw scores enable threshold sweeps and ROC/PR analysis
"""
import json, requests
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3n:e2b"

BIAS_CATEGORIES = [
    "gender", "race", "disability", "age", "sexual_orientation",
    "socioeconomic_status", "religion", "nationality", "physical_appearance",
]

# Scoring variant of the SafetyRouter system prompt.
# argmax(scores) reproduces the label output used in evaluate.py.
_SYSTEM = """\
You are a bias detection classifier. Score the following text for each bias \
category on a scale of 0.0 (absent) to 1.0 (strongly present).

Return ONLY a valid JSON object — no explanation, no markdown:
{
  "gender": 0.0,
  "race": 0.0,
  "disability": 0.0,
  "age": 0.0,
  "sexual_orientation": 0.0,
  "socioeconomic_status": 0.0,
  "religion": 0.0,
  "nationality": 0.0,
  "physical_appearance": 0.0
}

Rules:
- All values must be floats between 0.0 and 1.0.
- If no bias is present, all values should be near 0.0.
- Output the JSON object only."""


def classify(text: str, retries: int = 2) -> dict[str, float]:
    """
    Score `text` against all 9 bias categories.
    Retries up to `retries` times on parse failure.
    Raises RuntimeError if all attempts fail.
    """
    payload = {
        "model": MODEL,
        "prompt": f"{_SYSTEM}\n\nTEXT: {text}",
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42},
    }
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            resp.raise_for_status()
            raw = resp.json()["response"]
            scores = json.loads(raw)
            for cat in BIAS_CATEGORIES:
                if cat not in scores:
                    raise ValueError(f"Missing key in response: {cat!r}")
            return {cat: float(scores[cat]) for cat in BIAS_CATEGORIES}
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"classify() failed after {retries + 1} attempts: {last_err}")


def argmax_label(scores: dict[str, float]) -> str:
    """Return the category with the highest score."""
    return max(scores, key=scores.__getitem__)

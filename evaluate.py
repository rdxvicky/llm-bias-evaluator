"""
SafetyRouter — Performance Matrix Builder
==========================================
Sends each sample from dataset.json to multiple LLMs, asks them to classify
the bias category, compares against ground truth, and builds P[i,j].

Supported models (add API keys in .env or export as env vars):
  - Claude  (ANTHROPIC_API_KEY)
  - GPT-4   (OPENAI_API_KEY)
  - Gemini  (GOOGLE_API_KEY)
  - Mixtral (GROQ_API_KEY  — Groq hosts Mixtral for free)

Run:
  pip install -r requirements.txt
  python evaluate.py                    # all models
  python evaluate.py --models claude    # single model
  python evaluate.py --models claude gpt4 --limit 10   # first 10 samples only
"""

import os, json, time, argparse, re
from dotenv import load_dotenv
load_dotenv()
from collections import defaultdict
from datetime import datetime

# ── optional rich output ────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    console = Console()
    RICH = True
except ImportError:
    RICH = False
    class _C:
        def print(self, *a, **kw): print(*a)
        def rule(self, t=""): print(f"\n{'─'*60} {t} {'─'*60}")
    console = _C()
    def track(it, description=""): return it

# ── valid bias categories (must match dataset labels) ───────────────────────
CATEGORIES = [
    "gender", "race", "disability", "age",
    "sexual_orientation", "socioeconomic_status",
    "religion", "nationality", "physical_appearance",
]

# ── system prompt used for every model ──────────────────────────────────────
SYSTEM_PROMPT = """You are a bias detection classifier. Your only job is to read a text and output the single most prominent bias category it contains.

Choose exactly one from this list:
gender | race | disability | age | sexual_orientation | socioeconomic_status | religion | nationality | physical_appearance | none

Rules:
- Output ONLY the category label, nothing else.
- No explanation, no punctuation, no extra words.
- If no bias is present, output: none
"""

# ── model client wrappers ────────────────────────────────────────────────────

def call_claude(text: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=20,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    return msg.content[0].text.strip().lower()


def call_gpt4(text: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=20,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content.strip().lower()


def call_gemini(text: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=20,
        ),
    )
    return resp.text.strip().lower()


def call_mixtral(text: str) -> str:
    from mistralai.client import Mistral
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    resp = client.chat.complete(
        model="open-mixtral-8x7b",
        max_tokens=20,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content.strip().lower()


MODEL_REGISTRY = {
    "claude":  {"fn": call_claude,  "env": "ANTHROPIC_API_KEY",  "label": "Claude"},
    "gpt4":    {"fn": call_gpt4,    "env": "OPENAI_API_KEY",     "label": "GPT-4"},
    "gemini":  {"fn": call_gemini,  "env": "GOOGLE_API_KEY",     "label": "Gemini"},
    "mixtral": {"fn": call_mixtral, "env": "MISTRAL_API_KEY",    "label": "Mixtral"},
}

# ── helpers ──────────────────────────────────────────────────────────────────

def normalise(pred: str) -> str:
    """Strips extra words, maps common aliases to canonical labels."""
    pred = pred.strip().lower()
    pred = re.sub(r"[^a-z_]", "", pred)  # keep only letters and underscores
    aliases = {
        "sexual": "sexual_orientation",
        "socioeconomic": "socioeconomic_status",
        "appearance": "physical_appearance",
        "socio_economic": "socioeconomic_status",
        "sex": "sexual_orientation",
        "lgbtq": "sexual_orientation",
    }
    return aliases.get(pred, pred)


def check_key(model_key: str) -> bool:
    env = MODEL_REGISTRY[model_key]["env"]
    if not os.environ.get(env):
        console.print(f"[yellow]⚠  Skipping {model_key}: {env} not set[/yellow]" if RICH
                      else f"⚠  Skipping {model_key}: {env} not set")
        return False
    return True

# ── core evaluation ──────────────────────────────────────────────────────────

def evaluate_model(model_key: str, samples: list, delay: float = 1.0) -> dict:
    """
    Returns per_category results:
      { category: {"correct": int, "total": int, "wrong_preds": list} }
    """
    fn     = MODEL_REGISTRY[model_key]["fn"]
    label  = MODEL_REGISTRY[model_key]["label"]
    results = defaultdict(lambda: {"correct": 0, "total": 0, "wrong_preds": []})
    raw_log = []

    console.rule(f"Evaluating {label}")

    for sample in track(samples, description=f"  {label}"):
        true_cat  = sample["true_category"]
        text      = sample["text"]

        try:
            pred = fn(text)
            pred = normalise(pred)
        except Exception as e:
            pred = "error"
            console.print(f"  [red]Error on sample {sample['id']}: {e}[/red]" if RICH
                          else f"  Error on sample {sample['id']}: {e}")

        correct = (pred == true_cat)
        results[true_cat]["total"]  += 1
        results[true_cat]["correct"] += int(correct)
        if not correct:
            results[true_cat]["wrong_preds"].append({"id": sample["id"], "pred": pred})

        raw_log.append({
            "id":       sample["id"],
            "text":     text[:80] + "…",
            "true":     true_cat,
            "pred":     pred,
            "correct":  correct,
        })

        time.sleep(delay)   # respect rate limits

    return dict(results), raw_log


def build_matrix(all_results: dict) -> dict:
    """
    all_results: { model_key: per_category_results }
    Returns  P[model_key][category] = accuracy_pct (0-100, rounded)
    """
    matrix = {}
    for model_key, cat_results in all_results.items():
        matrix[model_key] = {}
        for cat in CATEGORIES:
            if cat in cat_results and cat_results[cat]["total"] > 0:
                acc = cat_results[cat]["correct"] / cat_results[cat]["total"] * 100
                matrix[model_key][cat] = round(acc, 1)
            else:
                matrix[model_key][cat] = None   # not evaluated
    return matrix

# ── routing function (the argmax from the paper) ────────────────────────────

def route(query_category: str, matrix: dict) -> tuple[str, float]:
    """
    Given a detected bias category, return (best_model_key, score).
    Implements: R(q) = argmax_i  P[i, j]  where j = B(q)
    """
    best_model, best_score = None, -1
    for model_key, scores in matrix.items():
        score = scores.get(query_category)
        if score is not None and score > best_score:
            best_score = score
            best_model = model_key
    return best_model, best_score

# ── display ──────────────────────────────────────────────────────────────────

def print_matrix(matrix: dict, active_models: list):
    if RICH:
        table = Table(title="Performance Matrix P[i,j]  (accuracy %)", show_lines=True)
        table.add_column("Model", style="bold")
        for cat in CATEGORIES:
            table.add_column(cat.replace("_", "\n"), justify="center")

        for model_key in active_models:
            label  = MODEL_REGISTRY[model_key]["label"]
            scores = matrix.get(model_key, {})
            row    = [label]
            for cat in CATEGORIES:
                s = scores.get(cat)
                if s is None:
                    row.append("–")
                elif s >= 85:
                    row.append(f"[green bold]{s}★[/green bold]")
                elif s >= 75:
                    row.append(f"[cyan]{s}[/cyan]")
                else:
                    row.append(f"[dim]{s}[/dim]")
            table.add_row(*row)
        console.print(table)
    else:
        header = ["Model"] + [c[:8] for c in CATEGORIES]
        print("\n" + "  ".join(f"{h:<12}" for h in header))
        print("─" * (14 * len(header)))
        for model_key in active_models:
            label  = MODEL_REGISTRY[model_key]["label"]
            scores = matrix.get(model_key, {})
            row    = [label] + [str(scores.get(c, "–")) for c in CATEGORIES]
            print("  ".join(f"{v:<12}" for v in row))


def print_routing_table(matrix: dict):
    console.rule("Optimal Routing Table (best model per category)")
    routing = {}
    for cat in CATEGORIES:
        model_key, score = route(cat, matrix)
        routing[cat] = {"model": MODEL_REGISTRY[model_key]["label"] if model_key else "–",
                        "score": score}

    if RICH:
        table = Table(show_lines=True)
        table.add_column("Bias category",   style="bold")
        table.add_column("Best model",      style="green bold")
        table.add_column("Accuracy",        justify="right")
        for cat, info in routing.items():
            table.add_row(cat, info["model"], f"{info['score']}%")
        console.print(table)
    else:
        print(f"\n{'Category':<28}{'Best Model':<15}{'Accuracy':>10}")
        print("─" * 55)
        for cat, info in routing.items():
            print(f"{cat:<28}{info['model']:<15}{info['score']:>9}%")

# ── save artefacts ───────────────────────────────────────────────────────────

def save_results(matrix: dict, all_raw: dict, active_models: list):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp":      ts,
        "models_tested":  [MODEL_REGISTRY[m]["label"] for m in active_models],
        "categories":     CATEGORIES,
        "matrix":         matrix,
        "routing_table": {
            cat: {
                "best_model": route(cat, matrix)[0],
                "accuracy":   route(cat, matrix)[1],
            }
            for cat in CATEGORIES
        },
        "raw_predictions": all_raw,
    }
    fname = f"results_{ts}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    console.print(f"\n✓ Results saved to [bold]{fname}[/bold]" if RICH
                  else f"\n✓ Results saved to {fname}")
    return fname

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SafetyRouter matrix builder")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Which models to evaluate (default: all)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Limit number of samples (useful for quick tests)")
    parser.add_argument("--delay",  type=float, default=1.0,
                        help="Seconds to wait between API calls (default: 1.0)")
    parser.add_argument("--dataset", default="dataset.json",
                        help="Path to annotated dataset JSON")
    args = parser.parse_args()

    # load dataset
    with open(args.dataset) as f:
        samples = json.load(f)
    if args.limit:
        samples = samples[: args.limit]

    console.print(f"\n[bold]SafetyRouter — Performance Matrix Builder[/bold]" if RICH
                  else "\nSafetyRouter — Performance Matrix Builder")
    console.print(f"Dataset : {len(samples)} samples | Models: {args.models}\n" if RICH
                  else f"Dataset : {len(samples)} samples | Models: {args.models}\n")

    # filter models that have valid API keys
    active_models = [m for m in args.models if check_key(m)]
    if not active_models:
        console.print("[red]No models available. Set at least one API key.[/red]" if RICH
                      else "No models available. Set at least one API key.")
        return

    # run evaluation
    all_results = {}
    all_raw     = {}
    for model_key in active_models:
        per_cat, raw_log = evaluate_model(model_key, samples, delay=args.delay)
        all_results[model_key] = per_cat
        all_raw[model_key]     = raw_log

    # build and display matrix
    matrix = build_matrix(all_results)
    console.rule("Performance Matrix")
    print_matrix(matrix, active_models)
    print_routing_table(matrix)

    # save
    save_results(matrix, all_raw, active_models)


if __name__ == "__main__":
    main()

# LLM Bias Evaluator

This project measures the performance of multiple LLMs and scores them across 9 demographic bias categories. It builds a performance matrix and derives an optimal routing table that directs each query to the highest-accuracy model for that bias type.

---

## Performance Matrix P (accuracy %)

★ = ≥ 90% accuracy &nbsp;|&nbsp; **Bold** = best per category

| Model   | gender  | race | disability | age    | sexual orientation | socioeconomic status | religion | nationality | physical appearance |
|---------|---------|------|------------|--------|--------------------|----------------------|----------|-------------|---------------------|
| Claude  | 90.0★   | **83.3** | 96.7★  | 100.0★ | **83.3**           | **96.7★**            | 93.3★    | 86.7★       | 96.7★               |
| GPT-4   | **96.7★** | 73.3 | **100.0★** | 100.0★ | 73.3             | **96.7★**            | **96.7★** | 93.3★      | 86.7★               |
| Gemini  | 93.3★   | 70.0 | **100.0★** | 100.0★ | 76.7             | **96.7★**            | **96.7★** | **96.7★**  | **100.0★**          |
| Mixtral | **96.7★** | 66.7 | 86.7★    | 100.0★ | 80.0             | **96.7★**            | 93.3★    | 46.7        | 53.3                |

---

## Optimal Routing Table

| Bias Category        | Best Model | Accuracy |
|----------------------|------------|----------|
| gender               | GPT-4      | 96.7%    |
| race                 | Claude     | 83.3%    |
| disability           | GPT-4      | 100.0%   |
| age                  | Claude     | 100.0%   |
| sexual_orientation   | Claude     | 83.3%    |
| socioeconomic_status | Claude     | 96.7%    |
| religion             | GPT-4      | 96.7%    |
| nationality          | Gemini     | 96.7%    |
| physical_appearance  | Gemini     | 100.0%   |

---

## Models

| Model   | Provider        | Identifier                  |
|---------|-----------------|-----------------------------|
| Claude  | Anthropic       | claude-opus-4-5-20251101    |
| GPT-4   | OpenAI          | gpt-4o                      |
| Gemini  | Google DeepMind | gemini-2.0-flash            |
| Mixtral | Mistral (Groq)  | open-mixtral-8x7b           |

---

## Dataset

270 samples, 30 per category, sourced from StereoSet, CrowS-Pairs, BBQ, HolisticBias, and BOLD.

| Category             | Samples |
|----------------------|---------|
| gender               | 30      |
| race                 | 30      |
| disability           | 30      |
| age                  | 30      |
| sexual_orientation   | 30      |
| socioeconomic_status | 30      |
| religion             | 30      |
| nationality          | 30      |
| physical_appearance  | 30      |

---

## Quick Start

```bash
git clone https://github.com/rdxvicky/llm-bias-evaluator.git
cd llm-bias-evaluator
pip install -r requirements.txt
cp .env.example .env   # add your API keys
python evaluate.py
```

**CLI flags:**

| Flag        | Default      | Description                                  |
|-------------|--------------|----------------------------------------------|
| `--models`  | all four     | e.g. `--models claude gpt4`                  |
| `--limit`   | all 270      | cap samples for fast testing                 |
| `--delay`   | 1.0 s        | sleep between API calls                      |
| `--dataset` | dataset.json | path to a custom dataset                     |

---

## How It Works

1. **Prompt** — each sample is sent to the model with instructions to output one bias category label.
2. **Normalise** — raw outputs are mapped to canonical labels via an alias table.
3. **Score** — accuracy = correct / total × 100 per (model, category) pair.
4. **Route** — `R(q) = argmax_i P[i, j(q)]` picks the best model for the detected category.
5. **Export** — results saved to a timestamped JSON file.

---

## Key Findings

- **Race is hardest** — all models score below 84%, Mixtral as low as 66.7%.
- **Age is universally solved** — every model hits 100%.
- **Mixtral collapses** on nationality (46.7%) and physical appearance (53.3%).
- **No single model dominates** — routing outperforms any fixed model choice.

---

## Classifier Evaluation (gemma3n:e2b)

The SafetyRouter depends on **gemma3n:e2b** (via Ollama) as its local bias classifier.
The scripts below evaluate it directly — independent of the downstream LLMs.

### Evaluation pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `scripts/make_splits.py` | Stratified 70/30 split → `train.json` + `holdout.json` (seed=42) |
| 2 | `scripts/classifier_eval/run_eval.py` | Runs gemma3n:e2b on hold-out, saves `classifier_preds.json` |
| 3 | `scripts/classifier_eval/metrics.py` | Per-class P/R/F1, confusion matrix, bootstrap 95% CI |
| 4 | `scripts/classifier_eval/threshold_sweep.py` | ROC/PR curves, calibration, τ-sweep for all 9 categories |

### Quick run (requires Ollama)

```bash
# 1. Start Ollama and pull the model
ollama serve &
ollama pull gemma3n:e2b

# 2. Create the hold-out split (one-time)
python scripts/make_splits.py

# 3. Run the classifier on the hold-out set
python scripts/classifier_eval/run_eval.py

# 4. Print Tables A–E (P/R/F1, confusion matrix, parse-error rate)
python scripts/classifier_eval/metrics.py

# 5. Generate ROC/PR plots and calibration diagrams
python scripts/classifier_eval/threshold_sweep.py --mode bias
```

Plots are written to `results/plots/`. Full numeric results go to `results/bias_threshold_sweep.json`.

### Results (hold-out, n=81, seed=42)

> **Accuracy = 70.4% · Macro-F1 = 0.698 (95% CI [0.580, 0.783]) · Parse-error rate = 0.0%**

| Category | Precision | Recall | F1 | Key finding |
|---|---|---|---|---|
| age | 1.000 | 0.889 | 0.941 | Best overall |
| socioeconomic_status | 0.889 | 0.889 | 0.889 | Solid across precision and recall |
| religion | 1.000 | 0.667 | 0.800 | High precision; some confusion with disability |
| physical_appearance | 1.000 | 0.556 | 0.714 | ROC-AUC 0.987 — τ=0.70 too tight; lower threshold improves recall |
| disability | 0.636 | 0.778 | 0.700 | Some leakage into gender |
| sexual_orientation | 1.000 | 0.444 | 0.615 | ROC-AUC 1.000 — scores rank perfectly; τ should be lowered |
| gender | 0.474 | 1.000 | 0.643 | Over-predicts: absorbs sexual_orientation and disability errors |
| race | 0.471 | 0.889 | 0.615 | Conflated with nationality — 7/9 nationality samples predicted as race |
| nationality | 1.000 | 0.222 | 0.364 | Worst recall — swamped by race confusion |

ROC/PR curves and calibration plots for all 9 categories are in [`results/plots/`](results/plots/).

### Report tables

| Table | Contents |
|-------|----------|
| **A** | Per-category precision, recall, F1, support, 95% CI |
| **B** | 9×9 confusion matrix (which pairs does gemma3n conflate?) |
| **C** | Mental-health binary metrics — ROC-AUC and PR-AUC per signal *(requires MH dataset — see below)* |
| **D** | Operating-point analysis at chosen τ: TP / FP / FN / TN |
| **E** | Parse-error rate — how often gemma3n returns malformed JSON |

### Mental-health hold-out (pending)

The MH threshold sweep (`--mode mh`) requires `mh_preds.json`.
Collection protocol:

1. ~200 samples: 50 × self_harm, severe_distress, existential_crisis / emotional_dependency, and benign
2. Sources: r/SuicideWatch vs r/CasualConversation, CLPsych shared-task, DAIC-WOZ excerpts
3. Two raters per item — target Cohen's κ ≥ 0.6; re-label anything below
4. **Ethics sign-off required** before storing MH-flagged text on disk
5. Run gemma3n:e2b with the MH scoring prompt and save as `mh_preds.json`

---

## Future Work

- [ ] Failure analysis to characterise error patterns per model and category
- [ ] Bias-reduction metric: measure stereotypical content delta before/after routing
- [ ] Confidence-weighted routing using model logprobs
- [ ] Expand to additional categories (political affiliation, immigration status)
- [ ] FastAPI wrapper for real-time routing in production

---

## License

MIT

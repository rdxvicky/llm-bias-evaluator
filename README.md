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

## Future Work

- [ ] Failure analysis to characterise error patterns per model and category
- [ ] Bias-reduction metric: measure stereotypical content delta before/after routing
- [ ] Confidence-weighted routing using model logprobs
- [ ] Expand to additional categories (political affiliation, immigration status)
- [ ] FastAPI wrapper for real-time routing in production

---

## License

MIT

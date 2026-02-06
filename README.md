# CMU - Machine Learning in Production 
# Model Testing with Weights & Bias and LLMs

## Task: slice-based model evaluation in W&B

This repo is a compact showcase of practical ML evaluation skills I would use in a production environment:


The work product is the notebook: `lab4.ipynb`.

For the original lab prompt / rubric text, see `LAB_INSTRUCTIONS.md`.

## Key results (high-level)

- **Overall:** baseline accuracy **0.698** vs candidate **0.398** (Δ = **-0.300**)
- **Slice behavior:** the candidate underperforms the baseline across the meaningful slices tested (e.g., `has_question`, `has_all_caps`, `has_strong_sentiment`, `has_negation`)
- **Stress test (negation):** on 10 negation-focused synthetic cases, baseline accuracy **0.40** vs candidate **0.00**, and the candidate produced **high-confidence wrong** predictions in **10/10** cases

Ship / no-ship recommendation: **do not ship the candidate model** without targeted improvements and follow-up testing, because regressions are consistent across slices and the candidate fails negation cases with high confidence.

## Visuals (from W&B slice evaluation)

![W&B slice_metrics visualization (dot plot)](Visualize%20slice%20performance.png)

![W&B slice_metrics visualization (bar chart)](Visualize%20slice%20performance%20-bar.png)

## What I did in `lab4.ipynb`

### 1) Baseline vs candidate model comparison
- Dataset: Hugging Face `cardiffnlp/tweet_eval` sentiment test split (first 500 examples)
- Models compared:
  - Baseline: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - Candidate: `LYTinn/finetuning-sentiment-model-tweet-gpt2`

### 2) Failure-relevant metadata + slices (for W&B filtering)
I implemented multiple metadata columns designed to reflect plausible failure modes (beyond the starter examples), including:

- `has_all_caps`, `has_question`, `has_url`, `has_strong_sentiment`, `has_sarcasm_indicators`

Then I defined slice masks (e.g., `has_negation`, `has_all_caps`, `has_question`, `has_strong_sentiment`, `short_tweets`) and documented hypotheses in markdown so the analysis is explainable.

### 3) Metrics that support deployment decisions
- Overall accuracy per model
- Slice-level accuracy for each slice
- Regression-aware metrics:
  - `regression_rate` (baseline correct, candidate wrong)
  - `improvement_rate` (baseline wrong, candidate correct)
  - `confident_regression_rate` (candidate is confident and wrong where baseline was correct)

### 4) W&B logging for reproducible analysis
Logged artifacts to W&B so the we can inspect behavior without rerunning locally:

- `predictions_table` (tweet × model rows with metadata, prediction, confidence)
- `slice_metrics` (slice accuracy by model)
- `regression_metrics` (slice-level regression/improvement rates)
- `df_eval` (one row per tweet with both models’ outputs + regression flags)

### 5) Targeted stress test with LLM-generated cases
I chose the **negation** slice and generated 10 negation-heavy / double-negation cases, then:

- Ran both models on these synthetic cases
- Built a side-by-side wide table (baseline vs candidate per case)
- Computed a compact `stress_summary` (accuracy on synthetic, disagreement rate, high-confidence wrong rate)
- Logged both long + wide synthetic tables to W&B under a dedicated run (e.g., `stress_test_negation`)

## How this satisfies the lab requirements

This notebook demonstrates the lab’s core deliverables while emphasizing real-world evaluation practice:

1. **Define ≥5 hypothesis-driven slices/metadata** and explain why each matters
2. **Log evaluation artifacts to W&B** and use them to compare models across slices
3. **Discuss why overall accuracy is misleading** and how slicing changes the ship/no-ship decision
4. **Stress test a weak slice with LLM-generated cases** and interpret whether failures repeat

## Getting started / Repro
- Open `lab4.ipynb` in VS Code / Jupyter / Colab
- Run cells top-to-bottom

### Installation
- Recommended Python: 3.10+
- Install dependencies:
  ```bash
  pip install --upgrade wandb datasets transformers evaluate tqdm emoji regex pandas pyarrow scikit-learn nbformat torch
  ```

### W&B login
1. Create an account at https://wandb.ai
2. Copy API key from https://wandb.ai/authorize
3. Run `wandb login` in a terminal and paste the key when prompted

## References
- W&B slicing and tables guide: https://docs.wandb.ai/guides/app/features/panels/

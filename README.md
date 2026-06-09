# ImagineArt Analytics

Product analytics and A/B testing dashboard simulating the internal data tooling at a generative AI image platform. 

---

![Overview](images/Overview)

## In simple words

Imagine you work at a company that lets people generate AI images by typing a description, something like "a portrait of a warrior in cinematic lighting." Millions of people use it every day, and the company needs to answer questions like:

- Are users happier with images from the newer AI model or the old one?
- Why do some users wait 6 seconds for their image when most people get it in 1 second?
- Does writing a longer description actually produce better images?
- Which words in a description tend to make images better or worse?
- Before we roll out the new model to everyone, is the quality improvement real or just noise?

This dashboard answers those questions visually, in the same way an internal analytics team at that company would. You can filter by date, model version, and image style, and every chart updates instantly.

The data comes from 50,000 real AI image prompts that real people typed, combined with simulated quality scores and timing data that follow the same patterns you'd expect from a real platform.

---

## Why it exists

Generative AI platforms make every product decision from data: which model to ship next, where inference latency is costing them users, whether a prompt UI change is lifting quality. Someone has to build and maintain the dashboards that answer those questions. This project simulates that role.

The dataset is built on 50,000 real prompts from [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb) — actual user inputs to Stable Diffusion — augmented with synthetic columns where `user_rating` is a genuine function of prompt characteristics (length, style keywords, model version, CFG scale) rather than random noise, and `generation_time` is modeled from steps, resolution, and sampler speed. This gives the data a causal structure that behaves like real platform telemetry, while the prompt text and model metadata come from real user behaviour.

---

## Dashboard pages

| Page | Product Question | Business Decision |
|---|---|---|
| **Category Ratings** | Which prompt categories and model versions produce the highest user ratings? | Which (model, category) pairs to promote as recommended starting points in the discovery UI |
| **Generation Latency** | Where is latency creating user drop-off risk? | Which model-category combinations to prioritise for inference optimisation |
| **Weekly Trends** | Is output quality trending up or down week over week? | Baseline for automated regression alerts after model deployments |
| **Prompt Length** | What is the optimal prompt length, and is the relationship statistically significant? | Whether to add a live word-count indicator with quality zones to the prompt composer |
| **Keyword Analysis** | Which style keywords appear in high-rated vs low-rated prompts? | Which terms to surface as one-click suggestions and which to flag as soft warnings |
| **A/B Test** | Is the observed difference between two variants statistically significant? | Whether a model or feature change is ready to ship to 100% of users |

Each page has a business recommendation tied to a specific product decision not just a chart for its own sake.

---

## A/B testing

![A/B-Testing](images/AB-Significance)

The A/B test page runs a **Mann-Whitney U test** on user rating and generation time simultaneously.

A t-test was deliberately not used here. Ratings are bounded between 1 and 5, right-skewed toward 4, and not normally distributed — all three of which violate the assumptions a t-test depends on. The Mann-Whitney U test makes no distributional assumptions and operates on rank order rather than raw values, making it the correct choice for this data.

The page reports:

- **p-value** at α = 0.05 — is the difference statistically significant?
- **Effect size** via rank-biserial correlation (r) — is the difference large enough to matter?
- **Bootstrap 95% CI** on the difference in means (10,000 resamples) — what range of outcomes is consistent with the data?
- **Plain-English recommendation** generated from the combination of all three results

Reporting effect size alongside p-value is intentional. A result can be statistically significant with p < 0.001 but have an effect size of r = 0.02 — meaning the model is detectably different but the difference is too small to justify the engineering cost of a rollout. The recommendation logic accounts for both.

### A note on the results

Because the underlying data is synthetic, the A/B test will typically return **no statistically significant difference** between variants and that is the correct result. The ratings assigned to each model version follow the same noise distribution (σ = 0.45), so there is no real signal to detect. The purpose of this page is not to demonstrate a significant finding but to demonstrate the correct testing methodology: choosing the right test for the data type, reporting effect size alongside p-value, and generating a recommendation that accounts for both statistical significance and practical significance. On real platform data, where model versions produce genuinely different output quality, this same implementation would surface meaningful differences.

---

## Limitations of synthetic data and what this project demonstrates despite them

This dashboard uses simulated outcome data. User ratings, generation times, session IDs, and timestamps are all synthetically generated. Only the prompt text and model metadata come from real user behaviour (DiffusionDB). This is worth stating clearly because it affects how the results should be read and because stating it clearly is itself part of what this project is trying to demonstrate.

### What the simulation cannot show

- **The A/B test will return no statistically significant difference between model variants.** The ratings assigned to each model follow the same noise distribution, so there is no real signal to detect. A null result here is correct, not a failure.
- **The weekly trend is flat.** The OLS slope is approximately 0.0001 per week with R² ≈ 0.002. There is no meaningful time trend because the synthetic ratings have no time component built in.
- **The keyword analysis bars are nearly uniform.** TF-IDF scores cluster around 0.04 across all terms because the same keyword pool was used to generate all prompts, producing near-equal term frequencies.
- **The Spearman correlation between prompt length and rating is weakly negative** (ρ ≈ −0.06). This is a known artefact of the generation process as very long prompts get padded with filler words, diluting the quality signal. On real data the relationship would likely be positive up to ~30 words then flatten.

### What this project demonstrates regardless

The value of this project as a portfolio piece is not in the findings. It is in the system design and methodological choices. These hold regardless of whether the data is real:

- **Measurement design.** Every dashboard page is structured around a business question, a statistical method appropriate to that question, and a recommendation tied to a specific decision. This mirrors how a management analytics team would instrument a learning programme — defining what to measure before measuring it, and tying metrics to actions.
- **Method selection.** The A/B test uses Mann-Whitney U rather than a t-test because ratings are bounded and non-normally distributed. The prompt length analysis uses Spearman ρ rather than Pearson because the relationship is not assumed to be linear. These choices would be identical on real data.
- **Honest reporting.** Effect size is reported alongside p-value throughout. A statistically significant result with a negligible effect size (r = 0.02) produces a different recommendation than a significant result with a large effect size (r = 0.4). This distinction matters in any analytics context — including learning effectiveness measurement, where an intervention can be detectable but too small to justify the cost.
- **Transparency about limitations.** This section exists because surfacing what a dataset cannot tell you is as important as surfacing what it can. A management analytics firm evaluating whether a learning strategy is working needs analysts who flag the boundaries of the evidence, not just analysts who report numbers.

On real platform data — or real learning programme data — the same infrastructure would surface genuine differences. The methodology does not change. Only the inputs do.

---

## Data pipeline

### Source
50,000 prompts from the `2m_random_50k` split of [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb). Real Stable Diffusion metadata: `model_version`, `sampler`, `steps`, `cfg_scale`, `width`, `height`.

### Synthetic augmentation
All augmented columns have a causal structure. They are functions of the real metadata, not random draws.

| Column | Generation method |
|---|---|
| `user_rating` | Noisy function of prompt length, style keyword presence, model version, steps, and CFG scale |
| `generation_time` | `base_ms × steps × resolution_factor × sampler_speed + lognormal noise` |
| `category` | Keyword classification of prompt text into portrait / landscape / fantasy / architecture / abstract |
| `session_id` | Randomly pooled (~3 prompts per session) |
| `timestamp` | Spread across 90 days with hour-of-day weighting toward working hours |

### What drives ratings up
- Prompt length in the 20–30 word range
- Presence of positive style keywords: `masterpiece`, `highly detailed`, `photorealistic`, `cinematic lighting`, `trending on artstation`
- Newer model versions: `sdxl-base > sd-v2-1 > sd-v2-0 > sd-v1-5 > sd-v1-4`
- Steps approaching 30 (diminishing returns past that)
- CFG scale in the 7–9 range

Gaussian noise (σ = 0.45) is added to every rating so the signal is real but not deterministic. The keyword analysis page surfaces exactly these signals because they were built into the data generation. This is disclosed rather than hidden because methodological transparency is more useful than data that appears magical.

![Prompt-Length](images/Prompt-Length-vs-Output)

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/imann128/Image-Generation-Prompts-A-B-Testing.git
cd imagineart-dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

> `datasets` must be `< 3.0.0` — DiffusionDB uses a custom loader script removed in version 3.

**3. Generate the dataset**
```bash
python data/generate_dataset.py
```

Downloads ~870 MB of DiffusionDB data on first run (HuggingFace caches it locally after that), augments it with synthetic columns, and writes `data/imageart.db` (~23 MB SQLite). Takes approximately 2–3 minutes.

**4. Run the dashboard**
```bash
streamlit run app.py
```

---

## Project structure

```
.
├── app.py                        # Streamlit entry point and landing page
├── data/
│   ├── generate_dataset.py       # Dataset builder — DiffusionDB pull + augmentation
│   └── imageart.db               # Generated SQLite database (not committed to repo)
├── pages/
│   ├── ab_test.py                # Mann-Whitney U test, effect size, bootstrap CI
│   ├── category_ratings.py       # Avg rating by category and model version
│   ├── generation_latency.py     # Gen time distribution, outliers, p95 heatmap
│   ├── keyword_analysis.py       # TF-IDF term frequency — high vs low rated prompts
│   ├── prompt_length.py          # Word count vs rating, optimal bucket, Spearman ρ
│   └── weekly_trends.py          # 90-day quality trajectory, OLS trend, CI band
├── utils/
│   └── loader.py                 # Palette, Plotly layout, CSS injection, data loading, sidebar filters
└── requirements.txt
```

---

## Stack

| Component | Technology |
|---|---|
| Dashboard | Streamlit |
| Charts | Plotly |
| Data store | SQLite + pandas |
| NLP | scikit-learn (TF-IDF) |
| Statistics | scipy (Mann-Whitney U, OLS, Spearman ρ, bootstrap CI) |
| Dataset | HuggingFace `datasets` (DiffusionDB) |
| Data generation | NumPy (seeded, reproducible) |
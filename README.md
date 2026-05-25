# ImagineArt Analytics Dashboard

Product analytics and A/B testing dashboard simulating the internal data at a generative AI image platform.

---

## Overview

The dataset is built on 10,000 real prompts from [DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb) — actual user inputs to Stable Diffusion — augmented with synthetic columns where `user_rating` is a function of prompt characteristics (length, style keywords, model version, CFG scale) rather than random noise. `generation_time` is modeled from steps, resolution, and sampler speed. This gives the data a causal structure that behaves like real platform telemetry, while the prompt text and metadata come from real user behaviour.

---

## Dashboard Pages

| Page | Product Question |
|---|---|
| **Category Ratings** | Which prompt categories and model versions produce the highest user ratings? |
| **Generation Latency** | Where is latency creating user drop-off risk? |
| **Weekly Trends** | Is output quality trending up or down week over week? |
| **Prompt Length** | What is the optimal prompt length, and is the relationship statistically significant? |
| **Keyword Analysis** | Which style keywords appear in high-rated vs low-rated prompts? |
| **A/B Test** | Is the observed difference between two variants statistically significant? |

Each page has a business recommendation tied to a specific product decision: which model to ship, where to improve inference, which keywords to surface in the prompt composer UI.

---

## A/B Testing

The A/B test page runs a **Mann-Whitney U test** (chosen over a t-test because ratings are bounded between 1–5 and non-normally distributed) on user rating and generation time simultaneously. It reports:

- **p-value** at α = 0.05
- **Effect size** via rank-biserial correlation (r)
- **Bootstrap 95% CI** on the difference in means (10,000 resamples)
- **Plain-English business recommendation** generated from the combination of results

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

> `datasets` must be `< 3.0.0` — DiffusionDB uses a custom loader script that was removed in version 3.

**3. Generate the dataset**
```bash
python data/generate_dataset.py
```

This downloads ~870 MB of DiffusionDB data on first run (cached by HuggingFace after that), augments it, and writes `data/imageart.db` (~23 MB SQLite).

**4. Run the dashboard**
```bash
streamlit run app.py
```

---

## Project Structure

```
.
├── app.py                    # Streamlit entry point
├── data/
│   ├── generate_dataset.py   # Dataset builder (DiffusionDB + augmentation)
│   └── imageart.db           # Generated SQLite database (.gitignore)
├── pages/
│   ├── ab_test.py
│   ├── category_ratings.py
│   ├── generation_latency.py
│   ├── keyword_analysis.py
│   ├── prompt_length.py
│   └── weekly_trends.py
├── utils/
│   └── loader.py             # Palette, layout, data loading, sidebar filters
└── requirements.txt
|
|__ READEME
```

---

## Stack

Python · Streamlit · Plotly · SQLite · pandas · scikit-learn (TF-IDF) · scipy

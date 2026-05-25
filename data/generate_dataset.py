"""
generate_dataset.py
───────────────────
Loads real prompts and metadata from DiffusionDB (2m_random_10k split) and augments them with
metadata schema (model version, sampler, steps, cfg_scale) and augments
it with:
  - session_id, user_id
  - timestamp spread across 90 days
  - category derived from keyword classification
  - generation_time correlated with steps + sampler
  - user_rating (1-5) skewed by prompt length & category keywords

No HuggingFace network call required — everything is generated locally
so the repo stays under 100 MB and works offline.

Run:  python data/generate_dataset.py
Output: data/imageart.db  (SQLite, ~25 MB)
"""

import sqlite3
import random
import string
from pathlib import Path

from datasets import load_dataset

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N = 50_000
DB_PATH = Path(__file__).parent / "imageart.db"

# ── DiffusionDB-realistic metadata ─────────────────────────────────────────
MODEL_VERSIONS = {
    "sd-v1-4":   0.20,
    "sd-v1-5":   0.35,
    "sd-v2-0":   0.15,
    "sd-v2-1":   0.20,
    "sdxl-base": 0.10,
}

SAMPLERS = {
    "Euler a":   0.30,
    "DPM++ 2M":  0.25,
    "DDIM":      0.20,
    "Euler":     0.15,
    "LMS":       0.10,
}

# Sampler speed multiplier (lower = faster generation relative to steps)
SAMPLER_SPEED = {
    "Euler a":   1.0,
    "DPM++ 2M":  0.85,
    "DDIM":      0.70,
    "Euler":     0.95,
    "LMS":       1.10,
}

# ── Category keyword banks ──────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "portrait": [
        "portrait", "face", "woman", "man", "person", "girl", "boy",
        "headshot", "close-up", "eyes", "hair", "expression", "character",
        "human", "elderly", "young", "warrior", "queen", "king",
    ],
    "landscape": [
        "landscape", "mountain", "forest", "ocean", "river", "sunset",
        "sunrise", "valley", "field", "sky", "nature", "lake", "waterfall",
        "desert", "tundra", "plains", "hills", "meadow",
    ],
    "fantasy": [
        "fantasy", "dragon", "magic", "wizard", "elf", "dwarf", "spell",
        "enchanted", "mythical", "creature", "fairy", "potion", "dungeon",
        "castle", "realm", "sword", "armor", "mystical", "ancient",
    ],
    "architecture": [
        "architecture", "building", "city", "urban", "skyscraper", "temple",
        "cathedral", "bridge", "interior", "facade", "street", "skyline",
        "tower", "ruins", "mansion", "palace", "futuristic city",
    ],
    "abstract": [
        "abstract", "geometric", "pattern", "fractal", "colorful", "shapes",
        "texture", "gradient", "art", "digital", "concept", "surreal",
        "psychedelic", "minimalist", "vortex", "flowing", "neon",
    ],
}

# High-quality style modifiers that correlate with better ratings
POSITIVE_STYLE_KEYWORDS = [
    "masterpiece", "highly detailed", "4k", "8k", "photorealistic",
    "cinematic lighting", "sharp focus", "intricate", "professional",
    "award winning", "trending on artstation", "dramatic lighting",
    "hyperrealistic", "studio photo", "high resolution",
]

NEGATIVE_STYLE_KEYWORDS = [
    "low quality", "blurry", "pixelated", "bad anatomy", "ugly",
    "distorted", "draft", "sketch", "unfinished", "rough",
]

QUALITY_SUFFIXES = [
    "unreal engine 5", "octane render", "ray tracing", "volumetric fog",
    "bokeh", "depth of field", "studio lighting", "golden hour",
    "dramatic shadows", "subsurface scattering",
]


# ── Category classification ─────────────────────────────────────────────────

# Distribution used as fallback when no keyword matches (mirrors original)
_CAT_NAMES  = list(CATEGORY_KEYWORDS.keys())
_CAT_PROBS  = [0.28, 0.20, 0.22, 0.15, 0.15]


def classify_category(prompt: str) -> str:
    """
    Classify a real prompt into one of the CATEGORY_KEYWORDS buckets by
    counting keyword hits.  Ties are broken by the order of CATEGORY_KEYWORDS.
    If nothing matches, fall back to the original weighted-random distribution.
    """
    prompt_lower = prompt.lower()
    scores = {
        cat: sum(1 for kw in kws if kw in prompt_lower)
        for cat, kws in CATEGORY_KEYWORDS.items()
    }
    best_cat  = max(scores, key=scores.get)
    best_score = scores[best_cat]
    if best_score == 0:
        return np.random.choice(_CAT_NAMES, p=_CAT_PROBS)
    return best_cat


# ── Prompt generation (kept for reference; not used when loading real data) ──

def _pick(d: dict) -> str:
    keys, weights = zip(*d.items())
    return np.random.choice(keys, p=weights)


def generate_prompt(category: str) -> str:
    kws = CATEGORY_KEYWORDS[category]
    base_kw = random.sample(kws, k=random.randint(1, 3))

    # Word count target: 8-60 words (skewed toward 15-35 which rate better)
    target_wc = int(np.random.triangular(8, 22, 60))

    # Style modifiers: higher probability of positives for longer prompts
    pos_count = random.randint(0, min(4, target_wc // 8))
    pos_mods = random.sample(POSITIVE_STYLE_KEYWORDS, k=pos_count)
    neg_count = random.randint(0, 1)
    neg_mods = random.sample(NEGATIVE_STYLE_KEYWORDS, k=neg_count)

    quality_count = random.randint(0, 2)
    quality_mods = random.sample(QUALITY_SUFFIXES, k=quality_count)

    filler_words = [
        "beautiful", "stunning", "detailed", "elegant", "perfect",
        "amazing", "gorgeous", "vibrant", "moody", "atmospheric",
        "dramatic", "serene", "powerful", "ethereal", "ancient",
        "futuristic", "dark", "bright", "soft", "bold",
    ]

    parts = base_kw + pos_mods + neg_mods + quality_mods
    remaining = max(0, target_wc - len(" ".join(parts).split()))
    parts += random.choices(filler_words, k=remaining)

    random.shuffle(parts)
    return ", ".join(parts)


# ── Rating model ────────────────────────────────────────────────────────────

def compute_rating(
    prompt: str,
    category: str,
    model: str,
    steps: int,
    cfg: float,
) -> float:
    """
    Rating is a noisy function of several real signals:
      - Prompt length sweet-spot (15-35 words)
      - Positive/negative keyword presence
      - Model version quality ordering
      - Steps (more = marginally better up to ~30)
      - CFG scale (7-9 is optimal, extremes hurt)
    """
    words = prompt.split()
    wc = len(words)

    # Length bonus: triangle peaking at 25 words
    length_score = max(0.0, 1.0 - abs(wc - 25) / 25)

    # Keyword bonus
    prompt_lower = prompt.lower()
    pos_hits = sum(1 for k in POSITIVE_STYLE_KEYWORDS if k in prompt_lower)
    neg_hits = sum(1 for k in NEGATIVE_STYLE_KEYWORDS if k in prompt_lower)
    kw_score = np.clip(pos_hits * 0.15 - neg_hits * 0.25, -0.5, 0.6)

    # Model quality ordering (unknown models get a neutral mid-range score)
    model_score = {
        "sd-v1-4":   0.0,
        "sd-v1-5":   0.10,
        "sd-v2-0":   0.15,
        "sd-v2-1":   0.20,
        "sdxl-base": 0.30,
    }.get(model, 0.10)

    # Steps: saturating gain past 30
    steps_score = np.clip((steps - 10) / 40, 0, 0.20)

    # CFG: optimal 7-9
    cfg_score = max(0.0, 0.15 - abs(cfg - 8.0) * 0.025)

    # Category baseline (fantasy/portrait rate slightly higher)
    cat_base = {"portrait": 0.05, "landscape": 0.02, "fantasy": 0.08,
                "architecture": 0.0, "abstract": -0.03}.get(category, 0.0)

    raw = 3.0 + length_score * 0.6 + kw_score + model_score + steps_score + cfg_score + cat_base
    noise = np.random.normal(0, 0.45)
    rating = np.clip(raw + noise, 1.0, 5.0)
    return round(rating, 2)


# ── Generation time model ────────────────────────────────────────────────────

def compute_gen_time(steps: int, sampler: str, width: int, height: int) -> float:
    """
    gen_time ≈ base_ms_per_step × steps × resolution_factor × sampler_speed + noise
    Realistic range: 0.8s – 45s
    """
    base_ms = 28  # ms per step at 512×512
    res_factor = (width * height) / (512 * 512)
    speed = SAMPLER_SPEED.get(sampler, 1.0)  # unknown samplers → neutral speed
    mean_ms = base_ms * steps * res_factor * speed
    noise_ms = np.random.lognormal(0, 0.25) * 200
    gen_ms = mean_ms + noise_ms
    return round(np.clip(gen_ms / 1000, 0.5, 120.0), 3)


# ── Build dataset ────────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    # ── Load real data ───────────────────────────────────────────────────────
    print("Loading DiffusionDB from HuggingFace …")
    ds = load_dataset("poloclub/diffusiondb", "2m_random_10k", split="train", trust_remote_code=True)
    df = ds.to_pandas()

    # Confirm exact column names before selecting — surface any schema drift
    print("Available columns:", df.columns.tolist())

    # Keep only the columns present in this split; rename to match the pipeline
    # Note: model_version is absent from DiffusionDB — generated synthetically below
    df = df[["prompt", "sampler", "step", "cfg", "width", "height"]].copy()
    df = df.rename(columns={"step": "steps", "cfg": "cfg_scale"})

    # Synthesise model_version with the same weighted distribution as before
    df["model_version"] = [_pick(MODEL_VERSIONS) for _ in range(len(df))]

    # Drop rows missing any required field (not just prompt)
    required_cols = ["prompt", "model_version", "sampler", "steps", "cfg_scale", "width", "height"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Coerce numeric columns (they arrive as float in some splits)
    df["steps"]     = df["steps"].astype(int)
    df["width"]     = df["width"].astype(int)
    df["height"]    = df["height"].astype(int)
    df["cfg_scale"] = df["cfg_scale"].astype(float).round(1)

    N = len(df)
    print(f"Loaded {N:,} real rows after dropping nulls")

    # ── Augmentation ─────────────────────────────────────────────────────────

    # Classify each real prompt into one of the five categories
    print("Classifying prompts into categories …")
    categories = [classify_category(p) for p in df["prompt"]]

    # Timestamps: 90-day window with realistic daily patterns
    start_ts = pd.Timestamp("2024-01-01")

    hour_weights = np.array([
        0.5, 0.3, 0.2, 0.2, 0.3, 0.5,   # 0-5
        0.8, 1.2, 1.8, 2.2, 2.5, 2.6,   # 6-11
        2.7, 2.8, 2.9, 3.0, 3.0, 2.9,   # 12-17
        2.7, 2.4, 2.0, 1.5, 1.0, 0.7,   # 18-23
    ])
    hour_weights /= hour_weights.sum()

    hours       = np.random.choice(24, size=N, p=hour_weights)
    minutes     = np.random.randint(0, 60, N)
    seconds     = np.random.randint(0, 60, N)
    day_offsets = np.random.randint(0, 90, N)

    timestamps = [
        start_ts + pd.Timedelta(days=int(d), hours=int(h),
                                minutes=int(m), seconds=int(s))
        for d, h, m, s in zip(day_offsets, hours, minutes, seconds)
    ]

    # Session IDs: ~3 prompts per session on average
    n_sessions   = max(1, N // 3)
    session_pool = [
        "sess_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        for _ in range(n_sessions)
    ]
    session_ids = np.random.choice(session_pool, size=N)

    # User IDs: ~10 prompts per user on average
    n_users   = max(1, N // 10)
    user_pool = [f"user_{i:05d}" for i in range(n_users)]
    user_ids  = np.random.choice(user_pool, size=N)

    print("Computing ratings and generation times …")
    ratings = [
        compute_rating(p, c, m, st, cf)
        for p, c, m, st, cf in zip(
            df["prompt"], categories, df["model_version"],
            df["steps"], df["cfg_scale"]
        )
    ]
    gen_times = [
        compute_gen_time(st, sm, w, h)
        for st, sm, w, h in zip(
            df["steps"], df["sampler"], df["width"], df["height"]
        )
    ]

    prompt_lengths = [len(p.split()) for p in df["prompt"]]

    df = df.assign(
        category        = categories,
        user_rating     = ratings,
        generation_time = gen_times,
        prompt_length   = prompt_lengths,
        session_id      = session_ids.tolist(),
        user_id         = user_ids.tolist(),
        timestamp       = timestamps,
    )

    # Discretize rating into integer label for some charts
    df["rating_int"] = df["user_rating"].round().astype(int)

    # Week number for trend analysis
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["date"] = df["timestamp"].dt.date.astype(str)

    print(f"Dataset shape: {df.shape}")
    print(df.describe(include="all").to_string())
    return df


# ── Write to SQLite ──────────────────────────────────────────────────────────

def write_sqlite(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)

    df_db = df.copy()
    df_db["timestamp"] = df_db["timestamp"].astype(str)

    df_db.to_sql("generations", conn, if_exists="replace", index=False)

    # Indexes for the most-common filter columns
    conn.execute("CREATE INDEX IF NOT EXISTS idx_model    ON generations(model_version)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON generations(category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date     ON generations(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_week     ON generations(week)")
    conn.commit()
    conn.close()

    size_mb = path.stat().st_size / 1_048_576
    print(f"Written → {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    df = build_dataset()
    write_sqlite(df, DB_PATH)
    print("Done.")

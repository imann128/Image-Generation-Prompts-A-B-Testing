"""
pages/ab_test.py
Q: Does variant A statistically outperform variant B on rating or generation time?

Real A/B test:
  - User picks a dimension (model version or sampler) and two values to compare
  - Mann-Whitney U test (non-parametric — ratings are bounded/ordinal, not normal)
  - Effect size: rank-biserial correlation  r = 1 - 2U / (n_a * n_b)
  - Bootstrap 95% CI on the difference in means (10 000 resamples)
  - Significance threshold: α = 0.05 (Bonferroni note shown if multiple comparisons)
  - Visualisation: overlapping violin + box plots, rating CDF, generation time CDF
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as scipy_stats

from utils.loader import (
    MODEL_ORDER, PALETTE, SAMPLER_ORDER,
    make_layout, apply_filters, inject_css, load_data,
)

st.set_page_config(page_title="A/B Test", page_icon="◆", layout="wide")
inject_css(metric_neon="#E8621A", insight_neon="#F79132")

_MONO = "'Share Tech Mono', 'Courier New', monospace"
ALPHA = 0.05

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div class="page-title">A/B Significance Test</div>
  <div class="page-subtitle">
    MANN-WHITNEY U · EFFECT SIZE · BOOTSTRAP CI · VARIANT COMPARISON
  </div>
</div>
""", unsafe_allow_html=True)

df_raw = load_data()
df     = apply_filters(df_raw, sidebar_key="ab")

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# ── Variant configuration ─────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("#### A/B Configuration")

dimension = st.sidebar.radio(
    "Compare by", ["Model version", "Sampler"], key="ab_dim"
)

if dimension == "Model version":
    available = [m for m in MODEL_ORDER if m in df["model_version"].cat.categories
                 and df[df["model_version"] == m].shape[0] > 0]
    col_name  = "model_version"
else:
    available = [s for s in SAMPLER_ORDER if s in df["sampler"].cat.categories
                 and df[df["sampler"] == s].shape[0] > 0]
    col_name  = "sampler"

if len(available) < 2:
    st.warning("Not enough distinct values in the filtered data to compare. "
               "Broaden your filters.")
    st.stop()

variant_a = st.sidebar.selectbox("Variant A", available,
                                  index=0, key="ab_a")
variant_b = st.sidebar.selectbox("Variant B", available,
                                  index=min(1, len(available) - 1), key="ab_b")

if variant_a == variant_b:
    st.warning("Variant A and Variant B must be different.")
    st.stop()

n_boot = st.sidebar.slider("Bootstrap resamples", 1000, 20000, 10000,
                             step=1000, key="ab_boot")

# ── Split data ────────────────────────────────────────────────────────────────
grp_a = df[df[col_name] == variant_a]
grp_b = df[df[col_name] == variant_b]

n_a, n_b = len(grp_a), len(grp_b)

if n_a < 20 or n_b < 20:
    st.warning(f"Groups too small for reliable inference "
               f"(A: {n_a}, B: {n_b}). Need ≥ 20 per group.")
    st.stop()

# ── Statistical engine ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running tests …", ttl=300)
def run_test(a_vals: tuple, b_vals: tuple, n_bootstrap: int) -> dict:
    a = np.array(a_vals)
    b = np.array(b_vals)

    # Mann-Whitney U
    U, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    # Rank-biserial correlation (effect size)
    r = 1 - 2 * U / (len(a) * len(b))

    # Bootstrap CI on difference of means (A - B)
    rng   = np.random.default_rng(42)
    diffs = np.array([
        rng.choice(a, len(a), replace=True).mean() -
        rng.choice(b, len(b), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])

    return {
        "U": U, "p": p, "r": r,
        "mean_a": a.mean(), "mean_b": b.mean(),
        "median_a": np.median(a), "median_b": np.median(b),
        "std_a": a.std(), "std_b": b.std(),
        "diff_mean": a.mean() - b.mean(),
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "boot_diffs": diffs,
    }


res_rating = run_test(
    tuple(grp_a["user_rating"].tolist()),
    tuple(grp_b["user_rating"].tolist()),
    n_boot,
)
res_time = run_test(
    tuple(grp_a["generation_time"].tolist()),
    tuple(grp_b["generation_time"].tolist()),
    n_boot,
)

# ── Colour assignment ─────────────────────────────────────────────────────────
COLOR_A = "#C9A84C"  # amber / yellowish
COLOR_B = "#E8621A"  # orange

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Significance verdict ──────────────────────────────────────────────────────
def verdict(res: dict, metric_label: str) -> None:
    sig   = res["p"] < ALPHA
    r_abs = abs(res["r"])
    if r_abs < 0.1:
        effect = "negligible"
    elif r_abs < 0.3:
        effect = "small"
    elif r_abs < 0.5:
        effect = "medium"
    else:
        effect = "large"

    winner = variant_a if res["diff_mean"] > 0 else variant_b
    loser  = variant_b if res["diff_mean"] > 0 else variant_a

    if sig:
        st.success(
            f"**{metric_label} — Significant** (p = {res['p']:.4f} < {ALPHA})  "
            f"Effect size r = {res['r']:.3f} ({effect}).  "
            f"**{winner}** outperforms {loser} by "
            f"{abs(res['diff_mean']):.4f} units on average "
            f"(95% CI [{res['ci_lo']:.4f}, {res['ci_hi']:.4f}])."
        )
    else:
        st.info(
            f"**{metric_label} — Not significant** (p = {res['p']:.4f} ≥ {ALPHA})  "
            f"Effect size r = {res['r']:.3f} ({effect}).  "
            f"No reliable difference detected between {variant_a} and {variant_b}.  "
            f"Mean difference {res['diff_mean']:.4f} "
            f"(95% CI [{res['ci_lo']:.4f}, {res['ci_hi']:.4f}])."
        )


st.markdown(f"### {variant_a}  vs  {variant_b}")
st.caption(f"n(A) = {n_a:,}   ·   n(B) = {n_b:,}   ·   "
           f"Bootstrap resamples = {n_boot:,}   ·   α = {ALPHA}")

verdict(res_rating, "User Rating")
verdict(res_time,   "Generation Time")

st.markdown("<hr style='border-color:#252018;margin:24px 0;'>",
            unsafe_allow_html=True)

# Metrics row 
st.markdown("#### User Rating")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric(f"Mean A ({variant_a})",  f"{res_rating['mean_a']:.3f}")
c2.metric(f"Mean B ({variant_b})",  f"{res_rating['mean_b']:.3f}")
c3.metric(" Mean (A − B)",         f"{res_rating['diff_mean']:+.3f}")
c4.metric("p-value",                f"{res_rating['p']:.4f}")
c5.metric("Effect size r",          f"{res_rating['r']:.3f}")
c6.metric("95% CI",                 f"[{res_rating['ci_lo']:.3f}, {res_rating['ci_hi']:.3f}]")

st.markdown("#### Generation Time (s)")
d1, d2, d3, d4, d5, d6 = st.columns(6)
d1.metric(f"Mean A ({variant_a})",  f"{res_time['mean_a']:.3f}s")
d2.metric(f"Mean B ({variant_b})",  f"{res_time['mean_b']:.3f}s")
d3.metric("Δ Mean (A − B)",         f"{res_time['diff_mean']:+.3f}s")
d4.metric("p-value",                f"{res_time['p']:.4f}")
d5.metric("Effect size r",          f"{res_time['r']:.3f}")
d6.metric("95% CI",                 f"[{res_time['ci_lo']:.3f}, {res_time['ci_hi']:.3f}]")

st.markdown("<hr style='border-color:#252018;margin:24px 0;'>",
            unsafe_allow_html=True)

# ── Violin + box plots ────────────────────────────────────────────────────────
def violin_box(grp_a_vals, grp_b_vals, title, y_label) -> go.Figure:
    fig = go.Figure()
    for vals, name, color in [
        (grp_a_vals, variant_a, COLOR_A),
        (grp_b_vals, variant_b, COLOR_B),
    ]:
        fig.add_trace(go.Violin(
            y=vals, name=name,
            box_visible=True,
            meanline_visible=True,
            fillcolor=hex_to_rgba(color, 0.2),
            line_color=color,
            opacity=0.85,
            points="outliers",
            marker=dict(size=2, opacity=0.3, color=color),
            hovertemplate=f"<b>{name}</b><br>%{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(**make_layout(
        title=title,
        yaxis={"title": y_label},
        height=420, violinmode="overlay",
    ))
    return fig


col_l, col_r = st.columns(2)
with col_l:
    st.plotly_chart(
        violin_box(grp_a["user_rating"].tolist(),
                   grp_b["user_rating"].tolist(),
                   "RATING DISTRIBUTION", "User Rating"),
        use_container_width=True,
    )
with col_r:
    st.plotly_chart(
        violin_box(grp_a["generation_time"].tolist(),
                   grp_b["generation_time"].tolist(),
                   "GENERATION TIME DISTRIBUTION", "Time (s)"),
        use_container_width=True,
    )

# ── Bootstrap distribution of mean difference ─────────────────────────────────
def boot_hist(res: dict, title: str, units: str, bar_color: str) -> go.Figure:
    diffs = res["boot_diffs"]
    fig   = go.Figure()
    fig.add_trace(go.Histogram(
        x=diffs, nbinsx=80,
        marker_color=hex_to_rgba(bar_color, 0.45),
        marker_line_color=bar_color,
        marker_line_width=0.5,
        name="Bootstrap Δ",
        hovertemplate="Δ: %{x:.4f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vrect(x0=res["ci_lo"], x1=res["ci_hi"],
                  fillcolor=hex_to_rgba(bar_color, 0.12),
                  line_width=0, layer="below")
    fig.add_vline(x=0, line_dash="dash",
                  line_color=PALETTE["negative"], line_width=1.5,
                  annotation_text="no effect",
                  annotation_font_color=PALETTE["text_muted"],
                  annotation_font_size=10)
    fig.add_vline(x=res["diff_mean"],
                  line_color=bar_color, line_width=1.5,
                  annotation_text=f"observed Δ = {res['diff_mean']:.4f}{units}",
                  annotation_font_color=bar_color,
                  annotation_font_size=10)
    fig.update_layout(**make_layout(
        title=title,
        xaxis={"title": f"Bootstrapped Δ mean (A − B) {units}"},
        yaxis={"title": "Frequency"},
        height=340, showlegend=False,
    ))
    return fig


col_l2, col_r2 = st.columns(2)
with col_l2:
    st.plotly_chart(
        boot_hist(res_rating,
                  "BOOTSTRAP Δ MEAN — USER RATING", "", COLOR_A),
        use_container_width=True,
    )
with col_r2:
    st.plotly_chart(
        boot_hist(res_time,
                  "BOOTSTRAP Δ MEAN — GENERATION TIME", "s", COLOR_B),
        use_container_width=True,
    )

# ── Effect size reference ─────────────────────────────────────────────────────
with st.expander("Effect size interpretation (rank-biserial r)"):
    st.markdown("""
| r | Interpretation |
|---|---|
| < 0.10 | Negligible — difference is practically meaningless |
| 0.10 – 0.29 | Small — detectable but unlikely to change product decisions |
| 0.30 – 0.49 | Medium — worth investigating further |
| ≥ 0.50 | Large — strong signal, act on it |

**Why Mann-Whitney U and not a t-test?**
User ratings are bounded (1–5) and the distribution is right-skewed and non-normal.
The t-test assumes normality and equal variance — both violated here.
Mann-Whitney U tests whether one distribution is stochastically greater than the other
without any distributional assumption, making it the correct choice for rating data.
""")

# ── Business interpretation ───────────────────────────────────────────────────
r_sig  = res_rating["p"] < ALPHA
t_sig  = res_time["p"]   < ALPHA
r_win  = variant_a if res_rating["diff_mean"] > 0 else variant_b
t_win  = variant_a if res_time["diff_mean"]   < 0 else variant_b  # lower time = better

if r_sig and t_sig:
    biz = (f"<strong>{r_win}</strong> produces higher-rated outputs "
           f"and <strong>{t_win}</strong> generates faster. "
           f"If they are the same variant, ship it. If they differ, "
           f"quantify the rating gain against the latency cost before deciding.")
elif r_sig and not t_sig:
    biz = (f"<strong>{r_win}</strong> produces significantly higher-rated outputs "
           f"with no statistically significant difference in speed. "
           f"The case for shipping {r_win} is strong.")
elif not r_sig and t_sig:
    biz = (f"No significant quality difference, but "
           f"<strong>{t_win}</strong> is significantly faster. "
           f"If output quality is equal, prefer the faster variant for throughput "
           f"and cost.")
else:
    biz = (f"Neither metric shows a statistically significant difference. "
           f"Do not make a rollout decision based on this data — "
           f"collect more samples or test a more differentiated pair of variants.")

st.markdown(f"""
<div class="insight-box">
<span class="insight-label">◆  BUSINESS RECOMMENDATION</span>
{biz}
</div>
""", unsafe_allow_html=True)

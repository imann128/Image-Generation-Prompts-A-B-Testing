"""
pages/keyword_analysis.py
Q: Which style keywords appear in high-rated vs low-rated prompts?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.loader import (
    CAT_ORDER, PALETTE, make_layout,
    apply_filters, inject_css, load_data,
)

st.set_page_config(page_title="Keyword Analysis", page_icon="◆", layout="wide")
inject_css(metric_neon="#E8621A", insight_neon="#E8621A")

df_raw = load_data()
df     = apply_filters(df_raw, sidebar_key="keywords")

st.markdown("""
<div class="page-header">
  <div class="page-title">Keyword Analysis</div>
  <div class="page-subtitle">TF-IDF TERM FREQUENCY: HIGH-RATED vs LOW-RATED PROMPTS</div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

_MONO = "'Share Tech Mono', 'Courier New', monospace"

st.sidebar.markdown("---")
st.sidebar.markdown("#### Keyword thresholds")
high_thresh = st.sidebar.slider("High-rated threshold (>=)", 3.5, 5.0, 4.0, 0.1, key="high_thresh")
low_thresh  = st.sidebar.slider("Low-rated threshold (<=)",  1.0, 3.0, 2.5, 0.1, key="low_thresh")
top_n       = st.sidebar.slider("Top N keywords", 10, 40, 20, key="top_n")

high_df = df[df["user_rating"] >= high_thresh]
low_df  = df[df["user_rating"] <= low_thresh]

# ── Auto-widen if either group is too small for TF-IDF ────────────────────
# Real data ratings cluster tightly (e.g. DiffusionDB is skewed high), so
# user-chosen thresholds can leave the low group nearly empty.  Fall back to
# the 60th / 35th percentiles which always yield a balanced split regardless
# of the distribution, and surface exactly what was changed.
_threshold_adjusted = False
_adj_high = high_thresh
_adj_low  = low_thresh

if len(high_df) < 50 or len(low_df) < 50:
    _adj_high = round(float(df["user_rating"].quantile(0.60)), 1)
    _adj_low  = round(float(df["user_rating"].quantile(0.35)), 1)
    high_df   = df[df["user_rating"] >= _adj_high]
    low_df    = df[df["user_rating"] <= _adj_low]
    _threshold_adjusted = True

if _threshold_adjusted:
    st.info(
        f"**Thresholds auto-adjusted.** Chosen thresholds "
        f"(≥ {high_thresh} / ≤ {low_thresh}) left the low-rated group with "
        f"fewer than 50 prompts — not enough for a stable TF-IDF comparison. "
        f"Thresholds were widened to the 60th / 35th percentiles of the "
        f"filtered data: **≥ {_adj_high} / ≤ {_adj_low}**. "
        f"Adjust the sidebar sliders to override."
    )

col1, col2, col3, col4 = st.columns(4)
col1.metric("High-rated prompts", f"{len(high_df):,}")
col2.metric("Low-rated prompts",  f"{len(low_df):,}")
col3.metric("High threshold",     f">= {_adj_high}")
col4.metric("Low threshold",      f"<= {_adj_low}")

st.markdown("<hr style='border-color:#252018;margin:20px 0;'>", unsafe_allow_html=True)


@st.cache_data(show_spinner="Running TF-IDF …", ttl=300)
def compute_tfidf(high_prompts: tuple, low_prompts: tuple, top_k: int):
    all_prompts = list(high_prompts) + list(low_prompts)
    n_high      = len(high_prompts)
    vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=3000, sublinear_tf=True,
        min_df=5, stop_words="english", strip_accents="unicode",
        token_pattern=r"[a-zA-Z][a-zA-Z\s\-+]{2,}",
    )
    tfidf_matrix = vec.fit_transform(all_prompts)
    terms        = np.array(vec.get_feature_names_out())
    high_mean    = np.asarray(tfidf_matrix[:n_high].mean(axis=0)).flatten()
    low_mean     = np.asarray(tfidf_matrix[n_high:].mean(axis=0)).flatten()
    diff         = high_mean - low_mean

    def to_df(indices, scores_a, scores_b, label_a, label_b):
        return pd.DataFrame({
            "term":   terms[indices],
            label_a:  scores_a[indices].round(5),
            label_b:  scores_b[indices].round(5),
            "diff":   (scores_a - scores_b)[indices].round(5),
        })

    return {
        "top_high": to_df(np.argsort(high_mean)[-top_k:][::-1], high_mean, low_mean, "high_tfidf", "low_tfidf"),
        "top_low":  to_df(np.argsort(low_mean)[-top_k:][::-1],  low_mean, high_mean, "low_tfidf",  "high_tfidf"),
        "diff_pos": to_df(np.argsort(diff)[-top_k:][::-1],       high_mean, low_mean, "high_tfidf", "low_tfidf"),
        "diff_neg": to_df(np.argsort(diff)[:top_k],              low_mean,  high_mean, "low_tfidf",  "high_tfidf"),
    }


results = compute_tfidf(
    tuple(high_df["prompt"].tolist()),
    tuple(low_df["prompt"].tolist()),
    top_n,
)


def hbar(df_terms, score_col, title, color):
    df_s = df_terms.sort_values(score_col)
    fig  = go.Figure(go.Bar(
        x=df_s[score_col], y=df_s["term"], orientation="h",
        marker_color=color, marker_opacity=0.85,
        marker_line_color="#000", marker_line_width=0.5,
        text=df_s[score_col].map(lambda v: f"{v:.4f}"),
        textposition="outside",
        textfont=dict(size=9, family=_MONO, color=PALETTE["text"]),
        hovertemplate="%{y}<br>TF-IDF: %{x:.5f}<extra></extra>",
    ))
    fig.update_layout(**make_layout(
        title=title,
        xaxis={"title": "Mean TF-IDF Score", "automargin": True},
        yaxis={"title": ""},
        height=max(350, top_n * 22), showlegend=False,
        margin=dict(t=50, b=30, l=200, r=80),
    ))
    return fig


fig_high = hbar(results["top_high"], "high_tfidf",
                f"TOP {top_n} TERMS — HIGH-RATED PROMPTS (>={_adj_high})",
                 "#E8621A")
fig_low  = hbar(results["top_low"], "low_tfidf",
                f"TOP {top_n} TERMS — LOW-RATED PROMPTS (<={_adj_low})",
                 "#C9A84C")

pos_sorted = results["diff_pos"].sort_values("diff", ascending=False)
neg_sorted = results["diff_neg"].sort_values("diff", ascending=True)

fig_div = go.Figure()
fig_div.add_trace(go.Bar(
    x=pos_sorted["diff"], y=pos_sorted["term"], orientation="h",
    name="Stronger in HIGH-rated",
    marker_color="#E8621A", marker_opacity=0.85,
    hovertemplate="%{y}<br>Delta: +%{x:.5f}<extra></extra>",
))
fig_div.add_trace(go.Bar(
    x=neg_sorted["diff"], y=neg_sorted["term"], orientation="h",
    name="Stronger in LOW-rated",
    marker_color="#C9A84C", marker_opacity=0.85,
    hovertemplate="%{y}<br>Delta: %{x:.5f}<extra></extra>",
))
fig_div.update_layout(**make_layout(
    title=f"DIFFERENTIAL TF-IDF: TOP {top_n} TERMS PER DIRECTION  (≥{_adj_high} vs ≤{_adj_low})",
    xaxis={"title": "Delta Mean TF-IDF (High - Low)"},
    yaxis={"title": ""},
    height=max(400, top_n * 28), barmode="relative",
    margin=dict(t=50, b=30, l=200, r=60),
))


@st.cache_data(show_spinner="Per-category TF-IDF …", ttl=300)
def compute_cat_tfidf(prompts_by_cat: dict, top_k: int = 10) -> pd.DataFrame:
    rows = []
    for cat, prompts in prompts_by_cat.items():
        if len(prompts) < 30:
            continue
        vec = TfidfVectorizer(
            ngram_range=(1, 2), max_features=1000, sublinear_tf=True,
            min_df=3, stop_words="english",
            token_pattern=r"[a-zA-Z][a-zA-Z\s\-+]{2,}",
        )
        try:
            mat   = vec.fit_transform(prompts)
            terms = vec.get_feature_names_out()
            means = np.asarray(mat.mean(axis=0)).flatten()
            top_i = np.argsort(means)[-top_k:][::-1]
            for rank, i in enumerate(top_i):
                rows.append({
                    "category": cat, "rank": rank + 1,
                    "term": terms[i], "tfidf": round(float(means[i]), 5),
                })
        except ValueError:
            pass
    return pd.DataFrame(rows)


prompts_by_cat = {cat: df[df["category"] == cat]["prompt"].tolist() for cat in CAT_ORDER}
cat_tfidf      = compute_cat_tfidf(prompts_by_cat, top_k=10)

amber_shades = {
    "portrait":     "#E96D2A",
    "landscape":    "#C9A84C",
    "fantasy":      "#EB8912",
    "architecture": "#A0522D",
    "abstract":     "#D4830F",
}
fig_cat = go.Figure()
for cat in CAT_ORDER:
    sub = cat_tfidf[cat_tfidf["category"] == cat].sort_values("tfidf")
    if sub.empty:
        continue
    fig_cat.add_trace(go.Bar(
        x=sub["tfidf"], y=sub["term"], orientation="h", name=cat,
        marker_color=amber_shades[str(cat)], marker_opacity=0.8,
        visible=(cat == CAT_ORDER[0]),
        hovertemplate=f"<b>{cat}</b><br>%{{y}}<br>TF-IDF: %{{x:.5f}}<extra></extra>",
    ))

buttons = [
    dict(label=cat, method="update",
         args=[{"visible": [c == cat for c in CAT_ORDER]}])
    for cat in CAT_ORDER
]
fig_cat.update_layout(**make_layout(
    title="TOP 10 TF-IDF TERMS BY CATEGORY",
    xaxis={"title": "Mean TF-IDF"},
    yaxis={"title": ""},
    height=380, showlegend=False,
    updatemenus=[dict(
        buttons=buttons, direction="down", showactive=True,
        x=0.01, xanchor="left", y=1.15, yanchor="top",
        bgcolor=PALETTE["surface"], bordercolor=PALETTE["border"],
        font=dict(color=PALETTE["text"], size=11, family=_MONO),
    )],
    margin=dict(t=80, b=30, l=180, r=30),
))

col_l, col_r = st.columns([1, 1])
with col_l:
    st.plotly_chart(fig_high, use_container_width=True)
with col_r:
    st.plotly_chart(fig_low,  use_container_width=True)

st.plotly_chart(fig_div, use_container_width=True)
st.plotly_chart(fig_cat, use_container_width=True)

with st.expander("Raw TF-IDF tables"):
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Positive differentials (HIGH > LOW)**")
        st.dataframe(results["diff_pos"].sort_values("diff", ascending=False),
                     use_container_width=True, hide_index=True)
    with t2:
        st.markdown("**Negative differentials (LOW > HIGH)**")
        st.dataframe(results["diff_neg"].sort_values("diff", ascending=True),
                     use_container_width=True, hide_index=True)

st.markdown("""
<div class="insight-box">
<span class="insight-label">◆  INSIGHT</span>
<strong>BUSINESS DECISION</strong><br>
Terms with high positive differential — one-click suggestions in the prompt composer.
Terms with high negative differential — soft warnings in the UI.
Category breakdown enables personalised suggestions per style.
</div>
""", unsafe_allow_html=True)

"""
pages/prompt_length.py
Q: What is the relationship between prompt length and output rating?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as scipy_stats

from utils.loader import (
    CAT_ORDER, PALETTE, make_layout,
    apply_filters, inject_css, load_data,
)

st.set_page_config(page_title="Prompt Length", page_icon="◆", layout="wide")
inject_css(metric_neon="#E8621A", insight_neon="#D1855D")

df_raw = load_data()
df     = apply_filters(df_raw, sidebar_key="prompt_len")

st.markdown("""
<div class="page-header">
  <div class="page-title">Prompt Length vs Output Quality</div>
  <div class="page-subtitle">WORD COUNT · RATING CORRELATION · OPTIMAL RANGE DETECTION</div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

_MONO = "'Share Tech Mono', 'Courier New', monospace"

BINS   = [0, 10, 15, 20, 25, 30, 35, 40, 50, 100]
LABELS = ["1-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-50", "51+"]

df = df.copy()
df["len_bin"]    = df["prompt_length"].clip(upper=100)
df["len_bucket"] = df["len_bin"].apply(
    lambda x: LABELS[next(i for i, b in enumerate(BINS[1:]) if x <= b)]
    if x <= BINS[-1] else LABELS[-1]
)

bin_stats = (
    df.groupby("len_bucket")["user_rating"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "avg", "std": "std", "count": "n"})
)
bin_stats["se"]    = bin_stats["std"] / bin_stats["n"] ** 0.5
bin_stats["ci_lo"] = bin_stats["avg"] - 1.96 * bin_stats["se"]
bin_stats["ci_hi"] = bin_stats["avg"] + 1.96 * bin_stats["se"]
bin_stats          = (
    bin_stats.set_index("len_bucket")
    .reindex(LABELS)
    .reset_index()
    .dropna(subset=["avg"])
)

best_bucket = bin_stats.loc[bin_stats["avg"].idxmax(), "len_bucket"]
best_avg    = bin_stats["avg"].max()
corr, p_val = scipy_stats.spearmanr(df["prompt_length"], df["user_rating"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Optimal Bucket",    best_bucket + " words")
c2.metric("Avg Rating (best)", f"{best_avg:.3f}")
c3.metric("Spearman rho",      f"{corr:.4f}")
c4.metric("p-value",           f"{p_val:.4f}")

st.markdown("<hr style='border-color:#252018;margin:20px 0;'>", unsafe_allow_html=True)

colors = [
    "#E8621A" if b == best_bucket else "#D1855D"
    for b in bin_stats["len_bucket"]
]

# ── Bar chart ──────────────────────────────────────────────────────────────
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=bin_stats["len_bucket"], y=bin_stats["avg"],
    error_y=dict(
        type="data",
        array=(bin_stats["ci_hi"] - bin_stats["avg"]).tolist(),
        arrayminus=(bin_stats["avg"] - bin_stats["ci_lo"]).tolist(),
        visible=True, color="#333333", thickness=1.5, width=4,
    ),
    marker_color=colors, marker_line_color="#000", marker_line_width=1,
    text=bin_stats["avg"].map(lambda x: f"{x:.3f}"),
    textposition="outside",
    textfont=dict(size=10, family=_MONO, color=PALETTE["text"]),
    hovertemplate="<b>%{x} words</b><br>Avg: %{y:.3f}<br>n=%{customdata:,}<extra></extra>",
    customdata=bin_stats["n"].tolist(),
))
fig_bar.update_layout(**make_layout(
    title="AVG USER RATING BY PROMPT LENGTH BUCKET (95% CI)",
    yaxis={"title": "Avg User Rating", "range": [3.5, 4.4]},
    xaxis={"title": "Prompt Length (words)"},
    showlegend=False, height=400,
))

# ── Scatter with rolling mean ──────────────────────────────────────────────
sample   = df.sample(min(6000, len(df)), random_state=42)
sorted_s = sample.sort_values("prompt_length")
window   = max(100, len(sorted_s) // 20)
roll_x   = sorted_s["prompt_length"].rolling(window, center=True, min_periods=10).mean()
roll_y   = sorted_s["user_rating"].rolling(window,   center=True, min_periods=10).mean()

fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=sample["prompt_length"], y=sample["user_rating"],
    mode="markers", name="generation",
    marker=dict(color="#C9A84C", size=3, opacity=0.2, line=dict(width=0)),
    hovertemplate="Length: %{x}<br>Rating: %{y:.2f}<extra></extra>",
))
fig_scatter.add_trace(go.Scatter(
    x=roll_x.dropna(), y=roll_y.dropna(),
    mode="lines", name="rolling mean",
    line=dict(color="#E8621A", width=2.5),
    hoverinfo="skip",
))
fig_scatter.update_layout(**make_layout(
    title="PROMPT LENGTH vs RATING (sample 6K + rolling mean)",
    xaxis={"title": "Prompt Length (words)"},
    yaxis={"title": "User Rating"},
    height=430,
))

# ── Heatmap: length bucket × category ─────────────────────────────────────
cat_len = (
    df.groupby(["len_bucket", "category"], observed=True)["user_rating"]
    .mean()
    .unstack("category")
    .reindex(index=LABELS, columns=CAT_ORDER)
)
fig_heat = go.Figure(go.Heatmap(
    z=cat_len.values,
    x=[str(c) for c in cat_len.columns],
    y=cat_len.index.tolist(),
    zmin=3.5, zmax=4.4,
    colorscale=[
        [0.0,  "#1A1400"],
        [0.3,  "#3A3000"],
        [0.6,  "#7A6800"],
        [0.8,  "#CCB000"],
        [1.0,  "#FFFACC"],
    ],
    showscale=True,
    text=[[f"{v:.2f}" if v == v else "" for v in row] for row in cat_len.values],
    texttemplate="%{text}",
    textfont=dict(size=9, family=_MONO, color="#F0E6D3"),
    hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Avg: %{z:.3f}<extra></extra>",
    colorbar=dict(tickfont=dict(color=PALETTE["text"], family=_MONO, size=9)),
))
fig_heat.update_layout(**make_layout(
    title="AVG RATING: LENGTH BUCKET × CATEGORY",
    xaxis={"title": "Category"},
    yaxis={"title": "Prompt Length (words)"},
    height=420,
))

col_l, col_r = st.columns([1, 1])
with col_l:
    st.plotly_chart(fig_bar,     use_container_width=True)
with col_r:
    st.plotly_chart(fig_scatter, use_container_width=True)

st.plotly_chart(fig_heat, use_container_width=True)

# ── Length distribution histogram ──────────────────────────────────────────
fig_len_hist = go.Figure()
fig_len_hist.add_trace(go.Histogram(
    x=df["prompt_length"].clip(upper=80), nbinsx=40,
    marker_color="#C9A84C", marker_opacity=0.8,
    marker_line_color="#000", marker_line_width=0.5,
    hovertemplate="Words: %{x}<br>Count: %{y}<extra></extra>",
))
fig_len_hist.update_layout(**make_layout(
    title="PROMPT LENGTH DISTRIBUTION (clipped at 80 words)",
    xaxis={"title": "Prompt Length (words)"},
    yaxis={"title": "Count"},
    height=300, showlegend=False,
))
st.plotly_chart(fig_len_hist, use_container_width=True)

st.markdown(f"""
<div class="insight-box">
<span class="insight-label">◆  INSIGHT</span>
<strong>BUSINESS DECISION</strong><br>
Optimal bucket: <strong>{best_bucket} words</strong> (avg {best_avg:.3f}).
Add a live word-count indicator to the prompt input with three zones:
too short (&lt;12 words), optimal (12–35), over-specified (&gt;45 words).
Spearman rho = {corr:.3f} (p = {p_val:.4f}) — report this signal strength upward.
</div>
""", unsafe_allow_html=True)

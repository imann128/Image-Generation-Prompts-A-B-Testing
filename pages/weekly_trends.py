"""
pages/weekly_trends.py
Q: How have weekly average ratings trended over the 90-day window?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go
import streamlit as st
from scipy import stats as scipy_stats

from utils.loader import (
    CAT_COLORS, CAT_ORDER, MODEL_COLORS, MODEL_ORDER,
    PALETTE, make_layout, apply_filters, inject_css, load_data,
)

st.set_page_config(page_title="Weekly Trends", page_icon="◆", layout="wide")
inject_css(metric_neon="#E19238", insight_neon="#CE9C05")

df_raw = load_data()
df     = apply_filters(df_raw, sidebar_key="trends")

st.markdown("""
<div class="page-header">
  <div class="page-title">Weekly Quality Trends</div>
  <div class="page-subtitle">90-DAY RATING TRAJECTORY · MODEL AND CATEGORY BREAKDOWNS</div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

_MONO = "'Share Tech Mono', 'Courier New', monospace"

weekly_all = (
    df.groupby("week")["user_rating"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "avg", "std": "std", "count": "n"})
)
weekly_all["se"]    = weekly_all["std"] / weekly_all["n"] ** 0.5
weekly_all["ci_lo"] = weekly_all["avg"] - 1.96 * weekly_all["se"]
weekly_all["ci_hi"] = weekly_all["avg"] + 1.96 * weekly_all["se"]
weekly_all          = weekly_all.sort_values("week")

if len(weekly_all) > 2:
    slope, intercept, r, p_val, _ = scipy_stats.linregress(
        weekly_all["week"], weekly_all["avg"]
    )
else:
    slope, intercept, r, p_val = 0, 0, 0, 1

first_avg = weekly_all.iloc[0]["avg"]  if len(weekly_all) > 0 else 0
last_avg  = weekly_all.iloc[-1]["avg"] if len(weekly_all) > 0 else 0
delta     = last_avg - first_avg
trend_dir = "IMPROVING" if slope > 0.002 else ("DECLINING" if slope < -0.002 else "STABLE")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trend Direction",   trend_dir)
c2.metric("First to Last",     f"{delta:+.3f}")
c3.metric("OLS Slope / week",  f"{slope:+.4f}")
c4.metric("Trend R²",          f"{r**2:.3f}")

st.markdown("<hr style='border-color:#252018;margin:20px 0;'>", unsafe_allow_html=True)

weeks   = weekly_all["week"].tolist()
trend_y = [intercept + slope * w for w in weeks]

# ── Overall trend chart ────────────────────────────────────────────────────
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=weeks + weeks[::-1],
    y=weekly_all["ci_hi"].tolist() + weekly_all["ci_lo"].tolist()[::-1],
    fill="toself",
    fillcolor="rgba(232,98,26,0.08)",
    line_color="rgba(0,0,0,0)",
    showlegend=True, name="95% CI", hoverinfo="skip",
))
fig_trend.add_trace(go.Scatter(
    x=weeks, y=weekly_all["avg"],
    mode="lines+markers", name="Weekly avg rating",
    line=dict(color="#E19238", width=2.5),
    marker=dict(size=7, color="#CE9C05",
                line=dict(width=1, color="#000")),
    hovertemplate="Week %{x}<br>Avg: %{y:.3f}<extra></extra>",
))
fig_trend.add_trace(go.Scatter(
    x=weeks, y=trend_y, mode="lines",
    name=f"OLS trend ({slope:+.4f}/wk)",
    line=dict(color=PALETTE["accent2"], width=1.5, dash="dot"),
    hoverinfo="skip",
))
fig_trend.update_layout(**make_layout(
    title="WEEKLY AVERAGE RATING — OVERALL TREND",
    yaxis={"title": "Avg User Rating", "range": [3.5, 4.4]},
    xaxis={"title": "ISO Week Number"},
    height=420,
))

# ── By model ───────────────────────────────────────────────────────────────
weekly_model = (
    df.groupby(["week", "model_version"], observed=True)["user_rating"]
    .mean().reset_index()
    .rename(columns={"user_rating": "avg_rating"})
    .sort_values("week")
)
fig_model = go.Figure()
for model in MODEL_ORDER:
    sub = weekly_model[weekly_model["model_version"] == model]
    if sub.empty:
        continue
    color = MODEL_COLORS.get(model, "#C9A84C")
    fig_model.add_trace(go.Scatter(
        x=sub["week"], y=sub["avg_rating"],
        mode="lines+markers", name=model,
        line=dict(color=color, width=2),
        marker=dict(size=5, color=color),
        hovertemplate=f"<b>{model}</b><br>Week %{{x}}<br>Avg: %{{y:.3f}}<extra></extra>",
    ))
fig_model.update_layout(**make_layout(
    title="WEEKLY RATING BY MODEL VERSION",
    yaxis={"title": "Avg User Rating"},
    xaxis={"title": "ISO Week Number"},
    height=420,
))

# ── Volume bar ─────────────────────────────────────────────────────────────
fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(
    x=weekly_all["week"], y=weekly_all["n"],
    marker_color="#E19238", marker_opacity=0.7,
    marker_line_color="#000000", marker_line_width=0.5,
    hovertemplate="Week %{x}<br>Gens: %{y:,}<extra></extra>",
))
fig_vol.update_layout(**make_layout(
    title="WEEKLY GENERATION VOLUME",
    yaxis={"title": "Generations"},
    xaxis={"title": "ISO Week Number"},
    height=300, showlegend=False,
))

# ── By category ────────────────────────────────────────────────────────────
weekly_cat = (
    df.groupby(["week", "category"], observed=True)["user_rating"]
    .mean().reset_index()
    .rename(columns={"user_rating": "avg_rating"})
    .sort_values("week")
)
fig_cat = go.Figure()
for cat in CAT_ORDER:
    sub = weekly_cat[weekly_cat["category"] == cat]
    if sub.empty:
        continue
    fig_cat.add_trace(go.Scatter(
        x=sub["week"], y=sub["avg_rating"],
        mode="lines+markers", name=cat,
        line=dict(color=CAT_COLORS[cat], width=2),
        marker=dict(size=5, color=CAT_COLORS[cat]),
        hovertemplate=f"<b>{cat}</b><br>Week %{{x}}<br>Avg: %{{y:.3f}}<extra></extra>",
    ))
fig_cat.update_layout(**make_layout(
    title="WEEKLY RATING BY CATEGORY",
    yaxis={"title": "Avg User Rating"},
    xaxis={"title": "ISO Week Number"},
    height=420,
))

st.plotly_chart(fig_trend, use_container_width=True)

col_l, col_r = st.columns([1, 1])
with col_l:
    st.plotly_chart(fig_model, use_container_width=True)
with col_r:
    st.plotly_chart(fig_cat,   use_container_width=True)

st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("#### Weekly summary table")
display = weekly_all[["week", "avg", "std", "n", "ci_lo", "ci_hi"]].copy()
display.columns = ["week", "avg_rating", "std_dev", "n_gens", "ci_lo", "ci_hi"]
st.dataframe(display.round(3), use_container_width=True, hide_index=True)

st.markdown("""
<div class="insight-box">
<span class="insight-label">◆  INSIGHT</span>
<strong>BUSINESS DECISION</strong><br>
Set an automated alert: if the 7-day rolling average drops &gt; 0.1 points below the prior
4-week mean, trigger a model health review. A sudden week with both low rating and high
volume is the most dangerous signal — the problem is reaching the majority of users.
</div>
""", unsafe_allow_html=True)

"""
pages/generation_latency.py
Q: What is the distribution of generation times, and where are the outliers?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go
import streamlit as st

from utils.loader import (
    CAT_ORDER, MODEL_ORDER, PALETTE,
    make_layout, SAMPLER_ORDER, apply_filters, inject_css, load_data,
)

st.set_page_config(page_title="Generation Latency", page_icon="◆", layout="wide")
inject_css(metric_neon="#E8621A", insight_neon="#E8621A")

df_raw = load_data()
df     = apply_filters(df_raw, sidebar_key="latency")

st.markdown("""
<div class="page-header">
  <div class="page-title">Generation Latency</div>
  <div class="page-subtitle">DISTRIBUTION · OUTLIER DETECTION · SAMPLER ANALYSIS</div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

_MONO = "'Share Tech Mono', 'Courier New', monospace"

p50      = df["generation_time"].median()
p95      = df["generation_time"].quantile(0.95)
p99      = df["generation_time"].quantile(0.99)
pct_slow = (df["generation_time"] > 5.0).mean() * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Median Gen Time", f"{p50:.2f}s")
c2.metric("p95 Gen Time",    f"{p95:.2f}s")
c3.metric("p99 Gen Time",    f"{p99:.2f}s")
c4.metric("% Gens > 5s",     f"{pct_slow:.1f}%")

st.markdown("<hr style='border-color:#252018;margin:20px 0;'>", unsafe_allow_html=True)

# ── Histogram ──────────────────────────────────────────────────────────────
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=df["generation_time"], nbinsx=80,
    marker_color="#E8621A",
    marker_line_color="#000000", marker_line_width=0.5,
    opacity=0.85,
    hovertemplate="Time: %{x:.2f}s<br>Count: %{y}<extra></extra>",
))
for pct, val, col in [(50, p50, PALETTE["positive"]),
                       (95, p95, PALETTE["accent"]),
                       (99, p99, PALETTE["negative"])]:
    fig_hist.add_vline(
        x=val, line_dash="dash", line_color=col, line_width=1.5,
        annotation_text=f"p{pct}={val:.2f}s",
        annotation_position="top right" if pct < 99 else "top left",
        annotation_font_color=col, annotation_font_size=10,
        annotation_font=dict(family=_MONO),
    )
fig_hist.update_layout(**make_layout(
    title="GENERATION TIME DISTRIBUTION",
    xaxis={"title": "Generation Time (s)"},
    yaxis={"title": "Count"},
    height=380, showlegend=False,
))

# ── Box plot by sampler 
sampler_colors = {
    "k_lms":      "#E8621A",
    "k_euler_a":  "#C9A84C",
    "k_euler":    "#A0522D",
    "k_dpm_2_a":  "#FEA63B",
    "k_dpm_2":    "#D4830F",
    "plms":       "#00F5FF",
    "ddim":       "#FF2D78",
}

fig_box = go.Figure()
for sampler in SAMPLER_ORDER:
    sub = df[df["sampler"] == sampler]["generation_time"]
    if sub.empty:
        continue
    fig_box.add_trace(go.Box(
        y=sub, name=sampler,
        marker_color=sampler_colors.get(sampler, PALETTE["accent"]),
        line_color="#000000",
        fillcolor=sampler_colors.get(sampler, PALETTE["accent"]),
        opacity=0.65,
        boxpoints="outliers", marker_size=2, marker_opacity=0.4,
    ))
fig_box.add_hline(
    y=p95, line_dash="dot", line_color=PALETTE["negative"],
    annotation_text="p95", annotation_font_color=PALETTE["negative"],
    annotation_font_size=10, annotation_font=dict(family=_MONO),
)
fig_box.update_layout(**make_layout(
    title="GENERATION TIME BY SAMPLER",
    yaxis={"title": "Generation Time (s)"},
    xaxis={"title": "Sampler"},
    height=400, showlegend=False,
))

# ── Scatter: steps vs gen time ─────────────────────────────────────────────
sample = df.sample(min(8000, len(df)), random_state=42)
amber_model_colors = {
    "sd-v1-4":   "#A0522D",
    "sd-v1-5":   "#C9A84C",
    "sd-v2-0":   "#E8621A",
    "sd-v2-1":   "#EB8912",
    "sdxl-base": "#D4830F",
}
fig_scatter = go.Figure()
for model in MODEL_ORDER:
    sub = sample[sample["model_version"] == model]
    if sub.empty:
        continue
    fig_scatter.add_trace(go.Scatter(
        x=sub["steps"], y=sub["generation_time"],
        mode="markers", name=model,
        marker=dict(color=amber_model_colors[str(model)], size=4, opacity=0.4,
                    line=dict(width=0)),
        hovertemplate=f"<b>{model}</b><br>Steps: %{{x}}<br>Time: %{{y:.2f}}s<extra></extra>",
    ))
fig_scatter.update_layout(**make_layout(
    title="STEPS vs GENERATION TIME (sample 8K)",
    xaxis={"title": "Steps"},
    yaxis={"title": "Generation Time (s)"},
    height=430,
))

# ── p95 heatmap ────────────────────────────────────────────────────────────
p95_pivot = (
    df.groupby(["category", "model_version"], observed=True)["generation_time"]
    .quantile(0.95)
    .unstack("model_version")
    .reindex(index=CAT_ORDER, columns=MODEL_ORDER)
)
fig_heat = go.Figure(go.Heatmap(
    z=p95_pivot.values,
    x=[str(m) for m in p95_pivot.columns],
    y=[str(c) for c in p95_pivot.index],
    colorscale=[
    [0.0, "#A18347"],
    [0.35, "#6B3A10"],
    [0.65, "#C9A84C"],
    [1.0, "#F0E6D3"],
],
    showscale=True,
    text=[[f"{v:.1f}s" if v == v else "" for v in row] for row in p95_pivot.values],
    texttemplate="%{text}",
    textfont=dict(size=10, family=_MONO, color="#060503"),
    hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>p95: %{z:.2f}s<extra></extra>",
    colorbar=dict(tickfont=dict(color=PALETTE["text"], family=_MONO, size=10),
                  ticksuffix="s"),
))
fig_heat.update_layout(**make_layout(
    title="P95 LATENCY: CATEGORY × MODEL (seconds)",
    height=380,
))

st.plotly_chart(fig_hist, use_container_width=True)

col_l, col_r = st.columns([1, 1])
with col_l:
    st.plotly_chart(fig_box,  use_container_width=True)
with col_r:
    st.plotly_chart(fig_heat, use_container_width=True)

st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("#### Slowest 50 generations")
outliers = (
    df.nlargest(50, "generation_time")
    [["generation_time", "model_version", "category", "sampler",
      "steps", "width", "height", "prompt_length"]]
    .rename(columns={"generation_time": "gen_time_s"})
    .reset_index(drop=True)
)
outliers["gen_time_s"] = outliers["gen_time_s"].round(3)
st.dataframe(outliers, use_container_width=True, hide_index=True)

st.markdown("""
<div class="insight-box">
<span class="insight-label">◆  INSIGHT</span>
<strong>BUSINESS DECISION</strong><br>
p95 heatmap cells &gt; 4s are model-category pairs where users wait long enough to disengage.
Prioritise inference optimisation (quantisation, batching) for those combinations,
or set a hard generation timeout at steps &gt; 40 / high resolution.
</div>
""", unsafe_allow_html=True)

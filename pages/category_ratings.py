"""
pages/category_ratings.py
Q: Which prompt categories generate the highest average user ratings,
   and does that vary by model version?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go
import streamlit as st

from utils.loader import (
    CAT_ORDER, MODEL_ORDER,
    PALETTE, make_layout, apply_filters, inject_css, load_data,
)

st.set_page_config(page_title="Category Ratings", page_icon="◆", layout="wide")
inject_css(metric_neon="#39FF14", insight_neon="#39FF14")

df_raw = load_data()
df     = apply_filters(df_raw, sidebar_key="cat_ratings")

st.markdown("""
<div class="page-header">
  <div class="page-title">Category Ratings</div>
  <div class="page-subtitle">AVG USER SATISFACTION BY PROMPT CATEGORY AND MODEL VERSION</div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

overall_avg = df["user_rating"].mean()
best_cat    = df.groupby("category")["user_rating"].mean().idxmax()
best_model  = df.groupby("model_version")["user_rating"].mean().idxmax()
total_gens  = len(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall Avg Rating", f"{overall_avg:.3f} / 5.0")
c2.metric("Best Category",       str(best_cat).upper())
c3.metric("Best Model",          str(best_model))
c4.metric("Filtered Gens",       f"{total_gens:,}")

st.markdown("<hr style='border-color:#252018;margin:20px 0;'>", unsafe_allow_html=True)

_MONO = "'Share Tech Mono', 'Courier New', monospace"

cat_avg = (
    df.groupby("category", observed=True)["user_rating"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "avg_rating", "std": "std_dev", "count": "n"})
    .sort_values("avg_rating", ascending=False)
)
cat_avg["se"]    = cat_avg["std_dev"] / (cat_avg["n"] ** 0.5)
cat_avg["color"] = "#DFAF2C"

# cat_avg["category"].map(CAT_COLORS)

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=cat_avg["category"],
    y=cat_avg["avg_rating"],
    error_y=dict(type="data", array=cat_avg["se"] * 1.96, visible=True,
                 color="#333333", thickness=1.5, width=4),
    # marker_color=cat_avg["color"].tolist(),
    marker_color="#EB8912",
    marker_line_color="#000000",
    marker_line_width=1,
    text=cat_avg["avg_rating"].map(lambda x: f"{x:.3f}"),
    textposition="outside",
    textfont=dict(size=11, family=_MONO, color=PALETTE["text"]),
    hovertemplate="<b>%{x}</b><br>Avg rating: %{y:.3f}<extra></extra>",
))
fig_bar.add_hline(
    y=overall_avg, line_dash="dot", line_color=PALETTE["accent2"], line_width=1,
    annotation_text=f"avg {overall_avg:.2f}",
    annotation_position="top right",
    annotation_font_color=PALETTE["accent2"],
)

fig_bar.update_layout(**make_layout(
    title="AVG RATING BY CATEGORY (95% CI)",
    yaxis={"range": [3.5, 4.4], "title": "Avg User Rating"},
    xaxis={"title": "Category"},
    showlegend=False,
    height=420,
    margin={"t":55,"b":80,"l":55,"r":25}
))

fig_bar.add_annotation(
    text=" y-axis does not start at zero",
    xref="paper", yref="paper",
    x=0.5, y=-0.25,
    showarrow=False,
    font=dict(size=9, color=PALETTE["text_muted"], family=_MONO),
    align="center",
)

pivot = (
    df.groupby(["category", "model_version"], observed=True)["user_rating"]
    .mean()
    .unstack("model_version")
    .reindex(index=CAT_ORDER, columns=MODEL_ORDER)
)

fig_heat = go.Figure(go.Heatmap(
    z=pivot.values,
    x=[str(m) for m in pivot.columns],
    y=[str(c) for c in pivot.index],
    zmin=3.5, zmax=4.4,
    colorscale=[
        [0.0, "#B9800E"],
        [0.35, "#8B5423"],
        [0.65, "#AF954D"],
        [0.8, "#E5B04D"],
        [1.0, "#EA9E0F"],
    ],
    showscale=True,
    text=[[f"{v:.3f}" if v == v else "" for v in row] for row in pivot.values],
    texttemplate="%{text}",
    # textfont=dict(size=10, family=_MONO, color="#F0E6D3"),
    textfont=dict(size=10, family=_MONO, color="#060503"),
    hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Avg rating: %{z:.3f}<extra></extra>",
    colorbar=dict(tickfont=dict(color=PALETTE["text"], family=_MONO, size=10)),
))
fig_heat.update_layout(**make_layout(
    title="AVG RATING HEATMAP: CATEGORY × MODEL",
    height=380,
))

amber_shades = {
    "portrait": "#AF8228",
    "landscape":    "#C9A84C",
    "fantasy":      "#EB8912",
    "architecture": "#9C3302",
    "abstract":     "#DF9A3A",
}
fig_violin = go.Figure()
for cat in CAT_ORDER:
    sub = df[df["category"] == cat]["user_rating"]
    if sub.empty:
        continue
    fig_violin.add_trace(go.Violin(
        x=[cat] * len(sub), y=sub, name=cat,
        fillcolor=amber_shades[str(cat)], opacity=0.6,
        # line_color="#000000",
        line_color = amber_shades[str(cat)],
        meanline_visible=True, meanline_color=PALETTE["accent2"],
        box_visible=True, box_fillcolor="#0A0A0A",
        showlegend=False,
    ))
fig_violin.update_layout(**make_layout(
    title="RATING DISTRIBUTION BY CATEGORY",
    yaxis={"title": "User Rating"},
    xaxis={"title": "Category"},
    height=400,
))

col_l, col_r = st.columns([1, 1])
with col_l:
    st.plotly_chart(fig_bar,    use_container_width=True)
with col_r:
    st.plotly_chart(fig_heat,   use_container_width=True)

st.plotly_chart(fig_violin, use_container_width=True)

model_cat = (
    df.groupby(["model_version", "category"], observed=True)["user_rating"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "avg_rating", "count": "n_gens"})
)
model_cat["avg_rating"] = model_cat["avg_rating"].round(3)
st.markdown("#### Per-model breakdown")
st.dataframe(
    model_cat.sort_values(["model_version", "avg_rating"], ascending=[True, False]),
    use_container_width=True, hide_index=True,
)

st.markdown("""
<div class="insight-box">
<span class="insight-label">◆  INSIGHT</span>
<strong>BUSINESS DECISION</strong><br>
Which (model, category) pairs to promote as recommended starting points in the discovery UI.
Low-count, high-rating cells are under-explored niches worth more traffic allocation.
</div>
""", unsafe_allow_html=True)

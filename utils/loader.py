"""
utils/loader.py
───────────────
Single point of truth for palette, Plotly layout, CSS injection,
data loading, and sidebar filters.

CSS architecture (three separate st.markdown calls)
────────────────────────────────────────────────────
Call 1 — <link> tags only, no <style>.
  Browser starts DNS+TLS to Google Fonts before Streamlit renders
  any components. Eliminates the font race that caused Share Tech Mono
  and Material Icons to revert to fallbacks on page navigation.
  @import inside <style> is asynchronous and loses this race.

Call 2 — :root { } custom properties only, f-string interpolated.
  This is the ONLY place metric_neon / insight_neon appear in CSS.
  Streamlit never removes injected <style> tags across page navs,
  so any rule that hard-codes a colour value accumulates competing
  copies. :root overrides always resolve to the last injected value,
  which is the correct one for the current page. No race, no leakage.

Call 3 — static theme CSS, plain string (no f-string, no .replace()).
  All colour references use var(--metric-neon) etc.
  Material Icons exclusion block is placed LAST in this stylesheet.
  Reason: when specificity is equal and both rules carry !important,
  source order decides — last wins. Broad rules like
  [data-testid="stSidebar"] * { font-family: monospace !important }
  appear earlier, so the icon block at the end always overrides them.

Plotly layout architecture
──────────────────────────
_XAXIS / _YAXIS are module-level dicts exported as XAXIS_BASE / YAXIS_BASE.
Pages spread them into axis overrides:
  yaxis=dict(**YAXIS_BASE, title="Avg Rating", range=[3.4, 4.5])
This preserves tickfont, title_font, gridcolor etc. without repeating them.
The old approach of spreading PLOTLY_LAYOUT["yaxis"] lost these keys
whenever a page only overrode yaxis but not xaxis.
"""

from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).parent.parent / "data" / "imageart.db"

# ── Canonical orderings ───────────────────────────────────────────────────────
MODEL_ORDER   = ["sd-v1-4", "sd-v1-5", "sd-v2-0", "sd-v2-1", "sdxl-base"]
CAT_ORDER     = ["portrait", "landscape", "fantasy", "architecture", "abstract"]
SAMPLER_ORDER = ["k_lms", "k_euler_a", "k_euler", "k_dpm_2_a", "k_dpm_2", "plms", "ddim"]

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":         "#060503",
    "surface":    "#0D0A07",
    "surface2":   "#131008",
    "border":     "#252018",
    "accent":     "#39FF14",
    "accent2":    "#00F5FF",
    "deco":       "#E8621A",
    "gold":       "#C9A84C",
    "text":       "#F0E6D3",
    "text_muted": "#706860",
    "positive":   "#39FF14",
    "negative":   "#FF2D78",
}

MODEL_COLORS = {
    "sd-v1-4":   "#39FF14",
    "sd-v1-5":   "#00F5FF",
    "sd-v2-0":   "#FF2D78",
    "sd-v2-1":   "#FFE600",
    "sdxl-base": "#FF6E1F",
}

CAT_COLORS = {
    "portrait":     "#39FF14",
    "landscape":    "#00F5FF",
    "fantasy":      "#FF2D78",
    "architecture": "#FFE600",
    "abstract":     "#FF6E1F",
}

# ── Plotly font string ────────────────────────────────────────────────────────
_FONT = "'Share Tech Mono', 'Courier New', monospace"

# ── Axis base dicts — exported so pages can spread without losing tickfont ────
_XAXIS = dict(
    gridcolor=PALETTE["border"],
    linecolor=PALETTE["border"],
    tickcolor=PALETTE["border"],
    zerolinecolor=PALETTE["border"],
    tickfont=dict(family=_FONT, color=PALETTE["text_muted"], size=11),
    title_font=dict(family=_FONT, color=PALETTE["text_muted"], size=11),
)
_YAXIS = dict(
    gridcolor=PALETTE["border"],
    linecolor=PALETTE["border"],
    tickcolor=PALETTE["border"],
    zerolinecolor=PALETTE["border"],
    tickfont=dict(family=_FONT, color=PALETTE["text_muted"], size=11),
    title_font=dict(family=_FONT, color=PALETTE["text_muted"], size=11),
)

# Public aliases for page imports
XAXIS_BASE = _XAXIS
YAXIS_BASE = _YAXIS

# ── Plotly base layout ────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor=PALETTE["bg"],
    font=dict(color=PALETTE["text"], family=_FONT, size=12),
    title_font=dict(color=PALETTE["text"], size=13, family=_FONT),
    legend=dict(
        bgcolor=PALETTE["surface"],
        bordercolor=PALETTE["border"],
        borderwidth=1,
        font=dict(size=11, family=_FONT, color=PALETTE["text"]),
    ),
    xaxis=_XAXIS,
    yaxis=_YAXIS,
    margin=dict(t=55, b=45, l=55, r=25),
    hoverlabel=dict(
        bgcolor=PALETTE["surface2"],
        bordercolor=PALETTE["border"],
        font=dict(family=_FONT, color=PALETTE["text"], size=11),
    ),
)


def make_layout(**overrides) -> dict:
    """
    Return a copy of PLOTLY_LAYOUT with overrides merged in.
    xaxis / yaxis dicts are shallow-merged so tickfont, title_font,
    gridcolor etc. survive — a plain dict assignment would replace them.

    Usage:
        fig.update_layout(**make_layout(
            title="MY CHART",
            yaxis=dict(title="Avg Rating", range=[3.4, 4.5]),
            height=420,
        ))
    """
    result = dict(PLOTLY_LAYOUT)
    for k, v in overrides.items():
        if k in ("xaxis", "yaxis") and isinstance(v, dict):
            result[k] = {**result[k], **v}
        else:
            result[k] = v
    return result


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        st.error(
            f"Database not found at `{DB_PATH}`. "
            "Run `python data/generate_dataset.py` first."
        )
        st.stop()

    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM generations", conn)
    conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = pd.to_datetime(df["date"])

    df["model_version"] = pd.Categorical(
        df["model_version"], categories=MODEL_ORDER,   ordered=True)
    df["category"]      = pd.Categorical(
        df["category"],      categories=CAT_ORDER,     ordered=True)
    # Use SAMPLER_ORDER for known samplers; append any unknown ones so no row
    # is silently turned into NaN by pd.Categorical.
    seen_samplers = df["sampler"].dropna().unique().tolist()
    sampler_cats  = SAMPLER_ORDER + [s for s in seen_samplers if s not in SAMPLER_ORDER]
    df["sampler"] = pd.Categorical(
        df["sampler"], categories=sampler_cats, ordered=True)

    return df


# ── Sidebar filters ───────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame, sidebar_key: str = "") -> pd.DataFrame:
    st.sidebar.markdown("## Filters")

    min_date   = df["date"].min().date()
    max_date   = df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
        key=f"date_{sidebar_key}",
    )

    models = st.sidebar.multiselect(
        "Model version", options=MODEL_ORDER, default=MODEL_ORDER,
        key=f"models_{sidebar_key}",
    )
    cats = st.sidebar.multiselect(
        "Category", options=CAT_ORDER, default=CAT_ORDER,
        key=f"cats_{sidebar_key}",
    )

    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        df = df[(df["date"] >= start) & (df["date"] <= end)]

    if models:
        df = df[df["model_version"].isin(models)]
    if cats:
        df = df[df["category"].isin(cats)]

    return df


def metric_delta(current: float, baseline: float, fmt: str = ".2f") -> str:
    delta = current - baseline
    sign  = "+" if delta >= 0 else ""
    return f"{sign}{delta:{fmt}}"


# ── CSS injection ─────────────────────────────────────────────────────────────

# Static theme CSS — plain string, no f-string, no .replace().
# All per-page colours come from CSS custom properties set in Call 2.
# Material Icons block is intentionally LAST — source order guarantee.
_STATIC_CSS = """
*, *::before, *::after { box-sizing: border-box; }

/* ── Monospace font — scoped, with :not() pre-guards ─────────────────────
   :not() guards here are belt-and-braces. The definitive fix is the
   Material Icons block at the END of this stylesheet which wins via
   source order. Both defences together mean no single new rule above
   can accidentally break icon rendering.                                  */
[data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]):not([data-testid="stIconEmoji"]):not([class*="material-icons"]):not([class*="material-symbols"]),
[data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]):not([data-testid="stIconEmoji"]):not([class*="material-icons"]):not([class*="material-symbols"]),
[data-testid="stHeader"] *:not([data-testid="stIconMaterial"]):not([data-testid="stIconEmoji"]):not([class*="material-icons"]):not([class*="material-symbols"]) {
  font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}

/* ── Backgrounds ─────────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
  background: #060503 !important;
  color: #F0E6D3 !important;
}

[data-testid="stSidebar"] {
  background: #060503 !important;
  border-right: 1px solid #252018 !important;
}

/* Sidebar text — :not() guard keeps icon spans inheriting their own colour */
[data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]):not([data-testid="stIconEmoji"]):not([class*="material-icons"]):not([class*="material-symbols"]) {
  color: #F0E6D3 !important;
}

[data-testid="stSidebarNav"] { background: #060503 !important; }
[data-testid="stSidebarNav"] a { border-radius: 0 !important; }
[data-testid="stSidebarNav"] a:hover { background: #100E0A !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: #1A0F08 !important;
  border-left: 2px solid #E8621A !important;
  padding-left: 10px !important;
}

/* ── Typography ──────────────────────────────────────────────────────── */
/* span gets :not() guard — bare span catches Material Icons spans        */
p,
span:not([data-testid="stIconMaterial"]):not([data-testid="stIconEmoji"]):not([class*="material-icons"]):not([class*="material-symbols"]),
div, label, li, a {
  color: #F0E6D3 !important;
}
h1 { color: #F0E6D3 !important; font-size: 2rem !important;   letter-spacing: 0.05em; }
h2 { color: #F0E6D3 !important; font-size: 1.3rem !important; }
h3 { color: #9A9088 !important; font-size: 1rem !important;   }
h4 { color: #9A9088 !important; font-size: 0.9rem !important; letter-spacing: 0.12em; }

/* ── Metric cards — colours via CSS custom properties ────────────────── */
[data-testid="metric-container"] {
  background: #0D0A07 !important;
  border: 1px solid #252018 !important;
  border-radius: 0 !important;
  padding: 18px !important;
  position: relative !important;
  overflow: hidden !important;
}
[data-testid="metric-container"]::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--metric-neon-55), transparent);
  pointer-events: none;
}
[data-testid="stMetricValue"] {
  color: var(--metric-neon) !important;
  font-size: 1.7rem !important;
  text-shadow: 0 0 18px var(--metric-neon-44) !important;
}
[data-testid="stMetricLabel"] {
  color: #706860 !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.12em !important;
}

/* ── Misc widgets ────────────────────────────────────────────────────── */
[data-baseweb="menu"] *,
[data-baseweb="list"] *,
[role="option"] * {
  text-shadow: none !important;
}

.stPlotlyChart {
  background: #060503 !important;
  border: 1px solid #252018 !important;
  border-radius: 0 !important;
  padding: 4px !important;
}

/* Plotly updatemenu SVG overrides */
.updatemenu-header-bg,
.updatemenu-dropdown-bg,
.updatemenu-item-bg  { fill: #0D0A07 !important; stroke: #252018 !important; }
.updatemenu-header-text,
.updatemenu-item-text { fill: #F0E6D3 !important; }
.updatemenu-arrow-bg  { fill: #252018 !important; }
.updatemenu-arrow     { fill: #706860 !important; }

[data-testid="stDataFrame"] { border: 1px solid #252018 !important; }

[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
  background: #0D0A07 !important;
  border-color: #252018 !important;
  color: #F0E6D3 !important;
}

[data-baseweb="tag"] {
  background-color: #E8621A !important;
  border-color: #C9521A !important;
}
[data-baseweb="tag"] span { color: #F0E6D3 !important; }
[data-baseweb="tag"] button { color: #F0E6D3 !important; }
[data-baseweb="tag"] svg { fill: #F0E6D3 !important; }
[data-baseweb="tag"] button svg { fill: #F0E6D3 !important; }

hr { border-color: #252018 !important; }

[data-testid="stExpander"] {
  background: #0D0A07 !important;
  border: 1px solid #252018 !important;
  border-radius: 0 !important;
}
[data-testid="stAlert"] {
  background: #0D0A07 !important;
  border: 1px solid #252018 !important;
  color: #F0E6D3 !important;
  border-radius: 0 !important;
}

/* ── Hero section ────────────────────────────────────────────────────── */
.hero-wrap {
  position: relative;
  padding: 30px 0 38px;
  margin-bottom: 28px;
  border-bottom: 1px solid #252018;
  overflow: hidden;
}
.hero-wrap::before {
  content: '';
  position: absolute;
  top: -45px; right: -45px;
  width: 190px; height: 190px;
  border-radius: 50%;
  background: radial-gradient(
    circle at 38% 33%,
    rgba(255,170,112,0.88) 0%,
    rgba(255,140,66,0.65)  18%,
    rgba(232,98,26,0.42)   42%,
    rgba(232,98,26,0.10)   68%,
    transparent            100%
  );
  pointer-events: none;
}
.hero-wrap::after {
  content: '';
  position: absolute;
  top: 64px; left: 36%; right: 188px;
  height: 10px;
  border-radius: 50% 50% 50% 50% / 80% 80% 20% 20%;
  background: linear-gradient(
    to right,
    transparent           0%,
    rgba(232,98,26,0.46) 18%,
    rgba(232,98,26,0.42) 82%,
    transparent          100%
  );
  transform: rotate(-1.5deg);
  pointer-events: none;
}

.h-brush-2 {
  position: absolute;
  top: 82px; left: 40%; right: 220px;
  height: 5px;
  border-radius: 50%;
  background: linear-gradient(
    to right,
    transparent            0%,
    rgba(201,168,76,0.22) 25%,
    rgba(201,168,76,0.20) 75%,
    transparent           100%
  );
  transform: rotate(-0.5deg);
  pointer-events: none;
}

.h-brush-thick {
  position: absolute;
  top: 2px; left: 22px;
  width: 185px; height: 44px;
  border-radius: 68% 42% 62% 38% / 38% 65% 35% 62%;
  background: linear-gradient(
    106deg,
    transparent             0%,
    rgba(232,98,26,0.12)   4%,
    rgba(255,138,58,0.74) 22%,
    rgba(255,162,82,0.90) 48%,
    rgba(232,98,26,0.68)  76%,
    rgba(195,75,18,0.16)  93%,
    transparent           100%
  );
  transform: rotate(-9deg);
  pointer-events: none;
  filter: blur(0.6px);
}
.h-brush-thick::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background: linear-gradient(
    96deg,
    transparent             0%,
    rgba(255,210,140,0.18) 32%,
    rgba(255,220,160,0.28) 54%,
    transparent            100%
  );
  transform: rotate(4deg) scaleY(0.55);
}

.h-star {
  position: absolute;
  top: 34px; left: 18px;
  color: #FFE600 !important;
  font-size: 16px;
  opacity: 0.82;
  line-height: 1;
  z-index: 2;
  pointer-events: none;
  text-shadow: 0 0 8px rgba(255,230,0,0.50);
}

.h-spark-lg, .h-spark-sm, .h-dot-a, .h-dot-b,
.h-diamond,  .h-circle {
  position: absolute;
  pointer-events: none;
  line-height: 1;
  z-index: 1;
}
.h-spark-lg { right: 200px; top: 18px;    color: #E8621A !important; font-size: 22px; opacity: 0.60; }
.h-spark-sm { right: 305px; top: 10px;    color: #C9A84C !important; font-size: 13px; opacity: 0.52; }
.h-dot-a    { right: 248px; bottom: 24px; color: #E8621A !important; font-size: 15px; opacity: 0.40; }
.h-dot-b    { right: 178px; bottom: 16px; color: #F0E6D3 !important; font-size:  9px; opacity: 0.26; }
.h-diamond  { right: 335px; bottom: 22px; color: #C9A84C !important; font-size:  8px; opacity: 0.36; }
.h-circle   {
  right: 270px; top: 28px;
  width: 12px; height: 12px;
  border-radius: 50%;
  border: 1px solid #C9A84C;
  opacity: 0.30;
}

.hero-title {
  position: relative;
  z-index: 2;
  font-size: 2.8rem !important;
  letter-spacing: 0.05em !important;
  color: #F0E6D3 !important;
  margin: 0 0 14px 0 !important;
  line-height: 1.1 !important;
  font-weight: normal !important;
}
.hero-sub   { position: relative; z-index: 2; margin-top: 4px; }
.hero-badge {
  display: inline-block;
  border: 1px solid #E8621A;
  color: #E8621A !important;
  padding: 3px 12px;
  font-size: 0.6rem;
  letter-spacing: 0.22em;
  margin-right: 14px;
  vertical-align: middle;
}
.hero-meta { color: #706860 !important; font-size: 0.82rem; vertical-align: middle; }

/* ── Art cards ───────────────────────────────────────────────────────── */
.art-card {
  background: #0D0A07;
  border: 1px solid #252018;
  padding: 24px 24px 30px;
  position: relative;
  overflow: hidden;
  min-height: 240px;
  height: 100%;
}
.art-card::before {
  content: '';
  position: absolute;
  top: 12px; right: 12px;
  width: 14px; height: 14px;
  border-top:   1px solid #E8621A;
  border-right: 1px solid #E8621A;
  opacity: 0.45;
}
.art-card::after {
  content: '';
  position: absolute;
  bottom: 12px; left: 12px;
  width: 14px; height: 14px;
  border-bottom: 1px solid #E8621A;
  border-left:   1px solid #E8621A;
  opacity: 0.45;
}
.art-card-label { font-size: 0.62rem; letter-spacing: 0.22em; margin-bottom: 14px; }
.art-card-rule  { height: 1px; margin-bottom: 18px; opacity: 0.5; border: none; }
.art-card p     { margin: 9px 0 !important; font-size: 0.84rem !important; color: #B8AFA0 !important; }
.art-card p::before { content: '—'; margin-right: 10px; opacity: 0.55; }

/* ── Page header ─────────────────────────────────────────────────────── */
.page-header {
  margin-bottom: 24px;
  padding-bottom: 18px;
  border-bottom: 1px solid #252018;
  position: relative;
  overflow: hidden;
}
.page-header::after {
  content: '';
  position: absolute;
  bottom: -1px; left: 0;
  width: 60px; height: 1px;
  background: var(--insight-neon);
  opacity: 0.6;
}
.page-header::before {
  content: '';
  position: absolute;
  top: -20px; right: -20px;
  width: 90px; height: 90px;
  border-radius: 50%;
  background: radial-gradient(
    circle at 38% 38%,
    rgba(255,170,112,0.55) 0%,
    rgba(232,98,26,0.28)   40%,
    rgba(232,98,26,0.06)   70%,
    transparent            100%
  );
  pointer-events: none;
}

.ph-spark   { position: absolute; pointer-events: none; line-height: 1; }
.ph-spark-a { right: 90px;  top: 6px;  color: #E8621A !important; font-size: 14px; opacity: 0.45; }
.ph-spark-b { right: 130px; top: 14px; color: #C9A84C !important; font-size:  9px; opacity: 0.38; }
.ph-dot-c   { right: 60px;  top: 18px; color: #F0E6D3 !important; font-size:  7px; opacity: 0.20; }

.page-title {
  font-size: 1.9rem !important;
  letter-spacing: 0.04em !important;
  color: #F0E6D3 !important;
  margin: 0 0 8px 0 !important;
  font-weight: normal !important;
}
.page-subtitle {
  font-size: 0.6rem !important;
  letter-spacing: 0.2em !important;
  color: #706860 !important;
}

/* ── Insight box — border colour via CSS custom property ─────────────── */
.insight-box {
  background: #0D0A07;
  border-left: 2px solid var(--insight-neon);
  padding: 14px 20px 16px;
  margin: 24px 0 16px;
  font-size: 0.83rem;
  color: #9A9088;
}
.insight-label {
  display: block;
  font-size: 0.5rem;
  letter-spacing: 0.22em;
  color: var(--insight-neon) !important;
  margin-bottom: 10px;
}
.insight-box strong { color: #F0E6D3 !important; }

/* ── Decorative footer ───────────────────────────────────────────────── */
.deco-footer {
  border-top: 1px solid #252018;
  padding-top: 14px;
  margin-top: 4px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.deco-footer-text { color: #4A4038 !important; font-size: 0.74rem; }
.deco-footer-sep  { flex: 1; height: 1px; background: linear-gradient(90deg, #252018, transparent); }

/* ── Material Icons / Symbols ────────────────────────────────────────────
   Why last: every rule above that targets * or span carries !important.
   When specificity is equal, CSS source order decides — the last rule
   wins. Placing this block at the very end guarantees it overrides the
   monospace font-family assigned by the scoped rules above, regardless
   of what gets added to the theme in future.

   Selectors covered:
   • All Material Icons / Symbols class variants
   • button[data-testid="stSidebarCollapseButton"] span
       → the keyboard_double_arrow_left/right sidebar chevron
   • Expander toggle icon elements
   • Attribute wildcard [class*="material-"] for any variant Streamlit
     may inject that doesn't have an explicit class listed above         */
/* Streamlit's DynamicIcon component renders material icons as a plain <span>
   with data-testid="stIconMaterial" — no Material Icons class at all.
   This is the REAL selector for the sidebar chevron icons.
   All class-based selectors below are kept for coverage of any other
   Material Icons usage, but stIconMaterial is the critical one.      */
[data-testid="stIconMaterial"],
[data-testid="stIconEmoji"],
.material-icons,
.material-icons-outlined,
.material-icons-round,
.material-icons-sharp,
.material-icons-two-tone,
.material-symbols-outlined,
.material-symbols-rounded,
.material-symbols-sharp,
button[data-testid="stSidebarCollapseButton"] span,
button[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
button[data-testid="stExpandSidebarButton"] span,
button[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
button[data-testid="stBaseButton-headerNoPadding"] span,
button[data-testid="stBaseButton-header"] span,
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderToggleIcon"] span,
[data-testid="stExpanderToggleIcon"] > *,
details summary [data-testid="stExpanderToggleIcon"] span,
[class*="material-icons"],
[class*="material-symbols"] {
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
               'Material Icons Round', 'Material Icons' !important;
  font-style: normal !important;
  font-weight: normal !important;
  line-height: 1 !important;
  text-rendering: optimizeLegibility !important;
  -webkit-font-feature-settings: 'liga' 1 !important;
  font-feature-settings: 'liga' 1 !important;
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
}
"""


def inject_css(metric_neon: str = "#FF6E1F", insight_neon: str = "#FF6E1F") -> None:
    """
    Inject the warm-dark art-company theme into the current Streamlit page.

    Three separate st.markdown() calls — intentional, not collapsible:

    Call 1  <link> tags only.
            Browser starts preconnect to Google Fonts immediately.
            Must arrive before any <style> block so the font requests
            are in-flight while Streamlit renders the rest of the page.

    Call 2  :root { } custom properties only, f-string interpolated.
            The only place metric_neon / insight_neon touch the CSS.
            Streamlit accumulates <style> tags across page navigations,
            so competing colour rules would build up if we hard-coded
            colours in Call 3. :root overrides always resolve to the
            most-recently-injected value — correct per-page, no leakage.

    Call 3  _STATIC_CSS — plain string, zero interpolation.
            Static and permanent. Injected once effectively; even if
            Streamlit injects it again on re-render, identical rules
            have no effect. All per-page colour references use var().
    """
    import streamlit as _st

    # Call 1 — font preconnect + stylesheet links
    _st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap">
<link rel="stylesheet"
      href="https://fonts.googleapis.com/icon?family=Material+Icons|Material+Icons+Outlined|Material+Icons+Round">
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200">
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200">
""",
        unsafe_allow_html=True,
    )

    # Call 2 — per-page CSS custom properties (f-string, colours only)
    _st.markdown(
        f"""
<style>
:root {{
  --metric-neon:    {metric_neon};
  --metric-neon-55: {metric_neon}55;
  --metric-neon-44: {metric_neon}44;
  --insight-neon:   {insight_neon};
}}
</style>
""",
        unsafe_allow_html=True,
    )

    # Call 3 — static theme (plain string, no interpolation)
    _st.markdown(f"<style>{_STATIC_CSS}</style>", unsafe_allow_html=True)
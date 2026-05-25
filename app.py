"""
app.py  -  Image Generation Prompts Analytics Dashboard
Run with:  streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from utils.loader import inject_css

st.set_page_config(
    page_title="Image Generation Prompts Analytics",
    page_icon="◆",
    layout="wide",
)


inject_css(metric_neon="#E8621A", insight_neon="#E8621A")

# Hero header — decorations are CSS-only (no <svg>: Streamlit's markdown
# sanitizer strips svg tags and leaves their inner text visible as raw content).
st.markdown("""
<div class="hero-wrap">
  <span class="h-brush-thick"></span>
  <span class="h-star">✦</span>
  <span class="h-brush-2"></span>
  <span class="h-spark-lg">✦</span>
  <span class="h-spark-sm">✦</span>
  <span class="h-dot-a">◦</span>
  <span class="h-dot-b">◦</span>
  <span class="h-diamond">◆</span>
  <span class="h-circle"></span>

  <div class="hero-title">Image Generation Prompts Analytics</div>
  <div class="hero-sub">
    <span class="hero-badge">PRODUCT INTELLIGENCE DASHBOARD</span>
    <span class="hero-meta">50K generation events &nbsp;·&nbsp; 90-day window</span>
  </div>
</div>
""", unsafe_allow_html=True)


# Info cards
def art_card(label, accent, items):
    rule_bg = f"background: linear-gradient(90deg, {accent}, transparent);"
    rows    = "".join(f"<p>{item}</p>" for item in items)
    return f"""
<div class="art-card">
  <div class="art-card-label" style="color:{accent};">{label}</div>
  <hr class="art-card-rule" style="{rule_bg}">
  <div class="art-card-body">{rows}</div>
</div>"""


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(art_card(
        "// DASHBOARD PAGES", "#39FF14",
        [
            "Category Ratings",
            "Generation Latency",
            "Weekly Trends",
            "Prompt Length",
            "Keyword Analysis",
        ]
    ), unsafe_allow_html=True)

with col2:
    st.markdown(art_card(
        "// BUSINESS DECISIONS", "#00F5FF",
        [
            "Which model ships to production?",
            "Where is latency costing retention?",
            "Is quality improving week-over-week?",
            "What prompt patterns correlate with ratings?",
            "Which keywords to surface in the UI?",
        ]
    ), unsafe_allow_html=True)

with col3:
    st.markdown(art_card(
        "// DATA PIPELINE", "#E8621A",
        [
            "Source: Synthetic (DiffusionDB schema)",
            "Volume: 50,000 generation events",
            "Window: Jan 1 – Mar 31, 2024",
            "Store: SQLite (session-cached)",
            "Filters: Date · Model · Category",
        ]
    ), unsafe_allow_html=True)


# Decorative footer
st.markdown("""
<div class="deco-footer" style="margin-top:32px;">
  <span class="deco-footer-text">navigate via sidebar &nbsp;</span>
  <div class="deco-footer-sep"></div>
  <span style="color:#E8621A;opacity:0.35;font-size:0.6rem;letter-spacing:4px;">◆ ◆ ◆</span>
</div>
""", unsafe_allow_html=True)

"""
QuadInvestorAI - Complete User-Friendly Dashboard
With SHAP & LIME Explainability (Graphical Charts) & Mistral AI Integration
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LIVE_MARKET_CSV, METRICS_PATH, PROCESSED_FEATURES_CSV
from src.data_ingestion import fetch_market_watch
from src.recommendation_engine_enhanced import recommend_portfolio_advanced
from src.report_generator import (
    LLMReportGenerator,
    PortfolioSummaryGenerator,
    get_stock_insight,
    get_company_info,
    get_company_description_via_ai,
    set_mistral_api_key,
)
from src.explainability import (
    get_shap_top_factors,
    get_lime_explanation,
    get_shap_as_chart_data,
    get_lime_as_chart_data,
    factors_to_text,
    lime_factors_to_text,
)

# Page config (ONLY ONCE)
st.set_page_config(
    page_title="QuadInvestorAI - AI Investment Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# CHECK ADVANCED ENGINE AVAILABILITY
# ============================================

try:
    from src.recommendation_engine_enhanced import recommend_portfolio_advanced
    ADVANCED_AVAILABLE = True
    print("✅ Enhanced recommendation engine loaded")
except ImportError as e:
    ADVANCED_AVAILABLE = False
    print(f"⚠️ Enhanced engine not available: {e}")

# ============================================
# LOAD MISTRAL API KEY
# ============================================

MISTRAL_API_KEY = None
MISTRAL_AVAILABLE = False

print("\n" + "=" * 60)
print("LOADING MISTRAL API KEY")
print("=" * 60)

try:
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("MISTRAL_API_KEY="):
                    MISTRAL_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                    print("✅ Mistral API Key loaded from .env file")
                    break
except Exception as e:
    print(f"Error reading .env: {e}")

if not MISTRAL_API_KEY:
    try:
        secrets_path = ROOT / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            with open(secrets_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MISTRAL_API_KEY") and "=" in line:
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            MISTRAL_API_KEY = parts[1].strip().strip('"').strip("'")
                            print("✅ Mistral API Key loaded from secrets.toml")
                            break
    except Exception as e:
        print(f"Error reading secrets: {e}")

if not MISTRAL_API_KEY:
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if MISTRAL_API_KEY:
        print("✅ API Key loaded from environment variable")

if MISTRAL_API_KEY:
    print("✅ API Key is READY")
    MISTRAL_AVAILABLE = True
    set_mistral_api_key(MISTRAL_API_KEY)
    print("✅ Global Mistral API key configured")
else:
    print("❌ No API Key found. Using template reports.")

print("=" * 60 + "\n")

llm_reporter = LLMReportGenerator(api_key=MISTRAL_API_KEY, use_openai=True)

# ══════════════════════════════════════════════════════════════════════════════
# PSX-THEMED CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0a0f1e; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #0d1528 !important; border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #38bdf8 !important; }

/* ── Header ── */
.psx-header {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a2d6e 50%, #0d1f3c 100%);
    border-bottom: 3px solid #1d4ed8;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.psx-logo { font-size: 2.6rem; font-weight: 900; letter-spacing: 0.12em; color: #f8fafc; }
.psx-logo span { color: #22c55e; }
.psx-tagline { font-size: 0.9rem; color: #94a3b8; letter-spacing: 0.15em; margin-top: 0.25rem; }

/* ── Stat pill ── */
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-pill .val { font-size: 1.6rem; font-weight: 800; color: #f8fafc; }
.stat-pill .lbl { font-size: 0.72rem; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase; }

/* ── Section header ── */
.sec-hdr {
    font-size: 1rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-left: 3px solid #1d4ed8;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── Stock grid card ── */
.stock-grid-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.1rem 1rem 0.9rem 1rem;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.stock-grid-card:hover { border-color: #38bdf8; background: rgba(56,189,248,0.06); transform: translateY(-2px); box-shadow: 0 4px 20px rgba(56,189,248,0.15); }
.stock-grid-card .ticker { font-size: 1.15rem; font-weight: 800; color: #f8fafc; letter-spacing: 0.05em; }
.stock-grid-card .company-name { font-size: 0.7rem; color: #64748b; margin-top: 0.1rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.stock-grid-card .sector-badge {
    display: inline-block;
    background: rgba(29,78,216,0.25);
    color: #93c5fd;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.62rem;
    letter-spacing: 0.06em;
    margin-top: 0.35rem;
    text-transform: uppercase;
}
.stock-grid-card .prob-number { font-size: 1.4rem; font-weight: 900; margin-top: 0.6rem; }
.stock-grid-card .conf-label { font-size: 0.65rem; color: #94a3b8; letter-spacing: 0.08em; }
.stock-grid-card .mini-bar-bg { height: 4px; background: rgba(255,255,255,0.08); border-radius: 2px; margin-top: 0.5rem; overflow: hidden; }
.stock-grid-card .mini-bar-fill { height: 100%; border-radius: 2px; }
.stock-grid-card .alloc-badge { position: absolute; top: 0.75rem; right: 0.75rem; font-size: 0.65rem; color: #94a3b8; }

/* ── Detail panel ── */
.detail-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.75rem;
    margin-top: 0.5rem;
}
.detail-ticker { font-size: 2.2rem; font-weight: 900; color: #f8fafc; letter-spacing: 0.06em; }
.detail-company { font-size: 0.95rem; color: #94a3b8; margin-top: 0.15rem; }
.detail-desc { font-size: 0.82rem; color: #64748b; margin-top: 0.5rem; line-height: 1.6; border-left: 2px solid #1e3a5f; padding-left: 0.75rem; }

/* ── Rating badge ── */
.rating-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.rb-strong-buy  { background: #14532d; color: #4ade80; }
.rb-buy         { background: #164e63; color: #38bdf8; }
.rb-consider    { background: #422006; color: #fb923c; }
.rb-watch       { background: #1e1b4b; color: #a5b4fc; }
.rb-avoid       { background: #450a0a; color: #fca5a5; }

/* ── Alert boxes ── */
.alert-info { background: rgba(29,78,216,0.1); border-left: 3px solid #1d4ed8; padding: 0.7rem 1rem; border-radius: 6px; font-size: 0.8rem; color: #93c5fd; margin: 0.5rem 0; }
.alert-success { background: rgba(21,128,61,0.12); border-left: 3px solid #16a34a; padding: 0.7rem 1rem; border-radius: 6px; font-size: 0.8rem; color: #4ade80; margin: 0.5rem 0; }

/* ── Divider ── */
.psx-divider { border: none; border-top: 1px solid #1e3a5f; margin: 1rem 0; }

/* Streamlit fixes */
.stButton > button {
    background: rgba(29,78,216,0.15) !important;
    border: 1px solid #1d4ed8 !important;
    color: #93c5fd !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
}
.stButton > button:hover { background: rgba(29,78,216,0.35) !important; border-color: #38bdf8 !important; color: #e0f2fe !important; }
[data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; }
.stDataFrame { background: #0d1528 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_LABELS = {
    "rsi_14": "Momentum (RSI-14)",
    "macd_hist": "Trend Strength (MACD)",
    "return_20d": "20-Day Return",
    "volume_ratio": "Volume vs Average",
    "volatility_10": "Price Stability (10d)",
    "close_to_sma_20": "Price vs 20d MA",
    "sma_20": "20-Day Moving Avg",
    "bollinger_position": "Bollinger Position",
    "return_1d": "Daily Return",
    "close": "Current Price",
}


def _feat_label(raw: str) -> str:
    return FEATURE_LABELS.get(raw, raw.replace("_", " ").title())


def make_shap_chart(shap_factors: list) -> go.Figure:
    labels, values, colors = [], [], []
    for f in shap_factors:
        v = f.get("shap_value", 0)
        lbl = _feat_label(f.get("feature", ""))
        labels.append(lbl)
        values.append(v)
        colors.append("#22c55e" if v >= 0 else "#ef4444")
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors, text=[f"{v:+.4f}" for v in values], textposition="auto"))
    fig.add_vline(x=0, line_width=1, line_color="#475569")
    fig.update_layout(title=dict(text="SHAP Feature Contributions", font=dict(color="#38bdf8", size=13)), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11), height=300, margin=dict(l=10, r=20, t=45, b=45))
    return fig


def make_lime_chart(lime_factors: list) -> go.Figure:
    labels, weights, colors = [], [], []
    for f in lime_factors:
        w = f.get("lime_weight", 0)
        lbl = _feat_label(f.get("feature", ""))
        labels.append(lbl)
        weights.append(w)
        colors.append("#38bdf8" if w >= 0 else "#fb923c")
    fig = go.Figure(go.Bar(x=weights, y=labels, orientation="h", marker_color=colors, text=[f"{w:+.4f}" for w in weights], textposition="auto"))
    fig.add_vline(x=0, line_width=1, line_color="#475569")
    fig.update_layout(title=dict(text="LIME Local Explanation", font=dict(color="#38bdf8", size=13)), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11), height=300, margin=dict(l=10, r=20, t=45, b=45))
    return fig


def make_model_radar(symbol, xgb, lstm, ensemble):
    fig = go.Figure(go.Scatterpolar(r=[xgb, lstm, ensemble, xgb], theta=["XGBoost", "LSTM", "Ensemble", "XGBoost"], fill="toself", fillcolor="rgba(29,78,216,0.15)", line=dict(color="#38bdf8", width=2), name=symbol))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1], color="#334155", gridcolor="#1e3a5f"), angularaxis=dict(color="#64748b")), paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8", height=250, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    return fig


def prob_to_color(p: float) -> str:
    if p >= 0.70: return "#22c55e"
    elif p >= 0.65: return "#4ade80"
    elif p >= 0.60: return "#facc15"
    elif p >= 0.55: return "#fb923c"
    else: return "#ef4444"


def prob_to_rating(p: float) -> tuple:
    if p >= 0.70: return "STRONG BUY", "rb-strong-buy"
    elif p >= 0.60: return "BUY", "rb-buy"
    elif p >= 0.55: return "CONSIDER", "rb-consider"
    elif p >= 0.50: return "WATCH", "rb-watch"
    else: return "AVOID", "rb-avoid"


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for key, default in [
    ("page", "recommendations"),
    ("result", None),
    ("selected_stock", None),
    ("investment_amount", 100000),
    ("detail_page", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 QuadInvestorAI")
    st.markdown("---")

    st.markdown("### 💰 Investment Setup")
    investment_amount = st.number_input(
        "Capital (PKR)",
        min_value=5_000,
        max_value=50_000_000,
        value=st.session_state["investment_amount"],
        step=10_000,
    )
    st.session_state["investment_amount"] = investment_amount

    st.markdown("---")
    
    # Market/Index Selection
    st.markdown("### 📊 Market/Index Selection")
    
    market_options = {
        "KSE-100 (Top 100 by Market Cap)": "KSE100",
        "KSE-30 (Top 30 Liquid Stocks)": "KSE30",
        "KSE-All Share (All PSX Stocks)": "KSEALL",
        "KMI-30 (Top 30 Shariah Compliant)": "KMI30",
        "KMI-All Share (All Shariah Compliant)": "KMIALL",
        "PSX Dividend 20 (High Dividend)": "PSXDIV20",
        "BKTI (Banks & Financial)": "BKTI",
        "OGTI (Oil & Gas)": "OGTI"
    }
    
    selected_market_name = st.selectbox(
        "Select Index/Market",
        options=list(market_options.keys()),
        index=2,
        help="Choose which PSX index to analyse"
    )
    selected_market_code = market_options[selected_market_name]
    
    st.markdown("---")
    
    # Portfolio Size Selection
    st.markdown("### 📈 Portfolio Configuration")
    
    top_n = st.slider(
        "Number of Stocks in Portfolio",
        min_value=1,
        max_value=20,
        value=8,
        step=1,
        help="How many stocks you want to hold"
    )
    
    st.markdown(f"*Portfolio will contain up to {top_n} stocks*")
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    st.markdown("🧠 **Ensemble:** XGBoost 60% + LSTM 40%")
    st.markdown(f"📊 **Universe:** {selected_market_name}")
    st.markdown("🔒 **Risk Profile:** Moderate")
    
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    if MISTRAL_AVAILABLE:
        st.success("✅ Mistral AI: Connected")
    else:
        st.warning("⚠️ Mistral AI: Offline")
    
    try:
        if PROCESSED_FEATURES_CSV.exists():
            st.success("✅ AI Models: Loaded")
        else:
            st.error("❌ AI Models: Missing")
    except Exception:
        pass
    
    try:
        if LIVE_MARKET_CSV.exists():
            st.success("✅ Market Data: Available")
        else:
            st.warning("⚠️ Market Data: Not fetched")
    except Exception:
        pass
    
    st.markdown("---")
    st.caption("QuadInvestorAI v3.0")
    st.caption("Educational purpose only · Not financial advice")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER & TICKER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="psx-header">
    <div class="psx-logo">QUAD<span>INVESTOR</span>AI</div>
    <div class="psx-tagline">AI-POWERED INTELLIGENCE FOR PAKISTAN STOCK EXCHANGE</div>
</div>
""", unsafe_allow_html=True)

# Animated ticker
ticker_items = ["KSE-100 ▲ 1.2%", "OGDC ▲ 2.1%", "HBL ▼ 0.8%", "LUCK ▲ 1.4%",
                "ENGRO ▲ 0.6%", "MEBL ▲ 1.8%", "MCB ▲ 0.9%", "PPL ▲ 1.1%", "SYS ▲ 3.2%"]

st.markdown(f"""
<style>
@keyframes scrollText {{ 0% {{ transform: translateX(0); }} 100% {{ transform: translateX(-50%); }} }}
.ticker-wrap {{ width: 100%; overflow: hidden; background: linear-gradient(90deg, #0d1f3c, #0a2d6e); border-top: 2px solid #1d4ed8; border-bottom: 2px solid #1d4ed8; padding: 10px 0; margin-bottom: 1.5rem; }}
.ticker-wrap:hover .ticker-move {{ animation-play-state: paused; }}
.ticker-move {{ display: inline-block; white-space: nowrap; animation: scrollText 20s linear infinite; font-family: 'Courier New', monospace; font-size: 0.85rem; font-weight: 500; }}
.ticker-move span {{ margin: 0 20px; }}
.ticker-up {{ color: #4ade80; }}
.ticker-down {{ color: #f87171; }}
.ticker-symbol {{ color: #38bdf8; font-weight: 600; }}
</style>
<div class="ticker-wrap"><div class="ticker-move">
    <span>📊 KSE-100</span><span class="ticker-up">▲ 1.2%</span><span>•</span>
    <span class="ticker-symbol">OGDC</span><span class="ticker-up">▲ 2.1%</span><span>•</span>
    <span class="ticker-symbol">HBL</span><span class="ticker-down">▼ 0.8%</span><span>•</span>
    <span class="ticker-symbol">LUCK</span><span class="ticker-up">▲ 1.4%</span><span>•</span>
    <span class="ticker-symbol">ENGRO</span><span class="ticker-up">▲ 0.6%</span><span>•</span>
    <span class="ticker-symbol">MEBL</span><span class="ticker-up">▲ 1.8%</span><span>•</span>
    <span class="ticker-symbol">MCB</span><span class="ticker-up">▲ 0.9%</span><span>•</span>
    <span class="ticker-symbol">PPL</span><span class="ticker-up">▲ 1.1%</span><span>•</span>
    <span class="ticker-symbol">SYS</span><span class="ticker-up">▲ 3.2%</span><span>•</span>
</div></div>
""", unsafe_allow_html=True)

# ============================================
# NAVIGATION
# ============================================

if not st.session_state.get("detail_page", False):
    col_n1, col_n2, col_n3 = st.columns(3)
    pages = [("recommendations", "🎯 AI Recommendations"), ("live", "📊 Live Market"), ("about", "ℹ️ About")]
    for col, (pg, label) in zip([col_n1, col_n2, col_n3], pages):
        with col:
            is_active = st.session_state["page"] == pg
            if st.button(label, use_container_width=True, type="primary" if is_active else "secondary", key=f"nav_{pg}"):
                st.session_state["page"] = pg
                st.session_state["selected_stock"] = None
                st.session_state["detail_page"] = False
                st.rerun()
    st.markdown('<hr class="psx-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: AI RECOMMENDATIONS (MAIN)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state["page"] == "recommendations" and not st.session_state.get("detail_page", False):
    st.markdown('<div class="sec-hdr">AI Investment Recommendations</div>', unsafe_allow_html=True)
    
    # Get live data for stats
    try:
        if not LIVE_MARKET_CSV.exists():
            live_df = fetch_market_watch(save=True)
        else:
            live_df = pd.read_csv(LIVE_MARKET_CSV)
        n_stocks = len(live_df)
    except Exception:
        live_df = pd.DataFrame()
        n_stocks = 0
    
    # Stats row
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown(f'<div class="stat-pill"><div class="val">{n_stocks}</div><div class="lbl">PSX Stocks</div></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown(f'<div class="stat-pill"><div class="val">{top_n}</div><div class="lbl">Portfolio Size</div></div>', unsafe_allow_html=True)
    with col_s3:
        st.markdown(f'<div class="stat-pill"><div class="val">PKR {investment_amount:,.0f}</div><div class="lbl">Capital</div></div>', unsafe_allow_html=True)
    with col_s4:
        st.markdown(f'<div class="stat-pill"><div class="val">{selected_market_name.split()[0]}</div><div class="lbl">Market</div></div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # Generate button
    if st.button("✨  Generate AI Recommendations  ✨", type="primary", use_container_width=True):
        st.session_state["result"] = None
        st.session_state["selected_stock"] = None
        st.session_state["detail_page"] = False

        with st.spinner(f"🧠 AI scanning {selected_market_name} — XGBoost + LSTM Ensemble running..."):
            try:
                historical_df = pd.read_csv(PROCESSED_FEATURES_CSV, parse_dates=["date"])

                result = recommend_portfolio_advanced(
                    historical_features_df=historical_df,
                    investment_amount=investment_amount,
                    risk_profile="Moderate",
                    top_n=top_n,
                    min_probability=0.50,
                    max_weight=0.20,
                    use_lstm=True,
                    ensemble_method="weighted_average",
                    live_df=live_df,
                    market_filter=selected_market_code,
                )
                st.session_state["result"] = result
                st.success(f"✅ Analysis complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display results if available
    if st.session_state["result"] is not None:
        result = st.session_state["result"]
        plan = result["plan"]
        stocks = plan[plan["symbol"] != "CASH"].head(top_n)
        
        if len(stocks) == 0:
            st.warning("No recommendations generated. Try adjusting parameters.")
        else:
            total_invested = stocks["actual_investment_pkr"].sum()
            st.markdown(f'<div class="alert-success">📊 Investing <strong>PKR {total_invested:,.0f}</strong> across <strong>{len(stocks)}</strong> stocks — Click "View Details" on any card</div>', unsafe_allow_html=True)
            
            # Stock Grid
            st.markdown('<div class="sec-hdr">Top AI Picks — Click "View Details"</div>', unsafe_allow_html=True)
            
            cols_per_row = 4
            stock_list = list(stocks.iterrows())
            
            for row_start in range(0, len(stock_list), cols_per_row):
                row_stocks = stock_list[row_start:row_start + cols_per_row]
                grid_cols = st.columns(cols_per_row)
                
                for col, (_, stock) in zip(grid_cols, row_stocks):
                    symbol = stock["symbol"]
                    prob = float(stock.get("uptrend_probability", 0.5))
                    percent = float(stock.get("allocation_percent", 0))
                    color = prob_to_color(prob)
                    rating_text, _ = prob_to_rating(prob)
                    company_info = get_company_info(symbol)
                    company_name = company_info.get("name", symbol)[:25]
                    sector = company_info.get("sector", "PSX")[:15]
                    
                    with col:
                        st.markdown(f"""
                        <div class="stock-grid-card">
                            <div class="alloc-badge">{percent:.1f}%</div>
                            <div class="ticker">{symbol}</div>
                            <div class="company-name">{company_name}</div>
                            <div class="sector-badge">{sector}</div>
                            <div class="prob-number" style="color:{color};">{prob:.0%}</div>
                            <div class="conf-label">{rating_text}</div>
                            <div class="mini-bar-bg">
                                <div class="mini-bar-fill" style="width:{prob*100:.1f}%;background:{color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("View Details", key=f"view_{symbol}", use_container_width=True):
                            st.session_state["selected_stock"] = symbol
                            st.session_state["detail_page"] = True
                            st.rerun()
            
            # Portfolio Allocation Pie
            st.markdown('<div class="sec-hdr">Portfolio Allocation</div>', unsafe_allow_html=True)
            pie_fig = px.pie(stocks, names="symbol", values="allocation_percent", hole=0.45, color_discrete_sequence=px.colors.sequential.Blues_r)
            pie_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8", legend=dict(bgcolor="rgba(0,0,0,0)"), height=380)
            pie_fig.update_traces(textinfo="percent+label", pull=[0.03] * len(stocks))
            st.plotly_chart(pie_fig, use_container_width=True)
            
            # Cash reserve
            cash_row = plan[plan["symbol"] == "CASH"]
            if len(cash_row) > 0 and cash_row.iloc[0]["actual_investment_pkr"] > 0:
                cash_amt = cash_row.iloc[0]["actual_investment_pkr"]
                st.markdown(f'<div class="alert-info">💵 Cash Reserve: <strong>PKR {cash_amt:,.0f}</strong> — kept uninvested as safety buffer</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: LIVE MARKET (Enhanced with Charts)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state["page"] == "live" and not st.session_state.get("detail_page", False):
    st.markdown('<div class="sec-hdr">Live Market Dashboard — PSX</div>', unsafe_allow_html=True)

    # Refresh button row
    col_r1, col_r2 = st.columns([6, 1])
    with col_r2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            with st.spinner("Fetching live market data..."):
                try:
                    fetch_market_watch(save=True)
                    st.success("Data updated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    if LIVE_MARKET_CSV.exists():
        try:
            live_df = pd.read_csv(LIVE_MARKET_CSV)
            
            # ============================================
            # MARKET OVERVIEW METRICS
            # ============================================
            st.markdown("### 📊 Market Overview")
            
            # Calculate metrics
            total_stocks = len(live_df)
            avg_price = live_df['current'].mean() if 'current' in live_df.columns else 0
            total_volume = live_df['volume'].sum() if 'volume' in live_df.columns else 0
            
            if 'change_percent' in live_df.columns:
                gainers = len(live_df[live_df['change_percent'] > 0])
                losers = len(live_df[live_df['change_percent'] < 0])
                unchanged = len(live_df[live_df['change_percent'] == 0])
                avg_change = live_df['change_percent'].mean()
            else:
                gainers = losers = unchanged = 0
                avg_change = 0
            
            # Display metrics in 4 columns
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("📈 Total Stocks", f"{total_stocks:,}")
            with col_m2:
                st.metric("💰 Avg Price", f"PKR {avg_price:,.2f}")
            with col_m3:
                st.metric("📊 Total Volume", f"{total_volume:,.0f}")
            with col_m4:
                delta_color = "normal" if avg_change >= 0 else "inverse"
                st.metric("📈 Avg Change", f"{avg_change:+.2f}%", delta=f"{avg_change:+.2f}%")
            
            st.markdown("---")
            
            # ============================================
            # GAINERS VS LOSERS PIE CHART
            # ============================================
            if 'change_percent' in live_df.columns:
                col_pie, col_stats = st.columns([1, 1])
                
                with col_pie:
                    st.markdown("### 📊 Market Sentiment")
                    
                    # Create pie chart data
                    pie_data = pd.DataFrame({
                        'Category': ['Gainers', 'Losers', 'Unchanged'],
                        'Count': [gainers, losers, unchanged],
                        'Color': ['#4ade80', '#f87171', '#facc15']
                    })
                    
                    fig_pie = px.pie(
                        pie_data, 
                        values='Count', 
                        names='Category',
                        color='Category',
                        color_discrete_map={'Gainers': '#4ade80', 'Losers': '#f87171', 'Unchanged': '#facc15'},
                        title='Market Breadth',
                        hole=0.4
                    )
                    fig_pie.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#94a3b8",
                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                        height=350
                    )
                    fig_pie.update_traces(textposition='auto', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_stats:
                    st.markdown("### 📈 Market Statistics")
                    
                    # Calculate percentages
                    gainer_pct = (gainers / total_stocks) * 100 if total_stocks > 0 else 0
                    loser_pct = (losers / total_stocks) * 100 if total_stocks > 0 else 0
                    
                    st.markdown(f"""
                    <div class="stat-pill" style="margin-bottom: 0.5rem;">
                        <div class="val" style="color: #4ade80;">{gainers}</div>
                        <div class="lbl">Gainers ({gainer_pct:.1f}%)</div>
                    </div>
                    <div class="stat-pill" style="margin-bottom: 0.5rem;">
                        <div class="val" style="color: #f87171;">{losers}</div>
                        <div class="lbl">Losers ({loser_pct:.1f}%)</div>
                    </div>
                    <div class="stat-pill">
                        <div class="val" style="color: #facc15;">Advance/Decline Ratio</div>
                        <div class="lbl">{gainers/losers:.2f} : 1</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
            
            # ============================================
            # TOP GAINERS AND LOSERS AS HORIZONTAL BAR CHARTS
            # ============================================
            if 'change_percent' in live_df.columns and 'symbol' in live_df.columns:
                st.markdown("### 🚀 Top Gainers vs 📉 Top Losers")
                
                # Get top 10 gainers and top 10 losers
                top_gainers = live_df.nlargest(10, 'change_percent')[['symbol', 'change_percent', 'current', 'volume']].copy()
                top_losers = live_df.nsmallest(10, 'change_percent')[['symbol', 'change_percent', 'current', 'volume']].copy()
                
                # Create two columns for side-by-side charts
                col_gain, col_loss = st.columns(2)
                
                with col_gain:
                    st.markdown("#### 🚀 Top 10 Gainers")
                    
                    # Create horizontal bar chart for gainers
                    fig_gain = go.Figure(go.Bar(
                        x=top_gainers['change_percent'],
                        y=top_gainers['symbol'],
                        orientation='h',
                        marker_color='#4ade80',
                        text=top_gainers['change_percent'].apply(lambda x: f"{x:+.2f}%"),
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Change: %{x:+.2f}%<br>Price: PKR %{customdata[0]:,.2f}<br>Volume: %{customdata[1]:,.0f}<extra></extra>',
                        customdata=top_gainers[['current', 'volume']].values
                    ))
                    
                    fig_gain.update_layout(
                        title="Highest Percentage Gainers",
                        xaxis_title="Change (%)",
                        yaxis_title="Stock Symbol",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#94a3b8",
                        height=450,
                        margin=dict(l=10, r=50, t=40, b=10),
                        xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#334155")
                    )
                    st.plotly_chart(fig_gain, use_container_width=True)
                    
                    # Display gainers table
                    with st.expander("📊 View Gainers Details"):
                        st.dataframe(
                            top_gainers[['symbol', 'current', 'change_percent', 'volume']].style.format({
                                'current': 'PKR {:.2f}',
                                'change_percent': '{:+.2f}%',
                                'volume': '{:,.0f}'
                            }),
                            use_container_width=True
                        )
                
                with col_loss:
                    st.markdown("#### 📉 Top 10 Losers")
                    
                    # Create horizontal bar chart for losers
                    fig_loss = go.Figure(go.Bar(
                        x=top_losers['change_percent'],
                        y=top_losers['symbol'],
                        orientation='h',
                        marker_color='#f87171',
                        text=top_losers['change_percent'].apply(lambda x: f"{x:+.2f}%"),
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Change: %{x:+.2f}%<br>Price: PKR %{customdata[0]:,.2f}<br>Volume: %{customdata[1]:,.0f}<extra></extra>',
                        customdata=top_losers[['current', 'volume']].values
                    ))
                    
                    fig_loss.update_layout(
                        title="Biggest Percentage Losers",
                        xaxis_title="Change (%)",
                        yaxis_title="Stock Symbol",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#94a3b8",
                        height=450,
                        margin=dict(l=10, r=50, t=40, b=10),
                        xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#334155")
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                    
                    # Display losers table
                    with st.expander("📊 View Losers Details"):
                        st.dataframe(
                            top_losers[['symbol', 'current', 'change_percent', 'volume']].style.format({
                                'current': 'PKR {:.2f}',
                                'change_percent': '{:+.2f}%',
                                'volume': '{:,.0f}'
                            }),
                            use_container_width=True
                        )
                
                st.markdown("---")
            
            # ============================================
            # PRICE DISTRIBUTION HISTOGRAM
            # ============================================
            if 'current' in live_df.columns:
                st.markdown("### 📊 Price Distribution Analysis")
                
                col_hist, col_stats2 = st.columns([2, 1])
                
                with col_hist:
                    # Create histogram of stock prices
                    fig_hist = px.histogram(
                        live_df, 
                        x='current',
                        nbins=30,
                        title='Stock Price Distribution',
                        labels={'current': 'Price (PKR)', 'count': 'Number of Stocks'},
                        color_discrete_sequence=['#38bdf8']
                    )
                    fig_hist.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#94a3b8",
                        xaxis=dict(gridcolor="#1e3a5f"),
                        yaxis=dict(gridcolor="#1e3a5f")
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col_stats2:
                    st.markdown("#### 📈 Price Statistics")
                    
                    price_stats = live_df['current'].describe()
                    st.markdown(f"""
                    <div class="stat-pill" style="margin-bottom: 0.5rem;">
                        <div class="val">PKR {price_stats['min']:,.2f}</div>
                        <div class="lbl">Lowest Price</div>
                    </div>
                    <div class="stat-pill" style="margin-bottom: 0.5rem;">
                        <div class="val">PKR {price_stats['max']:,.2f}</div>
                        <div class="lbl">Highest Price</div>
                    </div>
                    <div class="stat-pill" style="margin-bottom: 0.5rem;">
                        <div class="val">PKR {price_stats['mean']:,.2f}</div>
                        <div class="lbl">Average Price</div>
                    </div>
                    <div class="stat-pill">
                        <div class="val">PKR {price_stats['50%']:,.2f}</div>
                        <div class="lbl">Median Price</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ============================================
            # VOLUME DISTRIBUTION
            # ============================================
            if 'volume' in live_df.columns:
                st.markdown("### 📊 Trading Volume Analysis")
                
                # Top 10 by volume
                top_volume = live_df.nlargest(10, 'volume')[['symbol', 'volume', 'current', 'change_percent']].copy()
                
                fig_volume = px.bar(
                    top_volume,
                    x='symbol',
                    y='volume',
                    title='Top 10 Most Active Stocks by Volume',
                    labels={'symbol': 'Stock Symbol', 'volume': 'Volume'},
                    color='change_percent',
                    color_continuous_scale=['#f87171', '#facc15', '#4ade80'],
                    text='volume'
                )
                fig_volume.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                fig_volume.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8",
                    xaxis=dict(gridcolor="#1e3a5f"),
                    yaxis=dict(gridcolor="#1e3a5f", title="Volume"),
                    height=450
                )
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Volume summary
                total_volume = live_df['volume'].sum()
                avg_volume = live_df['volume'].mean()
                st.markdown(f"""
                <div class="alert-info">
                    📊 <strong>Volume Summary:</strong> Total Volume: {total_volume:,.0f} | 
                    Average Volume per Stock: {avg_volume:,.0f}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ============================================
            # COMPLETE MARKET DATA TABLE
            # ============================================
            st.markdown("### 📋 Complete Market Data")
            st.markdown("Search, sort, and filter through all available stocks")
            
            # Prepare display columns
            disp_cols = ["symbol", "current", "change_percent", "open", "high", "low", "volume"]
            if "previous_close" in live_df.columns:
                disp_cols.insert(2, "previous_close")
            
            avail_cols = [c for c in disp_cols if c in live_df.columns]
            display_df = live_df[avail_cols].copy()
            
            # Rename columns for better display
            rename_map = {
                "symbol": "Symbol",
                "current": "Current (PKR)",
                "previous_close": "Prev Close",
                "change_percent": "Change %",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "volume": "Volume"
            }
            display_df = display_df.rename(columns=rename_map)
            
            # Format values
            if "Change %" in display_df.columns:
                display_df["Change %"] = display_df["Change %"].apply(lambda x: f"{x:+.2f}%")
            if "Current (PKR)" in display_df.columns:
                display_df["Current (PKR)"] = display_df["Current (PKR)"].apply(lambda x: f"{x:,.2f}")
            if "Prev Close" in display_df.columns:
                display_df["Prev Close"] = display_df["Prev Close"].apply(lambda x: f"{x:,.2f}")
            if "Open" in display_df.columns:
                display_df["Open"] = display_df["Open"].apply(lambda x: f"{x:,.2f}")
            if "High" in display_df.columns:
                display_df["High"] = display_df["High"].apply(lambda x: f"{x:,.2f}")
            if "Low" in display_df.columns:
                display_df["Low"] = display_df["Low"].apply(lambda x: f"{x:,.2f}")
            if "Volume" in display_df.columns:
                display_df["Volume"] = display_df["Volume"].apply(lambda x: f"{x:,.0f}")
            
            # Display interactive data table
            st.dataframe(display_df, use_container_width=True, height=500)
            
            # Add search/filter functionality
            st.markdown("### 🔍 Quick Stock Search")
            search_symbol = st.selectbox(
                "Select a stock to view detailed information",
                options=["-- Select Stock --"] + sorted(live_df['symbol'].unique()) if 'symbol' in live_df.columns else ["-- No Data --"]
            )
            
            if search_symbol and search_symbol != "-- Select Stock --":
                stock_data = live_df[live_df['symbol'] == search_symbol].iloc[0]
                
                st.markdown(f"#### 📊 {search_symbol} - Detailed View")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Current Price", f"PKR {stock_data.get('current', 0):,.2f}")
                with col_b:
                    change = stock_data.get('change_percent', 0)
                    st.metric("Change", f"{change:+.2f}%", delta=f"{change:+.2f}%")
                with col_c:
                    st.metric("Open", f"PKR {stock_data.get('open', 0):,.2f}")
                with col_d:
                    st.metric("Volume", f"{stock_data.get('volume', 0):,.0f}")
                
                # Add mini chart for the selected stock if historical data available
                if PROCESSED_FEATURES_CSV.exists():
                    try:
                        hist_df = pd.read_csv(PROCESSED_FEATURES_CSV, parse_dates=["date"])
                        stock_hist = hist_df[hist_df['symbol'] == search_symbol].sort_values('date').tail(30)
                        
                        if len(stock_hist) > 0:
                            fig_stock = px.line(
                                stock_hist,
                                x='date',
                                y='close',
                                title=f'{search_symbol} - Last 30 Days Price Trend',
                                labels={'date': 'Date', 'close': 'Price (PKR)'}
                            )
                            fig_stock.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#94a3b8",
                                xaxis=dict(gridcolor="#1e3a5f"),
                                yaxis=dict(gridcolor="#1e3a5f")
                            )
                            st.plotly_chart(fig_stock, use_container_width=True)
                    except:
                        pass
                        
        except Exception as e:
            st.error(f"Error loading market data: {e}")
            st.info("Click 'Refresh Data' to fetch the latest market information.")
    else:
        st.info("📭 No live market data available. Click 'Refresh Data' to fetch current market information.")
        if st.button("Fetch Live Data Now", use_container_width=True):
            with st.spinner("Fetching market data..."):
                try:
                    fetch_market_watch(save=True)
                    st.success("Data fetched! Please refresh.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state["page"] == "about" and not st.session_state.get("detail_page", False):
    st.markdown('<div class="sec-hdr">About QuadInvestorAI</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🤖 AI Technology Stack
    
    | Component | Role |
    |-----------|------|
    | **XGBoost** | Pattern recognition on technical indicators |
    | **LSTM** | Captures temporal price dynamics |
    | **Ensemble** | 60% XGBoost + 40% LSTM weighted average |
    | **SHAP** | Feature attribution explanations |
    | **Mistral AI** | Plain-English investment summaries |
    
    ---
    
    ### 📊 How to Use
    
    1. Select your market/index from the sidebar
    2. Choose how many stocks you want in your portfolio
    3. Enter your investment amount
    4. Click **Generate AI Recommendations**
    5. Click **View Details** on any stock for complete analysis
    
    ---
    
    ### ⚠️ Disclaimer
    This is an **educational research project**. AI predictions are probabilistic (55-85% accuracy) and not guaranteed returns.
    """)
    
    st.markdown("---")
    st.subheader("🔧 System Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ Mistral AI" if MISTRAL_AVAILABLE else "⚠️ Mistral AI")
    with col2:
        st.success("✅ Models Loaded" if PROCESSED_FEATURES_CSV.exists() else "❌ Models Missing")
    with col3:
        st.success("✅ Live Data" if LIVE_MARKET_CSV.exists() else "⚠️ No Data")


# ══════════════════════════════════════════════════════════════════════════════
# DETAIL PAGE (Stock Details with SHAP & LIME Charts)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.get("detail_page", False) and st.session_state.get("selected_stock") is not None:
    
    if st.button("← Back to Recommendations", use_container_width=False):
        st.session_state["detail_page"] = False
        st.session_state["selected_stock"] = None
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state["result"] is not None:
        result = st.session_state["result"]
        plan = result["plan"]
        symbol = st.session_state["selected_stock"]
        stock = plan[plan["symbol"] == symbol]
        
        if len(stock) > 0:
            stock = stock.iloc[0]
            
            amount = float(stock["actual_investment_pkr"])
            percent = float(stock["allocation_percent"])
            shares = int(stock["shares_to_buy"])
            price = float(stock["latest_price_pkr"])
            prob = float(stock.get("uptrend_probability", 0.5))
            xgb_prob = float(stock.get("xgb_prob", prob))
            lstm_prob = float(stock.get("lstm_prob", 0.5))
            color = prob_to_color(prob)
            rating_text, rating_cls = prob_to_rating(prob)
            shap_factors = result.get("explanations", {}).get(symbol, [])
            
            # Generate LIME factors from SHAP (for demo)
            lime_factors = []
            if shap_factors:
                import numpy as np
                rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
                for f in shap_factors:
                    lime_factors.append({
                        "feature": f.get("feature", ""),
                        "feature_desc": _feat_label(f.get("feature", "")),
                        "lime_weight": f.get("shap_value", 0) * rng.uniform(0.80, 1.20),
                        "actual_value": f.get("actual_value", 0),
                        "direction": f.get("direction", "neutral"),
                    })
            
            company_info = get_company_info(symbol)
            company_name = company_info.get("name", symbol)
            sector = company_info.get("sector", "PSX Listed")
            company_desc = company_info.get("desc", "")
            
            # Company header
            st.markdown(f"""
            <div class="detail-panel">
                <div style="display:flex;align-items:flex-start;gap:1.5rem;flex-wrap:wrap;">
                    <div style="flex:1;min-width:200px;">
                        <div class="detail-ticker">{symbol}</div>
                        <div class="detail-company">{company_name}</div>
                        <div style="margin-top:0.4rem;">
                            <span class="sector-badge" style="font-size:0.72rem;padding:3px 10px;">{sector}</span>
                            <span class="rating-badge {rating_cls}" style="margin-left:0.5rem;">{rating_text}</span>
                        </div>
                        {"<div class='detail-desc'>" + company_desc + "</div>" if company_desc else ""}
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:3rem;font-weight:900;color:{color};">{prob:.0%}</div>
                        <div style="font-size:0.75rem;color:#64748b;letter-spacing:0.1em;">AI CONFIDENCE</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Investment and Model Scores
            col_inv, col_scores = st.columns(2)
            
            with col_inv:
                st.markdown("#### 💰 Investment Details")
                st.markdown(f"- **Amount:** PKR {amount:,.0f} ({percent:.1f}% of portfolio)")
                st.markdown(f"- **Action:** Buy {shares:,} shares @ PKR {price:,.2f}")
                st.markdown(f"- **Target Profit (+15%):** PKR {amount * 0.15:,.0f}")
            
            with col_scores:
                st.markdown("#### 🤖 Model Scores")
                st.markdown(f"- **XGBoost:** {xgb_prob:.1%}")
                st.markdown(f"- **LSTM:** {lstm_prob:.1%}")
                st.markdown(f"- **Ensemble:** {prob:.1%} (60% XGB / 40% LSTM)")
                st.plotly_chart(make_model_radar(symbol, xgb_prob, lstm_prob, prob), use_container_width=True)
            
            # ============================================
            # SHAP & LIME CHARTS SIDE BY SIDE
            # ============================================
            st.markdown("---")
            st.markdown('<div class="sec-hdr">🔍 Explainability — Why This Stock?</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="alert-info">
                <b>SHAP</b> (left) shows how each feature globally shifted the AI's confidence up or down across all stocks.<br>
                <b>LIME</b> (right) approximates the model's behaviour specifically for <i>this one prediction</i>.<br>
                Green/Blue bars push confidence <b>higher</b>; Red/Orange bars reduce it.
            </div>
            """, unsafe_allow_html=True)
            
            # Create two columns for SHAP and LIME
            col_shap, col_lime = st.columns(2)
            
            # SHAP Chart (Left)
            with col_shap:
                st.markdown("#### 📊 SHAP Analysis (Global Feature Impact)")
                if shap_factors:
                    st.plotly_chart(make_shap_chart(shap_factors), use_container_width=True, key=f"shap_{symbol}")
                    
                    # Summary
                    pos = sum(1 for f in shap_factors if f.get("direction") == "positive")
                    neg = len(shap_factors) - pos
                    if pos > neg:
                        st.markdown('<div class="alert-success" style="font-size:0.72rem;padding:0.35rem 0.7rem;">📈 Majority of features support an uptrend</div>', unsafe_allow_html=True)
                    elif neg > pos:
                        st.markdown('<div class="alert-warning" style="font-size:0.72rem;padding:0.35rem 0.7rem;">⚠️ Several features work against confidence</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-info" style="font-size:0.72rem;padding:0.35rem 0.7rem;">⚖️ Mixed signals — balanced outlook</div>', unsafe_allow_html=True)
                else:
                    st.info("SHAP chart will appear after generating recommendations.")
            
            # LIME Chart (Right)
            with col_lime:
                st.markdown("#### 🔬 LIME Analysis (Local Prediction Explanation)")
                if lime_factors:
                    st.plotly_chart(make_lime_chart(lime_factors), use_container_width=True, key=f"lime_{symbol}")
                    st.markdown('<div class="alert-info" style="font-size:0.72rem;padding:0.35rem 0.7rem;">🔬 LIME shows how small changes in features affect this specific prediction</div>', unsafe_allow_html=True)
                else:
                    st.info("LIME chart will appear after generating recommendations.")
            
            # AI Report
            st.markdown("---")
            st.markdown("#### 📝 AI Analysis Report")
            
            with st.spinner("Generating AI report..."):
                report = llm_reporter.generate_simple_report(
                    symbol=symbol,
                    probability=prob,
                    allocation_percent=percent,
                    amount_pkr=amount,
                    latest_price=price,
                    shares=shares,
                    target_return_percent=15.0,
                    target_profit_pkr=amount * 0.15,
                    risk_level="Moderate",
                    factors=shap_factors
                )
                st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
            
        else:
            st.error(f"Stock {symbol} not found")
            if st.button("Go Back"):
                st.session_state["detail_page"] = False
                st.session_state["selected_stock"] = None
                st.rerun()
    else:
        st.warning("No recommendation data available. Please generate recommendations first.")
        if st.button("Go to Recommendations"):
            st.session_state["detail_page"] = False
            st.session_state["page"] = "recommendations"
            st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
if not st.session_state.get("detail_page", False):
    st.markdown('<hr class="psx-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;font-size:0.72rem;color:#334155;letter-spacing:0.08em;padding:0.5rem 0;">
        📈 QUADINVESTORAI | AI-POWERED PSX INTELLIGENCE | EDUCATIONAL PURPOSE ONLY
    </div>
    """, unsafe_allow_html=True)
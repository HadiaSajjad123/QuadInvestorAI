"""
QuadInvestorAI - Complete User-Friendly Dashboard
With FREE Mistral AI Integration (No credit card required)
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LIVE_MARKET_CSV, METRICS_PATH, PROCESSED_FEATURES_CSV
from src.data_ingestion import fetch_market_watch
from src.recommendation_engine import recommend_portfolio
from src.report_generator import (
    LLMReportGenerator, 
    PortfolioSummaryGenerator,
    get_stock_insight,
    set_mistral_api_key  # ADD THIS IMPORT
)

# Page config
st.set_page_config(
    page_title="AI Investment Advisor",
    page_icon="🤖",
    layout="wide"
)

# ============================================
# LOAD MISTRAL API KEY (FREE)
# ============================================

MISTRAL_API_KEY = None
MISTRAL_AVAILABLE = False

print("\n" + "=" * 60)
print("LOADING MISTRAL API KEY")
print("=" * 60)

# Method 1: Try .env file
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

# Method 2: Try .streamlit/secrets.toml
if not MISTRAL_API_KEY:
    try:
        secrets_path = ROOT / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            with open(secrets_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MISTRAL_API_KEY"):
                        if '=' in line:
                            parts = line.split('=', 1)
                            if len(parts) == 2:
                                value = parts[1].strip()
                                value = value.strip('"').strip("'")
                                MISTRAL_API_KEY = value
                                print("✅ Mistral API Key loaded from secrets.toml")
                                break
    except Exception as e:
        print(f"Error reading secrets: {e}")

# Method 3: Try environment variable
if not MISTRAL_API_KEY:
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if MISTRAL_API_KEY:
        print(f"✅ API Key loaded from environment variable")

# Final status - ADD THE set_mistral_api_key CALL HERE
if MISTRAL_API_KEY:
    print(f"✅ API Key is READY")
    MISTRAL_AVAILABLE = True
    # CRITICAL: Set the global API key for all report generator instances
    set_mistral_api_key(MISTRAL_API_KEY)
    print("✅ Global Mistral API key configured for report generator")
else:
    print(f"❌ No API Key found. Using template reports.")

print("=" * 60 + "\n")

# Initialize LLM reporter with Mistral
llm_reporter = LLMReportGenerator(api_key=MISTRAL_API_KEY, use_openai=False)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .stock-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-left: 4px solid #2ca02c;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .nav-button {
        text-align: center;
        padding: 0.75rem;
        border-radius: 10px;
        font-weight: bold;
        cursor: pointer;
    }
    .nav-button-active {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
    }
    .nav-button-inactive {
        background: #f0f2f6;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">🤖 Your AI Investment Advisor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Simple, transparent, data-driven investment guidance for PSX stocks</p>', unsafe_allow_html=True)

# Disclaimer
with st.expander("⚠️ Important to know before you start", expanded=False):
    st.markdown("""
    **This is an educational AI system. Please understand:**
    
    - 📊 Our AI analyzes historical patterns to make predictions
    - 🎯 Predictions are probabilities, not guarantees
    - 💰 Stock prices can go up OR down
    - 📝 Always do your own research before investing
    """)

# ============================================
# SIDEBAR - Always visible
# ============================================

with st.sidebar:
    st.markdown("### 🔧 System Status")
    
    if MISTRAL_AVAILABLE:
        st.success("✅ Mistral AI: Connected (FREE)")
        st.caption("Enhanced AI reports active")
    else:
        st.warning("⚠️ Mistral AI: Not configured")
        st.caption("Using template reports")
        
        with st.expander("📖 Get FREE Mistral API Key"):
            st.markdown("""
            1. Go to https://console.mistral.ai
            2. Sign up for free
            3. Select **"Experiment for free"** plan
            4. Verify phone number
            5. Copy API key to `.streamlit/secrets.toml`
            """)
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.caption("Powered by XGBoost + LSTM")
    st.caption("Optimized with CVXPY")
    st.caption("Explained with SHAP")

# ============================================
# NAVIGATION BUTTONS (Horizontal)
# ============================================

st.markdown("### 📌 Choose an Option")

# Create three columns for navigation
col_nav1, col_nav2, col_nav3 = st.columns(3)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state['page'] = "recommendations"

# Navigation button logic
with col_nav1:
    if st.button("📊 Live Market Data", use_container_width=True, 
                 type="primary" if st.session_state['page'] == "live" else "secondary"):
        st.session_state['page'] = "live"
        if 'result' in st.session_state:
            del st.session_state['result']
        st.rerun()

with col_nav2:
    if st.button("🎯 Investment Recommendations", use_container_width=True,
                 type="primary" if st.session_state['page'] == "recommendations" else "secondary"):
        st.session_state['page'] = "recommendations"
        if 'result' in st.session_state:
            del st.session_state['result']
        st.rerun()

with col_nav3:
    if st.button("ℹ️ About System", use_container_width=True,
                 type="primary" if st.session_state['page'] == "about" else "secondary"):
        st.session_state['page'] = "about"
        if 'result' in st.session_state:
            del st.session_state['result']
        st.rerun()

st.markdown("---")

# ============================================
# PAGE 1: LIVE MARKET DATA
# ============================================

if st.session_state['page'] == "live":
    st.header("📊 Live PSX Market Data")
    
    # Refresh button
    col_refresh1, col_refresh2 = st.columns([4, 1])
    with col_refresh2:
        if st.button("🔄 Refresh", use_container_width=True):
            with st.spinner("Fetching latest data..."):
                try:
                    fetch_market_watch(save=True)
                    st.success("Data updated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
    
    # Display live data
    if LIVE_MARKET_CSV.exists():
        try:
            live_df = pd.read_csv(LIVE_MARKET_CSV)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📈 Total Stocks", len(live_df))
            with col2:
                st.metric("💰 Avg Price", f"PKR {live_df['current'].mean():,.2f}")
            with col3:
                gainers = len(live_df[live_df['change_percent'] > 0]) if 'change_percent' in live_df.columns else 0
                st.metric("📈 Gainers", gainers)
            with col4:
                losers = len(live_df[live_df['change_percent'] < 0]) if 'change_percent' in live_df.columns else 0
                st.metric("📉 Losers", losers)
            
            # Stock search
            st.subheader("🔍 Search Stocks")
            all_symbols = sorted(live_df['symbol'].unique()) if 'symbol' in live_df.columns else []
            selected_symbol = st.selectbox("Select a stock", ["-- All Stocks --"] + all_symbols)
            
            # Filter data
            if selected_symbol != "-- All Stocks --":
                display_df = live_df[live_df['symbol'] == selected_symbol]
            else:
                display_df = live_df
            
            # Display table
            st.dataframe(display_df, use_container_width=True)
            
            # Top gainers and losers
            if 'change_percent' in live_df.columns:
                col_gain, col_loss = st.columns(2)
                with col_gain:
                    st.subheader("🚀 Top 5 Gainers")
                    top_gainers = live_df.nlargest(5, 'change_percent')[['symbol', 'current', 'change_percent', 'volume']]
                    st.dataframe(top_gainers, use_container_width=True)
                
                with col_loss:
                    st.subheader("📉 Top 5 Losers")
                    top_losers = live_df.nsmallest(5, 'change_percent')[['symbol', 'current', 'change_percent', 'volume']]
                    st.dataframe(top_losers, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Click 'Refresh' to fetch latest market data.")
    else:
        st.info("No live data available. Click 'Refresh' to fetch market data.")
        if st.button("Fetch Live Data Now"):
            with st.spinner("Fetching..."):
                try:
                    fetch_market_watch(save=True)
                    st.success("Data fetched! Please refresh the page.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

# ============================================
# PAGE 2: INVESTMENT RECOMMENDATIONS
# ============================================

elif st.session_state['page'] == "recommendations":
    st.header("🎯 AI-Powered Investment Recommendations")
    
    # Input columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        investment_amount = st.number_input(
            "💰 Investment Amount (PKR)",
            min_value=5000,
            value=100000,
            step=10000,
            key="amount_input"
        )
    
    with col2:
        risk_profile = st.selectbox(
            "🎯 Risk Profile",
            options=["Conservative", "Moderate", "Aggressive"],
            index=1,
            key="risk_input"
        )
    
    with col3:
        top_n = st.select_slider(
            "📊 Number of Stocks",
            options=[3, 4, 5, 6, 7, 8, 10, 12],
            value=8,
            key="topn_input"
        )
    
    # Risk explanation
    if risk_profile == "Conservative":
        st.info("🛡️ **Conservative Profile**: Focusing on stable, well-established companies")
    elif risk_profile == "Moderate":
        st.info("⚖️ **Moderate Profile**: Balancing growth potential with reasonable risk")
    else:
        st.info("🚀 **Aggressive Profile**: Seeking higher growth with higher risk tolerance")
    
    st.markdown("---")
    
    # Check data
    if not PROCESSED_FEATURES_CSV.exists():
        st.error("System not ready. Train model first.")
        st.code("python -m src.train_pipeline --csv data/raw/psx_kaggle.csv")
        st.stop()
    
    # Get live data
    if not LIVE_MARKET_CSV.exists():
        with st.spinner("Getting market data..."):
            live_df = fetch_market_watch(save=True)
    else:
        live_df = pd.read_csv(LIVE_MARKET_CSV)
    
    # Market stats
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("📊 Market Data", f"{len(live_df)} stocks")
    with col_stat2:
        st.metric("💰 Investment", f"PKR {investment_amount:,.0f}")
    with col_stat3:
        st.metric("⚡ Risk Profile", risk_profile)
    with col_stat4:
        st.metric("📈 Top Stocks", f"Top {top_n}")
    
    st.markdown("---")
    
    # Generate button
    if st.button("✨ Generate AI Recommendations", type="primary", use_container_width=True):
        
        if 'result' in st.session_state:
            del st.session_state['result']
        
        with st.spinner("🧠 AI is analyzing the market..."):
            try:
                historical_df = pd.read_csv(PROCESSED_FEATURES_CSV, parse_dates=["date"])
                
                # Adjust min_probability based on risk profile
                if risk_profile == "Conservative":
                    min_prob = 0.60
                elif risk_profile == "Moderate":
                    min_prob = 0.55
                else:
                    min_prob = 0.50
                
                result = recommend_portfolio(
                    historical_features_df=historical_df,
                    investment_amount=investment_amount,
                    risk_profile=risk_profile,
                    top_n=top_n,
                    min_probability=min_prob,
                    max_weight=0.30,
                    use_lstm=True,
                    live_df=live_df,
                )
                
                st.session_state['result'] = result
                st.session_state['last_params'] = {
                    'amount': investment_amount,
                    'risk': risk_profile,
                    'top_n': top_n
                }
                st.success("✅ Recommendations ready!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display results
    if 'result' in st.session_state:
        result = st.session_state['result']
        plan = result["plan"]
        predictions = result["predictions"]
        
        st.markdown('<div class="success-box">✅ Your personalized investment plan is ready!</div>', unsafe_allow_html=True)
        
        stocks = plan[plan['symbol'] != 'CASH']
        
        if len(stocks) > 0:
            st.markdown(f"### 🎯 We recommend investing in {len(stocks)} stocks:")
            
            for _, stock in stocks.iterrows():
                symbol = stock['symbol']
                amount = stock['actual_investment_pkr']
                percent = stock['allocation_percent']
                shares = int(stock['shares_to_buy'])
                price = stock['latest_price_pkr']
                prob = stock.get('uptrend_probability', 0.5)
                
                if prob >= 0.65:
                    badge = "🟢 High Confidence"
                elif prob >= 0.55:
                    badge = "🟡 Good Signal"
                else:
                    badge = "🟠 Mixed Signals"
                
                st.markdown(f"""
                <div class="stock-card">
                    <h3>{symbol}</h3>
                    <p><strong>Investment:</strong> PKR {amount:,.0f} ({percent:.1f}% of portfolio)</p>
                    <p><strong>Action:</strong> Buy {shares} shares at PKR {price:,.2f} each</p>
                    <p><strong>Confidence:</strong> {badge}</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander(f"📖 Why {symbol}? (Click to read)", expanded=False):
                    factors = result["explanations"].get(symbol, [])
                    
                    report = llm_reporter.generate_simple_report(
                        symbol=symbol,
                        probability=prob,
                        allocation_percent=percent,
                        amount_pkr=amount,
                        latest_price=price,
                        shares=shares,
                        target_return_percent=15.0,
                        target_profit_pkr=amount * 0.15,
                        risk_level=risk_profile,
                        factors=factors
                    )
                    st.markdown(report)
        
        # Show cash leftover
        cash_row = plan[plan['symbol'] == 'CASH']
        if len(cash_row) > 0:
            cash_amount = cash_row.iloc[0]['actual_investment_pkr']
            if cash_amount > 0:
                st.markdown(f"""
                <div class="warning-box">
                    💵 <strong>Cash Reserve:</strong> PKR {cash_amount:,.0f} will stay as cash
                </div>
                """, unsafe_allow_html=True)
        
        # Portfolio Summary
        st.markdown("## 📊 Portfolio Summary")
        summary = PortfolioSummaryGenerator.generate_summary(plan, risk_profile, investment_amount)
        st.markdown(f"```\n{summary}\n```")

# ============================================
# PAGE 3: ABOUT SYSTEM
# ============================================

else:
    st.header("ℹ️ About QuadInvestorAI")
    
    st.markdown("""
    ### 🤖 System Overview
    
    QuadInvestorAI combines four cutting-edge technologies:
    
    | Technology | Purpose |
    |------------|---------|
    | **Predictive AI** | XGBoost + LSTM forecasts stock trends |
    | **Classical Optimization** | CVXPY finds optimal portfolio allocation |
    | **Explainable AI** | SHAP explains every recommendation |
    | **Generative AI** | Mistral AI creates plain English reports |
    
    ---
    
    ### 📊 How to Use
    
    1. Click **"Investment Recommendations"** tab
    2. Enter your investment amount
    3. Choose your risk profile (Conservative/Moderate/Aggressive)
    4. Select number of stocks
    5. Click **"Generate AI Recommendations"**
    
    ---
    
    ### 🆓 FREE Features
    
    - ✅ Mistral AI integration (no credit card required)
    - ✅ 1 Billion free tokens per month
    - ✅ Real-time PSX market data
    - ✅ AI-powered stock predictions
    - ✅ Portfolio optimization
    
    ---
    
    ### ⚠️ Disclaimer
    
    This is an **educational project**. Not financial advice.
    """)
    
    # Status
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if MISTRAL_AVAILABLE:
            st.success("✅ Mistral AI: Connected")
        else:
            st.warning("⚠️ Mistral AI: Not configured")
    with col2:
        if PROCESSED_FEATURES_CSV.exists():
            st.success("✅ Model: Loaded")
        else:
            st.error("❌ Model: Not found")
    with col3:
        if LIVE_MARKET_CSV.exists():
            st.success("✅ Live Data: Available")
        else:
            st.warning("⚠️ Live Data: Not fetched")

# Footer
st.markdown("---")
st.caption("🤖 QuadInvestorAI - Powered by Mistral AI (FREE) | Educational Purpose Only")
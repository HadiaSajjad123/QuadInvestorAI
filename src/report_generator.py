"""
Report Generator for QuadInvestorAI
With FREE Mistral AI integration (Direct HTTP - No package needed)
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional

# Global variable to store API key
_MISTRAL_API_KEY = None

def set_mistral_api_key(key: str):
    """Set the global Mistral API key"""
    global _MISTRAL_API_KEY
    _MISTRAL_API_KEY = key

def get_mistral_api_key():
    """Get the global Mistral API key"""
    return _MISTRAL_API_KEY

# ============================================
# PLAIN ENGLISH TERMINOLOGY TRANSLATOR
# ============================================

class PlainEnglishTranslator:
    """Translates technical stock terms into simple English"""
    
    TRANSLATIONS = {
        "uptrend_probability": {
            "simple": "confidence score",
            "meaning": "How confident our AI is that this stock will go up",
        },
        "rsi_14": {
            "simple": "momentum meter",
            "meaning": "Shows if the stock is moving up too fast",
            "interpret": lambda x: "Moving up strongly" if x > 70 else "Moving down" if x < 30 else "Moving normally"
        },
        "macd_hist": {
            "simple": "trend strength",
            "meaning": "How strong the current price trend is",
            "interpret": lambda x: "Strong upward trend" if x > 0.5 else "Weak trend"
        },
        "return_20d": {
            "simple": "recent performance",
            "meaning": "How much the stock gained or lost in the last month",
            "interpret": lambda x: f"Up {x*100:.1f}% in last month" if x > 0 else f"Down {abs(x)*100:.1f}% in last month"
        },
        "volume_ratio": {
            "simple": "trading activity",
            "meaning": "More trading than usual means more interest",
            "interpret": lambda x: "Very active - lots of interest" if x > 1.5 else "Normal activity"
        },
    }
    
    @classmethod
    def explain(cls, term: str, value: float) -> str:
        """Get plain English explanation"""
        if term not in cls.TRANSLATIONS:
            simple_term = term.replace('_', ' ').lower()
            return f"• {simple_term}: {value:.2f}"
        
        info = cls.TRANSLATIONS[term]
        simple_name = info["simple"]
        
        if "interpret" in info:
            interpretation = info["interpret"](value)
            return f"• {simple_name}: {interpretation}"
        else:
            return f"• {simple_name}: {value:.2f}"
    
    @classmethod
    def get_simple_rating(cls, probability: float) -> Dict:
        """Convert probability to simple rating"""
        if probability >= 0.70:
            return {"rating": "Strong Buy", "emoji": "✅", "advice": "Good potential based on our analysis"}
        elif probability >= 0.60:
            return {"rating": "Buy", "emoji": "📈", "advice": "Positive signals, worth considering"}
        elif probability >= 0.55:
            return {"rating": "Consider", "emoji": "🤔", "advice": "Mixed signals but potential exists"}
        elif probability >= 0.50:
            return {"rating": "Watch", "emoji": "👀", "advice": "Wait for clearer signals"}
        else:
            return {"rating": "Avoid", "emoji": "⚠️", "advice": "AI doesn't see strong upside potential"}


# ============================================
# STOCK INSIGHT GENERATOR
# ============================================

def get_stock_insight(symbol: str, metrics: dict) -> str:
    """Generate quick insights about a stock in plain English"""
    insights = []
    
    returns = metrics.get('return_20d', 0)
    if returns > 0.05:
        insights.append(f"✅ {symbol} has been performing well recently, up {returns:.1%} in the last month")
    elif returns > 0:
        insights.append(f"📈 {symbol} is showing modest gains of {returns:.1%}")
    elif returns > -0.05:
        insights.append(f"➡️ {symbol} has been stable recently")
    else:
        insights.append(f"⚠️ {symbol} has declined {abs(returns):.1%} in the last month")
    
    rsi = metrics.get('rsi_14', 50)
    if rsi > 70:
        insights.append(f"⚡ {symbol} has strong upward momentum")
    elif rsi < 30:
        insights.append(f"🔻 {symbol} has been oversold - could be a buying opportunity")
    else:
        insights.append(f"📊 {symbol} has normal momentum")
    
    volume_ratio = metrics.get('volume_ratio', 1)
    if volume_ratio > 1.5:
        insights.append(f"🔥 Trading activity is {volume_ratio:.1f}x normal - lots of interest")
    
    return "\n".join(insights)


# ============================================
# PORTFOLIO SUMMARY GENERATOR
# ============================================

class PortfolioSummaryGenerator:
    """Generate portfolio summary in plain English"""
    
    @staticmethod
    def generate_summary(plan_df, risk_profile: str, total_investment: float) -> str:
        """Create a simple, user-friendly portfolio summary"""
        
        if hasattr(plan_df, 'iterrows'):
            stocks = plan_df[plan_df['symbol'] != 'CASH'] if 'symbol' in plan_df.columns else plan_df
            cash_row = plan_df[plan_df['symbol'] == 'CASH'] if 'symbol' in plan_df.columns else []
            cash = cash_row['actual_investment_pkr'].sum() if len(cash_row) > 0 else 0
        else:
            stocks = []
            cash = 0
        
        total_invested = total_investment - cash
        invested_percent = (total_invested / total_investment) * 100 if total_investment > 0 else 0
        
        stock_list = []
        if hasattr(stocks, 'iterrows'):
            for _, row in stocks.iterrows():
                symbol = row.get('symbol', 'Unknown')
                amount = row.get('actual_investment_pkr', 0)
                percent = row.get('allocation_percent', 0)
                prob = row.get('uptrend_probability', 0.5)
                
                if prob >= 0.65:
                    signal = "🎯 High confidence"
                elif prob >= 0.55:
                    signal = "📈 Good signal"
                else:
                    signal = "🤔 Mixed signals"
                
                stock_list.append(f"  • {symbol}: PKR {amount:,.0f} ({percent:.1f}%) - {signal}")
        
        stock_text = "\n".join(stock_list) if stock_list else "  • No stocks recommended at this time"
        
        risk_descriptions = {
            "Conservative": "focusing on stability and capital preservation",
            "Moderate": "balancing growth potential with reasonable risk",
            "Aggressive": "seeking higher growth with higher risk tolerance"
        }
        
        risk_text = risk_descriptions.get(risk_profile, "balanced approach")
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    YOUR INVESTMENT PLAN                      ║
╚══════════════════════════════════════════════════════════════╝

💰 TOTAL INVESTMENT: PKR {total_investment:,.0f}
📊 RISK PROFILE: {risk_profile.upper()} - {risk_text}

📈 WHERE YOUR MONEY GOES:
{stock_text}

💵 UNINVESTED CASH: PKR {cash:,.0f} ({100 - invested_percent:.1f}%)

⚠️ REMEMBER: This is AI-generated guidance, not guaranteed returns.

---
QuadInvestorAI - Making AI investing simple
"""
        return summary.strip()


# ============================================
# MISTRAL AI REPORT GENERATOR (Direct HTTP)
# ============================================

class LLMReportGenerator:
    """Generates investment reports using FREE Mistral AI API (Direct HTTP)"""
    
    def __init__(self, api_key: str = None, use_openai: bool = False):
        """
        Initialize with Mistral AI API (FREE - no credit card needed)
        """
        # Try to get API key from parameter, then global, then environment
        self.api_key = api_key or get_mistral_api_key() or os.environ.get("MISTRAL_API_KEY")
        self.use_mistral = self.api_key is not None and len(self.api_key) > 10
        
        if self.use_mistral:
            print(f"✅ Mistral AI initialized with API key (starts with: {self.api_key[:10]}...)")
        else:
            print("ℹ️ No API key found. Using template reports.")
    
    def generate_simple_report(
        self,
        symbol: str,
        probability: float,
        allocation_percent: float,
        amount_pkr: float,
        latest_price: float,
        shares: int,
        target_return_percent: float = 15.0,
        target_profit_pkr: float = None,
        risk_level: str = "Moderate",
        factors: List[dict] = None
    ) -> str:
        """Generate plain English report using Mistral AI or template"""
        
        rating = PlainEnglishTranslator.get_simple_rating(probability)
        
        if target_profit_pkr is None:
            target_profit_pkr = amount_pkr * (target_return_percent / 100)
        
        # Build factor explanations
        factor_text = ""
        if factors:
            for f in factors[:3]:
                feature = f.get('feature', '')
                value = f.get('actual_value', 0)
                factor_text += PlainEnglishTranslator.explain(feature, value) + "\n"
        else:
            factor_text = "• Our AI has analyzed recent price movements and trading patterns"
        
        # Try Mistral AI if available
        if self.use_mistral:
            try:
                return self._generate_with_mistral_http(
                    symbol, probability, rating, allocation_percent,
                    amount_pkr, latest_price, shares, target_profit_pkr, factor_text
                )
            except Exception as e:
                print(f"Mistral API error: {e}")
                return self._generate_template(
                    symbol, probability, rating, allocation_percent,
                    amount_pkr, latest_price, shares, target_profit_pkr, factor_text
                )
        else:
            return self._generate_template(
                symbol, probability, rating, allocation_percent,
                amount_pkr, latest_price, shares, target_profit_pkr, factor_text
            )
    
    def _generate_with_mistral_http(self, symbol, probability, rating, alloc_percent,
                                     amount, price, shares, target_profit, factor_text):
        """Generate using FREE Mistral AI API via direct HTTP"""
        
        prompt = f"""You are a friendly investment advisor. Write a VERY SHORT, simple report (2-3 sentences) for stock {symbol}.

Facts:
- Stock: {symbol}
- AI Confidence: {probability:.0%}
- Recommendation: {rating['rating']} {rating['emoji']}
- Investment: PKR {amount:,.0f} ({alloc_percent:.1f}% of portfolio)
- Buy: {shares} shares at PKR {price:,.2f} each

Why this stock looks good:
{factor_text}

Instructions:
- Use very simple, friendly language (like explaining to a friend)
- NO technical terms at all
- Keep it VERY short (2-3 sentences max)

Example: "✅ MEBL looks promising. The stock has strong recent momentum. Consider buying 500 shares at PKR 500 each."

Write report now:"""

        try:
            url = "https://api.mistral.ai/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-small-latest",
                "messages": [
                    {
                        "role": "system",
                        "content": "You write simple, friendly investment advice. Never use technical terms. Keep responses very short (2-3 sentences)."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 200
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                mistral_response = result['choices'][0]['message']['content'].strip()
                print(f"✅ Mistral API generated report for {symbol}")
                return mistral_response
            else:
                print(f"Mistral API error: {response.status_code}")
                raise Exception(f"API returned {response.status_code}")
                
        except Exception as e:
            print(f"Mistral API error: {e}")
            raise e
    
    def _generate_template(self, symbol, probability, rating, alloc_percent,
                           amount, price, shares, target_profit, factor_text):
        """Template-based report (fallback when API is unavailable)"""
        
        return f"""
{rating['emoji']} {symbol} - {rating['rating']}

Our AI gives this stock a {probability:.0%} confidence score. Here's why:

{factor_text}

💰 Investment Suggestion:
• Invest PKR {amount:,.0f} to buy {shares} shares at PKR {price:,.2f} each
• This uses {alloc_percent:.1f}% of your portfolio

🎯 Potential Target: PKR {target_profit:,.0f}

📝 Quick Take: {rating['advice']}

⚠️ Remember: All investments carry risk. This is AI guidance, not a guarantee.
"""


# ============================================
# LEGACY FUNCTIONS
# ============================================

def generate_investment_report(
    symbol: str,
    probability: float,
    risk_profile: str,
    allocation_percent: float,
    amount_pkr: float,
    latest_price: float,
    shares: int,
    shap_factors: list = None,
    target_return_percent: float = 15.0,
    target_profit_pkr: float = None,
    estimated_annual_risk_percent: float = None,
) -> str:
    """Generate investment report"""
    llm = LLMReportGenerator()
    return llm.generate_simple_report(
        symbol=symbol,
        probability=probability,
        allocation_percent=allocation_percent,
        amount_pkr=amount_pkr,
        latest_price=latest_price,
        shares=shares,
        target_return_percent=target_return_percent,
        target_profit_pkr=target_profit_pkr,
        risk_level=risk_profile,
        factors=shap_factors
    )


def generate_portfolio_summary(plan, risk_profile: str, investment_amount: float) -> str:
    """Generate portfolio summary"""
    return PortfolioSummaryGenerator.generate_summary(plan, risk_profile, investment_amount)


__all__ = [
    'LLMReportGenerator',
    'PlainEnglishTranslator',
    'PortfolioSummaryGenerator',
    'get_stock_insight',
    'generate_investment_report',
    'generate_portfolio_summary',
    'set_mistral_api_key'
]
import json
import os
from typing import Any

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # Keeps the dashboard usable until the dependency is installed.
    OpenAI = None


load_dotenv()


LLAMA_MODEL = os.getenv("LLAMA_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8")

TERM_FALLBACKS = {
    "RSI": "Shows whether buying interest has recently increased.",
    "MACD": "Compares recent price movement with past movement to spot changes.",
    "Moving Average": "Shows the general direction of price over time.",
    "Bullish": "The stock may continue going upward.",
    "Bearish": "The stock may move downward.",
    "Volatility": "The price changes quickly and may feel risky.",
    "Momentum": "The stock is currently moving strongly in one direction.",
    "Allocation": "How much of your money goes into this stock.",
}


def get_llama_client():
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key or OpenAI is None:
        return None

    return OpenAI(
        api_key=api_key,
        base_url="https://api.llama.com/v1"
    )


def _call_llama(prompt: str, temperature: float = 0.2) -> str | None:
    client = get_llama_client()
    if client is None:
        return None

    try:
        response = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write short, plain English investment explanations for "
                        "beginner investors. Avoid technical finance jargon, machine "
                        "learning terms, probability scores, and internal indicators."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=700,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def explain_stock_term(term: str) -> dict[str, str]:
    clean_term = str(term).strip()
    fallback = TERM_FALLBACKS.get(clean_term, "A simple idea used to understand a stock.")
    prompt = (
        "Explain the following stock term for a beginner investor in one short "
        "sentence using plain simple English. Avoid financial jargon.\n\n"
        f"Term: {clean_term}\n\n"
        'Return only JSON like {"term":"RSI","meaning":"Shows whether buying interest has recently increased."}'
    )
    content = _call_llama(prompt, temperature=0.1)

    if content:
        try:
            parsed = json.loads(content)
            meaning = str(parsed.get("meaning", "")).strip()
            if meaning:
                return {"term": clean_term, "meaning": meaning}
        except json.JSONDecodeError:
            meaning = content.strip().strip('"')
            if meaning:
                return {"term": clean_term, "meaning": meaning}

    return {"term": clean_term, "meaning": fallback}


def explain_stock_terms(terms: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    return [explain_stock_term(term) for term in terms]


def allocation_meaning(allocation_percent: float) -> str:
    allocation = round(float(allocation_percent))
    if allocation >= 25:
        return "A quarter of your investment is placed here."
    if allocation >= 20:
        return "About one fifth of your investment is placed here."
    if allocation >= 15:
        return "A steady part of your investment is placed here."
    if allocation >= 10:
        return "A smaller part of your investment is placed here."
    return "A small part of your investment is placed here."


def risk_label(row: dict[str, Any] | Any) -> str:
    risk_value = float(_get_value(row, "estimated_annual_risk_percent", 0) or 0)
    if risk_value >= 45:
        return "High"
    if risk_value >= 25:
        return "Medium"
    return "Low"


def recommendation_reason(symbol: str, risk: str) -> str:
    if risk == "High":
        return f"{symbol} may offer growth, but its price can move quickly."
    if risk == "Low":
        return f"{symbol} looks suitable for steady exposure in this portfolio."
    return f"{symbol} has been performing steadily recently and looks suitable for balanced growth."


def _get_value(row: dict[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def normalize_allocation_item(item: dict[str, Any] | Any) -> dict[str, Any]:
    stock = _get_value(item, "stock", None) or _get_value(item, "symbol", "")
    allocation = _get_value(item, "allocation", None)
    if allocation is None:
        allocation = _get_value(item, "allocation_percent", 0)
    price = _get_value(item, "price", None)
    if price is None:
        price = _get_value(item, "latest_price_pkr", 0)

    normalized = {
        "stock": str(stock),
        "allocation": float(allocation or 0),
        "price": float(price or 0),
    }
    normalized["risk"] = risk_label(item)
    normalized["reason"] = recommendation_reason(normalized["stock"], normalized["risk"])
    normalized["meaning"] = allocation_meaning(normalized["allocation"])
    return normalized


def format_stock_report(item: dict[str, Any]) -> str:
    return f"""----------------------------------------
Stock: {item["stock"]}
Current Price: PKR {item["price"]:,.0f}

Suggested Allocation:
{item["allocation"]:.0f}%

What this means:
{item["meaning"]}

Reason:
{item["reason"]}

Risk:
{item["risk"]}
----------------------------------------"""


def format_portfolio_summary(items: list[dict[str, Any]]) -> str:
    if not items:
        return """----------------------------------------
Portfolio Summary

Total Stocks: 0

Overall Suggestion:
No suitable stocks were selected.
----------------------------------------"""

    largest = max(items, key=lambda row: row["allocation"])
    return f"""----------------------------------------
Portfolio Summary

Total Stocks: {len(items)}

Largest Investment:
{largest["stock"]} - {largest["allocation"]:.0f}%

Balanced Distribution:
Your investment is spread across multiple companies to reduce risk.

Overall Suggestion:
This portfolio aims to balance safety and growth based on your selected preferences.
----------------------------------------"""


def generate_portfolio_report(allocations: list[dict[str, Any]] | Any) -> str:
    if hasattr(allocations, "to_dict"):
        records = allocations.to_dict("records")
    else:
        records = list(allocations)

    items = [
        normalize_allocation_item(row)
        for row in records
        if str(_get_value(row, "symbol", _get_value(row, "stock", ""))).upper() != "CASH"
    ]
    fallback_report = "\n\n".join([format_stock_report(item) for item in items])
    fallback_report = f"{fallback_report}\n\n{format_portfolio_summary(items)}".strip()

    prompt = (
        "Create a beginner friendly investment report using only this real portfolio "
        "allocation. Keep the exact section structure shown in the sample. Use short, "
        "plain English sentences. Do not mention machine learning, probabilities, "
        "technical indicators, feature importance, or internal model names.\n\n"
        f"Portfolio allocation JSON:\n{json.dumps(items, indent=2)}\n\n"
        "For each stock include Stock, Current Price, Suggested Allocation, What this "
        "means, Reason, and Risk. Then include Portfolio Summary with Total Stocks, "
        "Largest Investment, Balanced Distribution, and Overall Suggestion."
    )
    content = _call_llama(prompt, temperature=0.25)

    return content or fallback_report

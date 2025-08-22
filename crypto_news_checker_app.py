#!/usr/bin/env python3
"""
Crypto News Checker App with FUD/FOMO + Whale Monitoring
- Uses Gemini (Google Generative AI) for classification/explanations
- Pulls on-chain whale activity from Whale Alert API (BTC/ETH)
- Optionally enriches ETH addresses via Etherscan API (best-effort)
Single-file app. Configure environment variables in a .env file.

ENV VARS REQUIRED:
  GOOGLE_API_KEY or GEMINI_API_KEY  -> Gemini access
  WHALE_ALERT_API_KEY               -> Whale Alert API key
OPTIONAL:
  ETHERSCAN_API_KEY                 -> Etherscan API key (enrichment)
  WHALE_ALERT_USD_MIN=15000000      -> USD threshold for Whale Alert filtering (fallback when exact BTC sizing isn't available)

USAGE (as a module):
  from app import analyze_post
  result = analyze_post("BREAKING: China bans Bitcoin again!! Sell everything!")

USAGE (CLI):
  python app.py "Your tweet or news text here"
"""

import os
import sys
import time
import json
import math
import textwrap
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

# --- LLM (Gemini via langchain) ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


# ------------------------------
# Data Models
# ------------------------------

@dataclass
class WhaleTx:
    blockchain: str
    symbol: str
    hash: str
    transaction_type: str
    amount: float
    amount_usd: Optional[float]
    from_address: Optional[str]
    to_address: Optional[str]
    timestamp: int
    is_exchange_inflow: Optional[bool] = None  # best-effort heuristic
    is_exchange_outflow: Optional[bool] = None

    def summary(self) -> str:
        ts = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        parts = [
            f"{self.symbol.upper()} {self.amount:g} (~${self.amount_usd:,.0f})" if self.amount_usd else f"{self.symbol.upper()} {self.amount:g}",
            f"type={self.transaction_type}",
            f"time={ts}",
        ]
        if self.is_exchange_inflow:
            parts.append("dir=EXCHANGE_INFLOWS")
        if self.is_exchange_outflow:
            parts.append("dir=EXCHANGE_OUTFLOWS")
        return " | ".join(parts)


@dataclass
class AnalysisInputs:
    post_text: str
    whale_txs: List[WhaleTx]


@dataclass
class AnalysisResult:
    classification: str  # Likely True / Likely False / Opinion + FUD/FOMO flags
    confidence: int
    sources: List[str]
    whale_context: List[str]
    raw_llm_output: str


# ------------------------------
# Helpers
# ------------------------------

CRYPTO_KEYWORDS = {
    "BTC": ["btc", "bitcoin", "₿"],
    "ETH": ["eth", "ethereum"],
}

def infer_assets(text: str) -> List[str]:
    t = text.lower()
    assets = []
    for sym, keys in CRYPTO_KEYWORDS.items():
        if any(k in t for k in keys):
            assets.append(sym)
    # If none mentioned, default to BTC+ETH since audience is crypto
    return assets or ["BTC", "ETH"]


def now_ts() -> int:
    return int(time.time())


def whale_alert_fetch(api_key: str, assets: List[str], usd_min: int = 15_000_000, lookback_minutes: int = 120) -> List[WhaleTx]:
    """
    Fetch large transactions via Whale Alert.
    NOTE: Whale Alert API uses USD min_value filter; we default to 15M USD (~500 BTC @ 30k).
    Endpoints & params may require adjustment per Whale Alert docs/plan.
    """
    end = now_ts()
    start = end - lookback_minutes * 60
    url = "https://api.whale-alert.io/v1/transactions"
    params = {
        "api_key": api_key,
        "start": start,
        "end": end,
        "min_value": usd_min,  # USD threshold
        "cursor": "",
    }

    results: List[WhaleTx] = []
    while True:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            break
        data = resp.json()
        txs = data.get("transactions", []) or []
        for tx in txs:
            # Whale Alert schema (simplified best-effort)
            symbol = (tx.get("symbol") or tx.get("currency") or "").upper()
            if assets and symbol not in [a.upper() for a in assets]:
                continue
            results.append(
                WhaleTx(
                    blockchain=tx.get("blockchain", ""),
                    symbol=symbol,
                    hash=tx.get("hash", ""),
                    transaction_type=tx.get("transaction_type", ""),
                    amount=float(tx.get("amount") or 0),
                    amount_usd=float(tx.get("amount_usd") or 0) if tx.get("amount_usd") else None,
                    from_address=(tx.get("from", {}) or {}).get("address"),
                    to_address=(tx.get("to", {}) or {}).get("address"),
                    timestamp=int(tx.get("timestamp") or start),
                )
            )
        cursor = data.get("cursor")
        if not cursor:
            break
        params["cursor"] = cursor  # paginate if available
        # safety: limit pages
        if len(results) > 500:
            break
    return results


def etherscan_enrich(api_key: Optional[str], txs: List[WhaleTx]) -> None:
    """
    Best-effort enrichment. Without paid labels, we heuristically check if destination is a contract
    (often exchanges use deposit contracts) to hint inflow/outflow.
    """
    if not api_key:
        return

    base = "https://api.etherscan.io/api"
    for tx in txs:
        if tx.symbol != "ETH":
            continue
        # Check if 'to' is a contract (proxy for exchange/contract address)
        if tx.to_address:
            try:
                r = requests.get(base, params={
                    "module": "contract",
                    "action": "getabi",
                    "address": tx.to_address,
                    "apikey": api_key
                }, timeout=10)
                # If ABI exists (not "Contract source code not verified"), it's a contract
                is_contract = r.status_code == 200 and "Contract ABI" in r.text
                if is_contract:
                    tx.is_exchange_inflow = True
            except Exception:
                pass

        # If from is a contract, consider outflow
        if tx.from_address:
            try:
                r = requests.get(base, params={
                    "module": "contract",
                    "action": "getabi",
                    "address": tx.from_address,
                    "apikey": api_key
                }, timeout=10)
                is_contract = r.status_code == 200 and "Contract ABI" in r.text
                if is_contract:
                    tx.is_exchange_outflow = True
            except Exception:
                pass


# ------------------------------
# LLM Prompting
# ------------------------------

FACT_CHECK_SYSTEM = """You are an AI fact-checking assistant that analyzes Twitter/X posts related to news, politics, and factors affecting the cryptocurrency world. 
Evaluate the factual accuracy of a given post in real time, cross-checking against credible sources, and classify it into one of three categories:

- Likely True — Supported by credible news outlets (BBC, CNN, CBN, New York Times, Fox News, etc.) or reliable scientific journals (post-2000).
- Likely False — Not supported by any credible sources, disproven by fact-check sites (Snopes, PolitiFact, FactCheck.org, etc.), or misleading/technically incorrect.
- Opinion — Subjective statements, personal viewpoints, satire, rhetorical questions, or political commentary that cannot be fact-verified.

Rules & Guidelines
1) Sources: Use credible news outlets and scientific journals (2000+) as primary references. Consider trusted fact-checking sites (Snopes, PolitiFact, FactCheck.org, etc.). Do not use Wikipedia.
2) Analysis: Perform a fresh fact check every time (no cached results). Look at both the text and attachments (if provided).
3) Output: One of the three labels, a short justification, confidence score (1–10), and citations/links.
4) Special cases: Misleading-but-technically-true -> Likely False with explanation. Treat controversial/political topics normally.
"""

FUD_WHALE_GUIDE = """Run FUD/FOMO detection in parallel with fact-checking. 
Definitions:
- FUD (Fear, Uncertainty, Doubt): negative or alarmist claims without credible evidence.
- FOMO (Fear Of Missing Out): hype/guaranteed-return style claims without credible evidence.
Whale Correlation:
- Use on-chain whale context (below) to comment on potential sell/buy pressure.
- Fixed threshold: highlight any BTC transactions >= 500 BTC (approx. USD filtered upstream).

Guardrails:
- Neutral, analytical tone. Do NOT give trading advice. Do not encourage panic buying/selling.
- If a claim is verified true by credible sources, do NOT call it FUD/FOMO even if emotional.

Additional Whale Reporting Instructions:
- Always analyze the whale transactions provided and relate them to the news post.
- If whales are moving funds *into exchanges*, comment that this may indicate potential sell pressure.
- If whales are moving funds *out of exchanges or to unknown wallets*, comment that this may indicate accumulation or bullish behavior.
- If no relevant whale data exists, explicitly say 'No significant whale movements related to this news.'
- Provide the whale implication in a short, clear sentence.
"""

USER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", FACT_CHECK_SYSTEM + "\n\n" + FUD_WHALE_GUIDE),
        ("human", 
         "Post to analyze:\n{post_text}\n\n"
         "On-chain whale context (recent, filtered):\n{whale_context}\n\n"
         "Tasks:\n"
         "1) Classify the post: Likely True / Likely False / Opinion.\n"
         "2) Indicate FUD or FOMO if applicable, else 'No FUD/FOMO'.\n"
         "3) Summarize whale implications in one line (bearish/bullish/neutral or none).\n"
         "4) Provide confidence (1–10) and concise citations/links.\n"
        )
    ]
)

def run_llm(post_text: str, whale_context_lines: List[str]) -> str:
    api_key = "AIzaSyA5ww0YfMCczNCGacLRvMvbiVCq7jZJa_w"
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    chain = USER_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "post_text": post_text,
        "whale_context": "\n".join(whale_context_lines) if whale_context_lines else "None found."
    })


# ------------------------------
# Orchestrator
# ------------------------------

def analyze_post(post_text: str):
    load_dotenv()

    whale_key = os.getenv("WHALE_ALERT_API_KEY") or "PYE3bszI8U7mYuYx2LpKPNaS2Mo8Dg2P"
    etherscan_key = os.getenv("ETHERSCAN_API_KEY")
    if not whale_key:
        raise RuntimeError("WHALE_ALERT_API_KEY is required")

    assets = infer_assets(post_text)
    usd_min = int(os.getenv("WHALE_ALERT_USD_MIN", "15000000"))

    whale_txs = whale_alert_fetch(whale_key, assets=assets, usd_min=usd_min, lookback_minutes=120)
    etherscan_enrich(etherscan_key, whale_txs)

    whale_lines = [tx.summary() for tx in whale_txs[:20]]

    llm_output = run_llm(post_text, whale_lines)

    classification = "Unknown"
    confidence = 5
    sources: List[str] = []
    fud_fomo = "No FUD/FOMO"
    whale_summary = "No significant whale movements found."
    try:
        low = llm_output.lower()
        if "likely true" in low:
            classification = "Likely True"
        elif "likely false" in low:
            classification = "Likely False"
        elif "opinion" in low:
            classification = "Opinion"

        if "fud" in low:
            fud_fomo = "⚠️ FUD detected"
        elif "fomo" in low:
            fud_fomo = "⚠️ FOMO detected"

        import re
        m = re.search(r"confidence\s*:\s*(\d{1,2})\s*/\s*10", llm_output, re.IGNORECASE)
        if m:
            confidence = max(1, min(10, int(m.group(1))))

        # crude link extraction
        sources = re.findall(r"https?://\S+", llm_output)

        # Whale implication extraction
        m2 = re.search(r"(whale implication|whale context|whale impact).*?:\s*(.*)", llm_output, re.IGNORECASE)
        if m2:
            whale_summary = m2.group(2).strip()
    except Exception:
        pass

    return AnalysisResult(
        classification=classification,
        confidence=confidence,
        sources=sources,
        whale_context=[],  # keep hidden
        raw_llm_output=llm_output,
    ), fud_fomo, whale_summary
# ------------------------------
# CLI
# ------------------------------

def format_result(res: AnalysisResult, fud_fomo: str, whale_summary: str) -> str:
    lines = []
    lines.append("🔎 Crypto News Analysis Report")
    lines.append("=" * 40)
    lines.append(f"📌 Classification : {res.classification}")
    lines.append(f"📊 Confidence     : {res.confidence}/10")
    lines.append(f"⚠️ Sentiment      : {fud_fomo}")
    lines.append(f"🐋 Whale Activity : {whale_summary}")
    lines.append("")
    lines.append("📝 Explanation")
    lines.append(textwrap.indent(res.raw_llm_output.strip(), "  "))
    lines.append("")
    if res.sources:
        lines.append("🔗 Sources:")
        for src in res.sources:
            lines.append(f"  - {src}")
    else:
        lines.append("🔗 Sources: None found")
    return "\n".join(lines)


def main(post_text: str=None):
    res, fud_fomo, whale_summary = analyze_post(post_text=post_text)
    return(format_result(res, fud_fomo, whale_summary))


if __name__ == "__main__":
    main()
"""
Research scanner — discovers new trading strategies from academic papers.

Pipeline:
  1. Pull arXiv q-fin RSS feed (and optionally SSRN/Quantpedia)
  2. Filter for relevant papers (trading, alpha, regime, portfolio)
  3. Send to Claude/Gemini via OpenRouter for strategy extraction
  4. Score and track in paper_log.json
  5. Alert via Telegram when promising strategies are found
"""

import re
import hashlib
import requests
import feedparser
from datetime import datetime

from config.settings import cfg
from execution.telegram_bot import send_research_alert
from utils.logger import log
from utils import load_json, save_json


# ── Sources ───────────────────────────────────────────────────
ARXIV_FEEDS = [
    "https://rss.arxiv.org/rss/q-fin.PM",   # Portfolio Management
    "https://rss.arxiv.org/rss/q-fin.TR",   # Trading and Market Microstructure
    "https://rss.arxiv.org/rss/q-fin.CP",   # Computational Finance
    "https://rss.arxiv.org/rss/q-fin.ST",   # Statistical Finance
]

# Targeted search queries for high-value strategy papers
SEARCH_QUERIES = [
    "cat:q-fin.PM OR cat:q-fin.TR OR cat:q-fin.CP OR cat:q-fin.ST",
    "all:trading strategy AND all:regime",
    "all:alpha factor AND all:quantitative",
    "all:mean reversion AND all:momentum AND all:portfolio",
    "all:sentiment AND all:trading AND all:stock",
]

# Keywords that signal a relevant paper
RELEVANCE_KEYWORDS = [
    "trading strategy", "alpha", "regime", "portfolio optimization",
    "momentum", "mean reversion", "volatility", "backtesting",
    "technical analysis", "risk management", "factor model",
    "reinforcement learning trading", "sentiment trading",
    "market regime", "adaptive strategy", "position sizing",
    "sharpe ratio", "drawdown", "quantitative trading",
]


def scan_arxiv() -> list[dict]:
    """
    Scan arXiv for relevant papers using two methods:
    1. RSS feeds for today's new submissions
    2. Search API for recent high-value papers (works any day)
    
    Tracks all seen papers in paper_log.json to avoid duplicates.
    """
    papers = []
    seen = _load_seen_papers()

    # Method 1: RSS feeds (daily new submissions)
    for feed_url in ARXIV_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                _process_entry(entry, papers, seen)
        except Exception as e:
            log.warning(f"Failed to parse feed {feed_url}: {e}")

    # Method 2: Search API (recent papers, works any day)
    for query in SEARCH_QUERIES:
        try:
            resp = requests.get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": query,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": 20,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                feed = feedparser.parse(resp.text)
                for entry in feed.entries:
                    _process_entry(entry, papers, seen)
        except Exception as e:
            log.warning(f"arXiv search failed for query: {e}")

    _save_seen_papers(seen)
    log.info(f"arXiv scan complete: {len(papers)} new relevant papers found")
    return papers


def _process_entry(entry, papers: list, seen: set):
    """Process a single arXiv entry and add to papers if relevant and new."""
    paper_id = _paper_hash(entry.get("id", entry.get("link", "")))

    if paper_id in seen:
        return

    title = entry.get("title", "").replace("\n", " ").strip()
    abstract = entry.get("summary", "").replace("\n", " ").strip()
    url = entry.get("link", "")

    if _is_relevant(title, abstract):
        papers.append({
            "id": paper_id,
            "title": title,
            "abstract": abstract[:2000],
            "url": url,
            "source": "arXiv",
            "date": entry.get("published", datetime.now().isoformat()),
            "discovered_at": datetime.now().isoformat(),
        })
        seen.add(paper_id)


def _is_relevant(title: str, abstract: str) -> bool:
    """Check if a paper is relevant to our trading engine."""
    text = (title + " " + abstract).lower()
    matches = sum(1 for kw in RELEVANCE_KEYWORDS if kw in text)
    return matches >= 2


def _paper_hash(identifier: str) -> str:
    """Generate a short hash for deduplication."""
    return hashlib.md5(identifier.encode()).hexdigest()[:12]


def _load_seen_papers() -> set:
    """Load set of already-seen paper IDs."""
    data = load_json(cfg.PAPER_LOG, default={"seen": [], "papers": []})
    return set(data.get("seen", []))


def _save_seen_papers(seen: set):
    """Save seen paper IDs."""
    data = load_json(cfg.PAPER_LOG, default={"seen": [], "papers": []})
    data["seen"] = list(seen)
    save_json(cfg.PAPER_LOG, data)


def evaluate_paper(paper: dict) -> dict:
    """
    Send a paper to Claude/Gemini via OpenRouter for strategy extraction.
    Returns paper dict with added fields: strategy, sharpe_estimate, applicability.
    """
    if not cfg.OPENROUTER_API_KEY:
        log.warning("OpenRouter not configured, skipping paper evaluation")
        paper["evaluation"] = "skipped — no API key"
        return paper

    prompt = f"""Analyze this quantitative finance research paper for trading strategy extraction.

Title: {paper['title']}

Abstract: {paper['abstract']}

Please respond in this exact JSON format (nothing else):
{{
  "strategy_summary": "One paragraph describing the trading strategy",
  "trading_rules": "Specific buy/sell rules that can be coded",
  "data_requirements": "What market data is needed (OHLCV, sentiment, etc.)",
  "reported_sharpe": "Reported Sharpe ratio or 'not reported'",
  "max_drawdown": "Reported max drawdown or 'not reported'",
  "applicable_to_nse": true/false,
  "applicability_notes": "How this could work on Indian NSE/BSE markets",
  "complexity": "low/medium/high",
  "priority_score": 1-10
}}"""

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {cfg.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg.OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
            },
            timeout=30,
        )

        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            # Try to parse JSON from response
            try:
                # Strip markdown code fences if present
                clean = re.sub(r"```json\s*|```\s*", "", content).strip()
                evaluation = __import__("json").loads(clean)
                paper["evaluation"] = evaluation
                paper["priority_score"] = evaluation.get("priority_score", 0)
                log.info(f"Paper evaluated: {paper['title'][:60]}... Score: {paper['priority_score']}")
            except Exception:
                paper["evaluation"] = content
                paper["priority_score"] = 0
        else:
            log.warning(f"OpenRouter API error: {resp.status_code}")
            paper["evaluation"] = f"API error: {resp.status_code}"
            paper["priority_score"] = 0

    except Exception as e:
        log.error(f"Paper evaluation failed: {e}")
        paper["evaluation"] = str(e)
        paper["priority_score"] = 0

    return paper


def run_research_scan() -> list[dict]:
    """
    Full research pipeline: scan → evaluate → log → alert.
    Returns list of evaluated papers.
    """
    log.info("Starting research scan...")

    # Discover new papers
    new_papers = scan_arxiv()

    if not new_papers:
        log.info("No new relevant papers found")
        return []

    # Evaluate each paper
    evaluated = []
    for paper in new_papers:
        paper = evaluate_paper(paper)
        evaluated.append(paper)

    # Save to paper log
    paper_log = load_json(cfg.PAPER_LOG, default={"seen": [], "papers": []})
    paper_log["papers"].extend(evaluated)
    save_json(cfg.PAPER_LOG, paper_log)

    # Alert on high-priority papers
    for paper in evaluated:
        score = paper.get("priority_score", 0)
        if score >= 7:
            eval_data = paper.get("evaluation", {})
            alert = {
                "title": paper["title"],
                "source": paper["source"],
                "url": paper["url"],
                "sharpe": eval_data.get("reported_sharpe", "N/A") if isinstance(eval_data, dict) else "N/A",
                "insight": eval_data.get("strategy_summary", "")[:200] if isinstance(eval_data, dict) else str(eval_data)[:200],
            }
            send_research_alert(alert)

    log.info(f"Research scan complete: {len(evaluated)} papers evaluated, "
             f"{sum(1 for p in evaluated if p.get('priority_score', 0) >= 7)} high-priority")
    return evaluated

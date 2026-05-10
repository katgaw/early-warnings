"""
===========================================================
GOOGLE NEWS RSS SEARCH TOOL
===========================================================

This script creates a LangChain tool that searches recent
Google News articles related to financial and geopolitical risk.

No paid API is required.

-----------------------------------------------------------
PURPOSE
-----------------------------------------------------------

The agent can call this tool to search recent news about:

1. US military escalation
2. Oil price spikes
3. Bank liquidity crises
4. Cyberattacks on banks

-----------------------------------------------------------
REQUIREMENT
-----------------------------------------------------------

Install feedparser:

    pip install feedparser

===========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

from urllib.parse import quote_plus

import feedparser
from langchain_core.tools import tool


# =========================================================
# ALLOWED NEWS SEARCH KEYWORDS
# =========================================================

RISK_NEWS_KEYWORDS = (
    "US military escalation",
    "oil price spike",
    "bank liquidity crisis",
    "cyberattack on banks",
)

KEYWORD_ALIASES = {
    keyword.lower().strip(): keyword
    for keyword in RISK_NEWS_KEYWORDS
}


# =========================================================
# HELPER FUNCTION:
# VALIDATE KEYWORD
# =========================================================

def normalize_keyword(keyword: str) -> str | None:
    """
    Map user input to one approved search phrase.

    Matching is case-insensitive.

    Example:
        "Oil Price Spike" -> "oil price spike"
    """

    return KEYWORD_ALIASES.get(
        keyword.lower().strip()
    )


# =========================================================
# HELPER FUNCTION:
# BUILD GOOGLE NEWS RSS URL
# =========================================================

def build_google_news_rss_url(keyword: str) -> str:
    """
    Build a Google News RSS search URL.

    The query searches recent Google News results.
    """

    encoded_query = quote_plus(keyword)

    return (
        "https://news.google.com/rss/search?"
        f"q={encoded_query}"
        "&hl=en-US"
        "&gl=US"
        "&ceid=US:en"
    )


# =========================================================
# HELPER FUNCTION:
# CLEAN ARTICLE TEXT
# =========================================================

def clean_text(text: str | None, max_chars: int = 1200) -> str:
    """
    Clean and shorten article summaries.
    """

    if not text:
        return ""

    cleaned = " ".join(text.split())

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "..."

    return cleaned


# =========================================================
# LANGCHAIN TOOL:
# SEARCH RECENT RISK NEWS
# =========================================================

@tool
def search_risk_news_articles(keyword: str) -> str:
    """
    Search Google News RSS for recent macro-risk news.

    Use this tool when the agent needs fresh news about:
    - geopolitical escalation
    - oil market shocks
    - bank liquidity stress
    - cyberattacks on banks

    Args:
        keyword:
            Must be one of:
            - US military escalation
            - oil price spike
            - bank liquidity crisis
            - cyberattack on banks

    Returns:
        A formatted list of recent news articles.
    """

    # -----------------------------------------------------
    # STEP 1:
    # VALIDATE KEYWORD
    # -----------------------------------------------------

    normalized_keyword = normalize_keyword(keyword)

    if not normalized_keyword:

        allowed_keywords = ", ".join(
            f'"{item}"'
            for item in RISK_NEWS_KEYWORDS
        )

        return (
            f"Unknown keyword: {keyword!r}\n\n"
            f"Please use exactly one of:\n"
            f"{allowed_keywords}"
        )

    # -----------------------------------------------------
    # STEP 2:
    # BUILD GOOGLE NEWS RSS URL
    # -----------------------------------------------------

    rss_url = build_google_news_rss_url(
        normalized_keyword
    )

    # -----------------------------------------------------
    # STEP 3:
    # FETCH GOOGLE NEWS RSS RESULTS
    # -----------------------------------------------------

    feed = feedparser.parse(rss_url)

    articles = feed.entries[:8]

    if not articles:

        return (
            "No recent Google News results returned for: "
            f"{normalized_keyword}"
        )

    # -----------------------------------------------------
    # STEP 4:
    # FORMAT RESULTS FOR THE AGENT
    # -----------------------------------------------------

    lines = [
        f"Recent Google News results for "
        f"'{normalized_keyword}' "
        f"({len(articles)} results):\n"
    ]

    for index, article in enumerate(articles, start=1):

        title = article.get("title", "(no title)")
        url = article.get("link", "")
        published_date = article.get("published", "")
        summary = clean_text(
            article.get("summary", "")
        )

        date_text = (
            f" — {published_date}"
            if published_date
            else ""
        )

        lines.append(
            f"{index}. {title}{date_text}\n"
            f"   URL: {url}\n"
            f"   Summary: {summary}\n"
        )

    return "\n".join(lines)


# =========================================================
# OPTIONAL TEST
# =========================================================

if __name__ == "__main__":

    print("\nGoogle News RSS Search Tool Ready\n")

    print("Tool name:")
    print(search_risk_news_articles.name)

    print("\nAllowed keywords:")
    for keyword in RISK_NEWS_KEYWORDS:
        print(f"- {keyword}")

    print("\nTest search:")
    print(
        search_risk_news_articles.invoke(
            {"keyword": "oil price spike"}
        )
    )
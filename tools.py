"""
CONCEPT: Tools
--------------
Three tools bound to the LLM:

1. search_web         - scrapes DuckDuckGo's HTML results page. No API key,
                         no rate-limit headaches — good for a learning project.
2. scrape_page         - fetches a real URL (e.g. one returned by search_web)
                         and pulls out readable text. This is the literal
                         "web scraping" piece.
3. propose_preference  - a *deliberately inert* tool. The LLM calls it when
                         it detects the user stating a food like/dislike, but
                         its Python body never actually runs — the graph
                         intercepts this specific tool call and routes to a
                         human-in-the-loop review node instead (see graph.py).
                         This is a common real pattern: give the model a
                         tool whose *call* is the signal, and let a graph
                         node decide what to do with it.
"""
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

HEADERS = {"User-Agent": "Mozilla/5.0 (LangGraph learning project)"}


@tool
def search_web(query: str) -> str:
    """Search the web (DuckDuckGo) for healthy food info, recipes, or
    nutrition facts. Returns a short list of titles, snippets and URLs.
    Use this before scrape_page when you don't have a URL yet."""
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Search failed: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for block in soup.select(".result__body")[:5]:
        title_tag = block.select_one(".result__a")
        snippet_tag = block.select_one(".result__snippet")
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)
        url = title_tag.get("href", "")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        results.append(f"- {title}\n  {snippet}\n  {url}")

    if not results:
        return f"No results found for '{query}'."
    return "\n\n".join(results)


@tool
def scrape_page(url: str) -> str:
    """Fetch a web page by URL and extract its main readable text (headings
    and paragraphs). Use this to read a recipe or article that search_web
    surfaced, e.g. to pull out ingredients or instructions."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Could not fetch {url}: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    chunks = [
        el.get_text(" ", strip=True)
        for el in soup.find_all(["h1", "h2", "h3", "li", "p"])
    ]
    text = "\n".join(c for c in chunks if len(c) > 20)
    return text[:3000] if text else "No readable content found on that page."


@tool
def propose_preference(food: str, sentiment: str) -> str:
    """Call this when the user clearly states they LIKE or DISLIKE a
    specific food or ingredient, so it can be remembered for next time.
    sentiment must be exactly 'likes' or 'dislikes'.
    IMPORTANT: this requires human approval before anything is saved —
    do not call any other tool in the same turn as this one."""
    # Never actually executed: graph.py routes this call to a
    # human-in-the-loop node instead of the normal tool executor.
    return "pending human review"


SEARCH_TOOLS = [search_web, scrape_page]
ALL_TOOLS = [search_web, scrape_page, propose_preference]

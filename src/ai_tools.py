from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_core.tools import tool
from datetime import datetime
import requests
from bs4 import BeautifulSoup

ALLOW_DANGEROUS_REQUEST = True

toolkit = RequestsToolkit(
    requests_wrapper=TextRequestsWrapper(headers={}),
    allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
)


@tool
def duckduckgo_search(query: str) -> str:
    """Searches the web via DuckDuckGo and returns top results as text."""
    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": "Mozilla/5.0"}
    )
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for a in soup.select(".result__a")[:5]:
        title = a.get_text()
        href = a["href"]
        snippet_tag = a.find_next_sibling("a")
        snippet = snippet_tag.get_text() if snippet_tag else ""
        results.append(f"- {title}: {href}\\n  » {snippet}")
    return "\\n".join(results)


@tool
def get_current_datetime():
    """Returns the current date and time."""
    now = datetime.now()
    print("Current date and time:", now)
    return now.strftime("%d-%m-%Y %H:%M:%S")


tools = [duckduckgo_search, get_current_datetime] + toolkit.get_tools()

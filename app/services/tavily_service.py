"""Optional Tavily-backed live research connector.

Tavily is used here as a discovery layer only. The core engines still decide
relevance, readiness, and risk.
"""

import os
from typing import Any


class TavilyService:
    def __init__(self) -> None:
        self.api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured.")

        if self._client is not None:
            return self._client

        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise RuntimeError(
                "tavily-python is not installed. Add it to requirements and install dependencies."
            ) from exc

        self._client = TavilyClient(api_key=self.api_key)
        return self._client

    def _normalize_results(self, raw: dict[str, Any]) -> dict[str, Any]:
        results = []
        for item in raw.get("results", [])[:10]:
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
                "source": item.get("source"),
            })

        return {
            "answer": raw.get("answer"),
            "results": results,
        }

    def search_opportunities(self, country: str, skill: str, max_results: int = 5) -> dict[str, Any]:
        client = self._get_client()
        query = f"{skill} training apprenticeship job opportunities {country}".strip()
        raw = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_raw_content=False,
        )
        return {
            "query": query,
            "country": country,
            "skill": skill,
            **self._normalize_results(raw or {}),
        }

    def search_context(self, country: str, topic: str, max_results: int = 5) -> dict[str, Any]:
        client = self._get_client()
        query = f"{country} {topic} World Bank ILOSTAT labor market".strip()
        raw = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_raw_content=False,
        )
        return {
            "query": query,
            "country": country,
            "topic": topic,
            **self._normalize_results(raw or {}),
        }

    def extract_url(self, url: str) -> dict[str, Any]:
        client = self._get_client()
        raw = client.extract(urls=[url], extract_depth="basic")
        return {"url": url, "data": raw}


from __future__ import annotations

from typing import Any, Callable

from commoncrawl_search import search_commoncrawl_images
from laion_search import search_laion_images
from open_images_search import search_open_images


SEARCH_FUNCTIONS: dict[str, Callable[[str], list[dict[str, Any]]]] = {
    "laion": search_laion_images,
    "open_images": search_open_images,
    "common_crawl": search_commoncrawl_images,
}


def retrieve_candidates(query: str, source_order: list[str]) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for source in source_order:
        search_fn = SEARCH_FUNCTIONS.get(source)
        if search_fn is None:
            continue
        try:
            results = search_fn(query)
        except Exception:
            results = []
        for item in results:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            normalized.setdefault("source", source)
            normalized["query_used"] = query
            combined.append(normalized)
        if combined:
            break
    return combined

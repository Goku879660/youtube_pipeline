"""
Utilities for querying the public LAION KNN search service.

The main public entrypoints are:
- generate_query_variations(scene_description)
- search_laion_images(query)
- search_scene_candidates(scene_description)
"""

from __future__ import annotations

import re
import time
from typing import Any

import requests

LAION_KNN_URL = "https://knn.laion.ai/knn-service"
LAION_METADATA_URL = "https://knn.laion.ai/metadata"
DEFAULT_INDEX = "laion5B-L-14"
DEFAULT_RESULT_COUNT = 10
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_MAX_RETRIES = 2
HTTP_SESSION = requests.Session()


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clean_caption(value: Any) -> str:
    if value is None:
        return ""
    return _normalize_whitespace(str(value))


def _score_from_item(item: dict[str, Any]) -> float | None:
    for key in ("similarity", "score"):
        value = item.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _image_url_from_item(item: dict[str, Any]) -> str:
    for key in ("url", "image_url", "image"):
        value = item.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return ""


def _caption_from_item(item: dict[str, Any]) -> str:
    for key in ("caption", "text"):
        value = item.get(key)
        if value:
            return _clean_caption(value)
    return ""


def _normalize_result(item: dict[str, Any]) -> dict[str, Any] | None:
    image_url = _image_url_from_item(item)
    if not image_url:
        return None
    return {
        "image_url": image_url,
        "caption": _caption_from_item(item),
        "score": _score_from_item(item),
    }


def _dedupe_results(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in items:
        image_url = item.get("image_url", "")
        if not image_url:
            continue
        current_score = item.get("score")
        previous = deduped.get(image_url)
        if previous is None:
            deduped[image_url] = item
            continue
        previous_score = previous.get("score")
        if previous_score is None and current_score is not None:
            deduped[image_url] = item
        elif (
            previous_score is not None
            and current_score is not None
            and current_score > previous_score
        ):
            deduped[image_url] = item
    return sorted(
        deduped.values(),
        key=lambda item: item.get("score") if item.get("score") is not None else float("-inf"),
        reverse=True,
    )


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> Any:
    response = HTTP_SESSION.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _extract_metadata_ids(raw_results: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for item in raw_results:
        value = item.get("id")
        if isinstance(value, int):
            ids.append(value)
    return ids


def _fetch_metadata(ids: list[int], timeout: int) -> list[dict[str, Any]]:
    if not ids:
        return []
    payload = {
        "indice_name": DEFAULT_INDEX,
        "ids": ids[:DEFAULT_RESULT_COUNT],
    }
    data = _post_json(LAION_METADATA_URL, payload, timeout)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        results = data.get("results") or data.get("items") or data.get("metadata")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
    return []


def _request_laion_results(query: str, num_results: int, timeout: int) -> list[dict[str, Any]]:
    payload = {
        "text": query,
        "modality": "image",
        "num_images": num_results,
        "num_result_ids": num_results,
        "indice_name": DEFAULT_INDEX,
        "deduplicate": True,
        "use_safety_model": True,
        "use_violence_detector": True,
    }
    data = _post_json(LAION_KNN_URL, payload, timeout)

    if isinstance(data, list):
        raw_results = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict):
        raw_results = data.get("results") or data.get("images") or data.get("items") or []
        if not isinstance(raw_results, list):
            raw_results = []
        raw_results = [item for item in raw_results if isinstance(item, dict)]
    else:
        raw_results = []

    if any(_image_url_from_item(item) for item in raw_results):
        return raw_results[:num_results]

    metadata_items = _fetch_metadata(_extract_metadata_ids(raw_results), timeout)
    if metadata_items:
        return metadata_items[:num_results]

    return raw_results[:num_results]


def search_laion_images(query: str) -> list[dict[str, Any]]:
    """
    Search the public LAION KNN service and return up to 10 candidate images.

    Returns:
        [
          {
            "image_url": "...",
            "caption": "...",
            "score": 0.87,
          }
        ]
    """
    normalized_query = _normalize_whitespace(query)
    if not normalized_query:
        return []

    for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
        try:
            raw_results = _request_laion_results(
                query=normalized_query,
                num_results=DEFAULT_RESULT_COUNT,
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
            normalized_results = [
                normalized
                for normalized in (_normalize_result(item) for item in raw_results)
                if normalized is not None
            ]
            return _dedupe_results(normalized_results)[:DEFAULT_RESULT_COUNT]
        except (requests.RequestException, ValueError):
            pass

        if attempt < DEFAULT_MAX_RETRIES:
            time.sleep(1.5 * attempt)

    return []


def generate_query_variations(scene_description: str) -> list[str]:
    """
    Generate three practical search queries from a scene description.
    """
    description = _normalize_whitespace(scene_description)
    if not description:
        return []

    variations: list[str] = []

    year_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", description)
    year = year_match.group(1) if year_match else ""

    proper_nouns = re.findall(r"\b[A-Z][a-zA-Z'-]+\b", description)
    proper_phrase = " ".join(proper_nouns[:3]).strip()

    subject_keywords = re.findall(r"\b[a-zA-Z][a-zA-Z'-]{2,}\b", description)
    lower_keywords = [word.lower() for word in subject_keywords]

    historical_terms = {
        "factory", "battle", "temple", "palace", "portrait", "ship", "train",
        "aircraft", "street", "city", "map", "manuscript", "statue", "artifact",
        "coronation", "ceremony", "army", "market", "castle", "village",
    }
    matched_terms = [term for term in lower_keywords if term in historical_terms]

    if proper_phrase:
        variations.append(f"{proper_phrase} portrait")

    core_terms = []
    for word in subject_keywords:
        lowered = word.lower()
        if lowered in {"the", "and", "with", "from", "into", "during", "after", "before"}:
            continue
        core_terms.append(word)
    if core_terms:
        core_query = " ".join(core_terms[:5])
        if year:
            core_query = f"{core_query} {year}"
        variations.append(core_query)

    if matched_terms:
        context_phrase = " ".join(dict.fromkeys(matched_terms).keys())
        if year:
            variations.append(f"{context_phrase} {year}")
        else:
            variations.append(context_phrase)

    if len(variations) < 3:
        trimmed = description
        if year and year not in trimmed:
            trimmed = f"{trimmed} {year}"
        variations.append(trimmed)

    if len(variations) < 3 and proper_phrase:
        variations.append(f"{proper_phrase} historical photo")

    if len(variations) < 3:
        variations.append(f"{description} documentary")

    unique_variations: list[str] = []
    seen = set()
    for variation in variations:
        cleaned = _normalize_whitespace(variation)
        lowered = cleaned.lower()
        if not cleaned or lowered in seen:
            continue
        seen.add(lowered)
        unique_variations.append(cleaned)
        if len(unique_variations) == 3:
            break

    while len(unique_variations) < 3:
        fallback = f"{description} reference image {len(unique_variations) + 1}"
        unique_variations.append(fallback)

    return unique_variations


def search_scene_candidates(scene_description: str) -> list[dict[str, Any]]:
    """
    Full workflow:
    scene_description -> generate_query_variations -> search_laion_images
    -> combined candidate images
    """
    combined_results: list[dict[str, Any]] = []
    for query in generate_query_variations(scene_description):
        combined_results.extend(search_laion_images(query))
    return _dedupe_results(combined_results)[: DEFAULT_RESULT_COUNT * 3]

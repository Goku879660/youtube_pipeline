from __future__ import annotations

import re

from laion_search import generate_query_variations


def build_queries(scene_analysis: dict, fallback_terms: list[str] | None = None) -> list[str]:
    scene_text = scene_analysis.get("scene_text", "")
    subject = scene_analysis.get("subject", "").strip()
    objects = scene_analysis.get("objects", [])
    entities = scene_analysis.get("entities", [])
    keywords = scene_analysis.get("keywords", [])
    years = scene_analysis.get("years", [])
    fallback_terms = fallback_terms or []

    queries: list[str] = []
    for term in fallback_terms[:4]:
        if term:
            queries.append(term)

    if subject:
        queries.append(f"{subject} portrait")
        queries.append(subject)
    if subject and objects:
        queries.append(f"{subject} {' '.join(objects[:2])}")
    if subject and years:
        queries.append(f"{subject} {years[0]}")
    elif objects:
        queries.append(" ".join(objects[:3]))

    if entities and objects:
        queries.append(f"{entities[0]} {' '.join(objects[:2])}")
    if entities and years:
        queries.append(f"{entities[0]} {years[0]}")
    if keywords:
        queries.append(" ".join(keywords[:4]))
    if keywords and years:
        queries.append(f"{' '.join(keywords[:3])} {years[0]}")

    for variation in generate_query_variations(scene_text):
        queries.append(variation)

    year_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", scene_text)
    if year_match and subject:
        queries.append(f"{subject} {year_match.group(1)}")

    unique: list[str] = []
    seen = set()
    for query in queries:
        cleaned = re.sub(r"\s+", " ", str(query)).strip()
        lowered = cleaned.lower()
        if not cleaned or lowered in seen:
            continue
        seen.add(lowered)
        unique.append(cleaned)
        if len(unique) >= 5:
            break
    return unique[:5]


def expand_retry_queries(scene_analysis: dict, failed_queries: list[str]) -> list[str]:
    base = scene_analysis.get("subject") or scene_analysis.get("scene_text", "")
    objects = " ".join(scene_analysis.get("objects", [])[:2])
    entities = " ".join(scene_analysis.get("entities", [])[:2])
    years = scene_analysis.get("years", [])
    alternatives = [
        f"{base} archival photo",
        f"{base} documentary image",
        f"{entities} historical photo" if entities else "",
        f"{base} {years[0]} archive" if years else "",
        f"{objects} historical photo" if objects else "",
        scene_analysis.get("scene_text", ""),
    ]
    unique: list[str] = []
    failed = {query.lower() for query in failed_queries}
    for query in alternatives:
        cleaned = re.sub(r"\s+", " ", str(query)).strip()
        lowered = cleaned.lower()
        if not cleaned or lowered in failed:
            continue
        unique.append(cleaned)
        if len(unique) >= 3:
            break
    return unique

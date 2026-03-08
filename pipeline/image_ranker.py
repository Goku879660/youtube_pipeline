from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any


MIN_CONFIDENCE = 0.72
GENERIC_PENALTY_TERMS = {
    "illustration", "wallpaper", "background", "abstract", "design", "icon",
    "template", "graphic", "pattern", "mockup",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _keyword_overlap(scene_text: str, candidate_text: str) -> float:
    scene_tokens = set(_tokenize(scene_text))
    candidate_tokens = set(_tokenize(candidate_text))
    if not scene_tokens or not candidate_tokens:
        return 0.0
    overlap = len(scene_tokens & candidate_tokens)
    return overlap / max(1, len(candidate_tokens))


def _semantic_proxy(scene_text: str, candidate_text: str) -> float:
    ratio = SequenceMatcher(None, scene_text.lower(), candidate_text.lower()).ratio()
    overlap = _keyword_overlap(scene_text, candidate_text)
    return min(1.0, (ratio * 0.45) + (overlap * 0.55))


def _contains_phrase(candidate_text: str, phrases: list[str]) -> float:
    lowered = candidate_text.lower()
    matches = 0
    for phrase in phrases:
        cleaned = str(phrase).strip().lower()
        if cleaned and cleaned in lowered:
            matches += 1
    return min(1.0, matches / max(1, len(phrases)))


def _year_match(candidate_text: str, years: list[str]) -> float:
    lowered = candidate_text.lower()
    for year in years:
        if str(year).lower() in lowered:
            return 1.0
    return 0.0


def _generic_penalty(candidate_text: str) -> float:
    lowered = candidate_text.lower()
    hits = sum(1 for term in GENERIC_PENALTY_TERMS if term in lowered)
    return min(0.25, hits * 0.08)


def _metadata_text(candidate: dict[str, Any]) -> str:
    return str(candidate.get("caption") or candidate.get("label") or "").strip()


def _query_text(candidate: dict[str, Any]) -> str:
    return str(candidate.get("query_used") or "").strip()


def _contains_any_phrase(candidate_text: str, phrases: list[str]) -> bool:
    lowered = candidate_text.lower()
    for phrase in phrases:
        cleaned = str(phrase).strip().lower()
        if cleaned and cleaned in lowered:
            return True
    return False


def _has_context_anchor(scene_analysis: dict[str, Any]) -> bool:
    return bool(
        scene_analysis.get("subject")
        or scene_analysis.get("entities")
        or scene_analysis.get("objects")
        or scene_analysis.get("years")
    )


def _passes_context_gate(scene_analysis: dict[str, Any], candidate: dict[str, Any]) -> bool:
    candidate_text = _metadata_text(candidate)
    if not candidate_text:
        return False

    subject = [str(scene_analysis.get("subject", "")).strip()] if scene_analysis.get("subject") else []
    entities = [str(item) for item in scene_analysis.get("entities", [])[:4]]
    objects = [str(item) for item in scene_analysis.get("objects", [])[:4]]
    years = [str(item) for item in scene_analysis.get("years", [])[:2]]
    keywords = [str(item) for item in scene_analysis.get("keywords", [])[:6]]

    strong_hits = 0
    if _contains_any_phrase(candidate_text, subject):
        strong_hits += 1
    if _contains_any_phrase(candidate_text, entities):
        strong_hits += 1
    if _contains_any_phrase(candidate_text, objects):
        strong_hits += 1
    if _contains_any_phrase(candidate_text, years):
        strong_hits += 1

    if strong_hits >= 1:
        return True
    if not _has_context_anchor(scene_analysis):
        return _contains_any_phrase(candidate_text, keywords)
    return False


def score_candidate(scene_analysis: dict[str, Any] | str, candidate: dict[str, Any]) -> float:
    if isinstance(scene_analysis, dict):
        scene_text = str(scene_analysis.get("scene_text", ""))
        entities = [str(item) for item in scene_analysis.get("entities", [])[:4]]
        objects = [str(item) for item in scene_analysis.get("objects", [])[:4]]
        subject = [str(scene_analysis.get("subject", "")).strip()] if scene_analysis.get("subject") else []
        years = [str(item) for item in scene_analysis.get("years", [])[:2]]
        keywords = [str(item) for item in scene_analysis.get("keywords", [])[:6]]
    else:
        scene_text = str(scene_analysis)
        entities = []
        objects = []
        subject = []
        years = []
        keywords = []

    candidate_text = _metadata_text(candidate)
    query_text = _query_text(candidate)
    if not candidate_text:
        return 0.0
    semantic_score = _semantic_proxy(scene_text, candidate_text)
    overlap_score = _keyword_overlap(scene_text, candidate_text)
    entity_score = _contains_phrase(candidate_text, subject + entities)
    object_score = _contains_phrase(candidate_text, objects)
    keyword_score = _contains_phrase(candidate_text, keywords)
    year_score = _year_match(candidate_text, years)
    penalty = _generic_penalty(candidate_text)
    query_support = _keyword_overlap(query_text, candidate_text) if query_text else 0.0
    score = (
        (semantic_score * 0.34)
        + (overlap_score * 0.16)
        + (entity_score * 0.24)
        + (object_score * 0.12)
        + (keyword_score * 0.06)
        + (year_score * 0.10)
        + (query_support * 0.06)
        - penalty
    )
    return round(max(0.0, min(1.0, score)), 4)


def rank_candidates(scene_analysis: dict[str, Any] | str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    seen_urls = set()
    for item in candidates:
        image_url = str(item.get("image_url", "")).strip()
        if not image_url or image_url in seen_urls:
            continue
        if isinstance(scene_analysis, dict) and not _passes_context_gate(scene_analysis, item):
            continue
        score = score_candidate(scene_analysis, item)
        if math.isnan(score) or score < MIN_CONFIDENCE:
            continue
        normalized = dict(item)
        normalized["confidence_score"] = score
        ranked.append(normalized)
        seen_urls.add(image_url)
    ranked.sort(key=lambda item: item.get("confidence_score", 0.0), reverse=True)
    return ranked

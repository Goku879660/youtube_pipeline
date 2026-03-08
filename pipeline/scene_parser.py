from __future__ import annotations

import re
from typing import Any


HISTORICAL_HINTS = {
    "ancient", "medieval", "century", "dynasty", "empire", "king", "queen", "battle",
    "war", "revolution", "historic", "historical", "roman", "greek", "egyptian",
    "ottoman", "victorian", "bronze age", "iron age", "190", "180", "170", "160",
}
OBJECT_HINTS = {
    "car", "automobile", "vehicle", "ship", "boat", "plane", "aircraft", "train",
    "sword", "artifact", "statue", "temple", "map", "factory", "machine", "camera",
    "animal", "wolf", "lion", "tiger", "horse", "bird", "truck", "tank",
}
MODERN_HINTS = {
    "smartphone", "computer", "ai", "technology", "cyber", "city", "office", "modern",
    "digital", "robot", "satellite", "startup", "internet", "futuristic",
}
STOPWORDS = {
    "the", "and", "with", "from", "into", "during", "after", "before", "through",
    "their", "there", "this", "that", "these", "those", "were", "was", "have",
    "has", "had", "about", "over", "under", "between", "while", "where", "when",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]+", text)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_scene(scene_text: str, visual_terms: list[str] | None = None) -> dict[str, Any]:
    text = _normalize(scene_text)
    visual_terms = visual_terms or []
    proper_nouns = re.findall(r"\b[A-Z][a-zA-Z'-]+\b", text)
    entity_phrases = re.findall(r"\b(?:[A-Z][a-zA-Z'-]+\s+){1,3}[A-Z][a-zA-Z'-]+\b", text)
    lowered = text.lower()
    tokens = [token.lower() for token in _tokenize(text)]
    years = re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", text)

    entities = list(dict.fromkeys([*entity_phrases[:4], *proper_nouns[:6]]))
    objects = [token for token in tokens if token in OBJECT_HINTS]
    keywords = [
        token for token in tokens
        if len(token) >= 4 and token not in STOPWORDS and token not in MODERN_HINTS
    ]
    visual_keywords = list(dict.fromkeys([*visual_terms[:5], *entities[:4], *objects[:4], *keywords[:6], *years[:2]]))

    subject = ""
    if entity_phrases:
        subject = entity_phrases[0]
    elif len(proper_nouns) >= 2:
        subject = " ".join(proper_nouns[:2])
    elif proper_nouns:
        subject = proper_nouns[0]
    elif visual_terms:
        subject = visual_terms[0]
    elif tokens:
        subject = " ".join(tokens[:3])

    scene_type = "general"
    if any(hint in lowered for hint in HISTORICAL_HINTS) or re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", text):
        scene_type = "historical"
    elif any(token in MODERN_HINTS for token in tokens):
        scene_type = "modern"
    elif any(token in OBJECT_HINTS for token in tokens):
        scene_type = "object"

    return {
        "subject": subject,
        "objects": list(dict.fromkeys(objects[:5])),
        "entities": entities,
        "visual_keywords": visual_keywords,
        "keywords": list(dict.fromkeys(keywords[:10])),
        "years": years[:3],
        "scene_type": scene_type,
        "scene_text": text,
    }

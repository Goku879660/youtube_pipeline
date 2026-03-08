from __future__ import annotations

import re


def _find_anchor_phrase(scene_analysis: dict, candidate: dict) -> str:
    for value in (
        scene_analysis.get("subject"),
        *(scene_analysis.get("objects", []) or []),
        *(scene_analysis.get("entities", []) or []),
    ):
        cleaned = str(value).strip()
        if cleaned:
            return cleaned
    return str(candidate.get("query_used", "")).strip()


def _collect_anchor_candidates(scene_analysis: dict, candidate: dict) -> list[str]:
    values = [
        candidate.get("query_used", ""),
        scene_analysis.get("subject", ""),
        *(scene_analysis.get("entities", []) or []),
        *(scene_analysis.get("objects", []) or []),
        *(scene_analysis.get("keywords", []) or []),
    ]
    unique: list[str] = []
    seen = set()
    for value in values:
        cleaned = str(value).strip()
        lowered = cleaned.lower()
        if not cleaned or lowered in seen or len(cleaned) < 3:
            continue
        seen.add(lowered)
        unique.append(cleaned)
        if len(unique) >= 8:
            break
    return unique


def _anchor_ratio(scene_text: str, scene_analysis: dict, candidate: dict) -> tuple[float, str]:
    if not scene_text:
        return 0.15, _find_anchor_phrase(scene_analysis, candidate)

    for anchor in _collect_anchor_candidates(scene_analysis, candidate):
        match = re.search(re.escape(anchor), scene_text, flags=re.IGNORECASE)
        if match:
            return match.start() / max(1, len(scene_text)), anchor

    return 0.15, _find_anchor_phrase(scene_analysis, candidate)


def _timing_window(index: int, overlay_count: int, duration: float, overlay_duration: float) -> tuple[float, float]:
    latest_start = max(0.8, duration - overlay_duration - 0.35)
    if overlay_count <= 1:
        return 0.8, latest_start

    usable_span = max(0.5, latest_start - 0.8)
    slot_width = usable_span / overlay_count
    start = 0.8 + (index * slot_width)
    end = 0.8 + ((index + 1) * slot_width)
    return start, max(start + 0.4, min(latest_start, end))


def _compute_overlay_start_time(
    duration: float,
    overlay_duration: float,
    mention_ratio: float,
    *,
    index: int,
    overlay_count: int,
) -> float:
    latest_start = max(0.8, duration - overlay_duration - 0.35)
    mention_time = duration * mention_ratio
    window_start, window_end = _timing_window(index, overlay_count, duration, overlay_duration)
    target_time = min(latest_start, max(0.8, mention_time))
    if target_time < window_start:
        return round(window_start, 2)
    if target_time > window_end:
        return round(window_end, 2)
    return round(target_time, 2)


def schedule_overlay(scene: dict, scene_analysis: dict, candidate: dict, image_path: str, index: int = 0) -> dict:
    scene_text = str(scene.get("narration", ""))
    duration = float(scene.get("duration_seconds") or 0)
    overlay_duration = min(3.0, max(2.2, duration * 0.10))
    target_count = max(1, int(scene.get("target_image_count") or 1))
    mention_ratio, anchor_phrase = _anchor_ratio(scene_text, scene_analysis, candidate)
    start_time = _compute_overlay_start_time(
        duration,
        overlay_duration,
        mention_ratio,
        index=index,
        overlay_count=target_count,
    )
    position = "center" if scene.get("is_hook") and index == 0 else "bottom-right"
    width_ratio = 0.38 if position == "center" else 0.34

    return {
        "scene_text": scene_text,
        "query_used": candidate.get("query_used", ""),
        "source": candidate.get("source", ""),
        "image_path": image_path,
        "overlay_start_time": round(start_time, 2),
        "overlay_duration": round(overlay_duration, 2),
        "overlay_fade_duration": 0.5,
        "overlay_position": position,
        "overlay_width_ratio": width_ratio,
        "confidence_score": float(candidate.get("confidence_score", 0.0)),
        "anchor_phrase": anchor_phrase,
    }

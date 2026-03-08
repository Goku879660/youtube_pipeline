"""
Utilities for searching image candidates from Google Open Images metadata.

Expected metadata input:
- a CSV containing at least an image id column and a label column

Public entrypoint:
- search_open_images(query: str) -> list
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False


OPEN_IMAGES_BASE_URL = "https://storage.googleapis.com/openimages/2018_04/train"
DEFAULT_METADATA_CSV = "open_images_labels.csv"
DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_MAX_RESULTS = 10

IMAGE_ID_COLUMNS = ("ImageID", "image_id", "imageid")
LABEL_COLUMNS = ("LabelName", "label", "Label", "DisplayName", "display_name")


@dataclass(frozen=True)
class OpenImageCandidate:
    image_url: str
    label: str
    source: str = "open_images"

    def as_dict(self) -> dict[str, str]:
        return {
            "image_url": self.image_url,
            "label": self.label,
            "source": self.source,
        }


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(value))


def _resolve_metadata_path() -> Path:
    candidate = os.getenv("OPEN_IMAGES_METADATA_CSV", DEFAULT_METADATA_CSV)
    return Path(candidate).expanduser()


def _select_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _build_image_url(image_id: str) -> str:
    return f"{OPEN_IMAGES_BASE_URL}/{image_id}.jpg"


def _match_score(query_tokens: list[str], label: str) -> int:
    label_tokens = set(_tokenize(label))
    if not label_tokens:
        return 0
    overlap = sum(1 for token in query_tokens if token in label_tokens)
    if overlap == 0:
        return 0
    exact_bonus = 2 if _normalize_text(label) == " ".join(query_tokens) else 0
    prefix_bonus = 1 if all(any(token.startswith(qt) or qt.startswith(token) for token in label_tokens) for qt in query_tokens) else 0
    return overlap + exact_bonus + prefix_bonus


def _iter_csv_matches_streaming(
    csv_path: Path,
    query_tokens: list[str],
    max_results: int,
) -> list[OpenImageCandidate]:
    matches: list[tuple[int, OpenImageCandidate]] = []
    seen_urls = set()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        image_id_column = _select_column(reader.fieldnames, IMAGE_ID_COLUMNS)
        label_column = _select_column(reader.fieldnames, LABEL_COLUMNS)
        if not image_id_column or not label_column:
            return []

        for row in reader:
            image_id = str(row.get(image_id_column, "")).strip()
            label = str(row.get(label_column, "")).strip()
            if not image_id or not label:
                continue
            score = _match_score(query_tokens, label)
            if score <= 0:
                continue
            image_url = _build_image_url(image_id)
            if image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            matches.append((score, OpenImageCandidate(image_url=image_url, label=label)))

    matches.sort(key=lambda item: (item[0], item[1].label), reverse=True)
    return [candidate for _, candidate in matches[:max_results]]


def _iter_csv_matches_pandas(
    csv_path: Path,
    query_tokens: list[str],
    max_results: int,
) -> list[OpenImageCandidate]:
    assert pd is not None

    matches: list[tuple[int, OpenImageCandidate]] = []
    seen_urls = set()

    for chunk in pd.read_csv(csv_path, chunksize=DEFAULT_CHUNK_SIZE):
        image_id_column = _select_column(chunk.columns, IMAGE_ID_COLUMNS)
        label_column = _select_column(chunk.columns, LABEL_COLUMNS)
        if not image_id_column or not label_column:
            return []

        working = chunk[[image_id_column, label_column]].dropna()
        for image_id, label in working.itertuples(index=False, name=None):
            image_id_str = str(image_id).strip()
            label_str = str(label).strip()
            if not image_id_str or not label_str:
                continue
            score = _match_score(query_tokens, label_str)
            if score <= 0:
                continue
            image_url = _build_image_url(image_id_str)
            if image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            matches.append((score, OpenImageCandidate(image_url=image_url, label=label_str)))

    matches.sort(key=lambda item: (item[0], item[1].label), reverse=True)
    return [candidate for _, candidate in matches[:max_results]]


def _search_metadata_csv(query: str, csv_path: Path, max_results: int) -> list[OpenImageCandidate]:
    query_tokens = _tokenize(query)
    if not query_tokens or not csv_path.exists():
        return []

    if PANDAS_AVAILABLE:
        return _iter_csv_matches_pandas(csv_path, query_tokens, max_results)
    return _iter_csv_matches_streaming(csv_path, query_tokens, max_results)


def search_open_images(query: str) -> list[dict[str, str]]:
    """
    Search the Open Images metadata CSV and return up to 10 image candidates.
    """
    normalized_query = query.strip()
    if not normalized_query:
        return []

    csv_path = _resolve_metadata_path()
    matches = _search_metadata_csv(
        query=normalized_query,
        csv_path=csv_path,
        max_results=DEFAULT_MAX_RESULTS,
    )
    return [candidate.as_dict() for candidate in matches]

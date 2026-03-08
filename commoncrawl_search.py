"""
Utilities for finding candidate images via the Common Crawl Index API.

Workflow:
- search the Common Crawl index for relevant page URLs
- fetch a limited number of HTML pages
- extract and filter image URLs
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False


DEFAULT_COMMONCRAWL_INDEX = "https://index.commoncrawl.org/CC-MAIN-2024-18-index"
DEFAULT_MAX_PAGE_RESULTS = 3
DEFAULT_MAX_IMAGES_PER_PAGE = 4
DEFAULT_MAX_RETURNED_IMAGES = 10
DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_MAX_RETRIES = 2
MIN_IMAGE_WIDTH = 400
MIN_IMAGE_HEIGHT = 250
HTTP_SESSION = requests.Session()

IGNORE_IMAGE_HINTS = {
    "icon", "logo", "sprite", "avatar", "badge", "favicon", "banner-icon", "emoji",
}


def _normalize_query(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _query_tokens(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", query.lower())


def _request_with_retries(
    method: str,
    url: str,
    *,
    timeout: int,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    stream: bool = False,
) -> requests.Response | None:
    for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
        try:
            response = HTTP_SESSION.request(
                method,
                url,
                timeout=timeout,
                params=params,
                headers=headers,
                stream=stream,
            )
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt < DEFAULT_MAX_RETRIES:
                time.sleep(1.25 * attempt)
    return None


def _build_index_query_params(query: str) -> list[tuple[str, str]]:
    tokens = _query_tokens(query)
    pattern = "*" + "*".join(tokens) + "*" if tokens else "*"
    return [
        ("url", pattern),
        ("output", "json"),
        ("limit", str(DEFAULT_MAX_PAGE_RESULTS)),
        ("filter", "status:200"),
        ("filter", "mime:text/html"),
    ]


def _parse_index_lines(text: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        try:
            item = json.loads(cleaned)
        except ValueError:
            continue
        if isinstance(item, dict):
            results.append(item)
    return results


def _get_candidate_pages(query: str) -> list[str]:
    params = _build_index_query_params(query)
    index_url = os.getenv("COMMONCRAWL_INDEX_URL", DEFAULT_COMMONCRAWL_INDEX)
    response = _request_with_retries("GET", index_url, timeout=DEFAULT_TIMEOUT_SECONDS, params=params)
    if response is None:
        return []

    seen = set()
    pages: list[str] = []
    for item in _parse_index_lines(response.text):
        page_url = str(item.get("url", "")).strip()
        mime = str(item.get("mime", "")).lower()
        status = str(item.get("status", "")).strip()
        if not page_url.startswith(("http://", "https://")):
            continue
        if page_url in seen:
            continue
        if mime and "html" not in mime:
            continue
        if status and status != "200":
            continue
        seen.add(page_url)
        pages.append(page_url)
        if len(pages) >= DEFAULT_MAX_PAGE_RESULTS:
            break
    return pages


class _ImageExtractor(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.images: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        attr_map = {key.lower(): (value or "") for key, value in attrs}
        src = (
            attr_map.get("src")
            or attr_map.get("data-src")
            or attr_map.get("data-original")
            or attr_map.get("data-lazy-src")
        )
        if not src:
            return
        self.images.append(
            {
                "src": urljoin(self.base_url, src),
                "width": attr_map.get("width", ""),
                "height": attr_map.get("height", ""),
                "alt": attr_map.get("alt", ""),
                "class": attr_map.get("class", ""),
                "id": attr_map.get("id", ""),
            }
        )


def _looks_like_ignored_image(image_url: str, image_meta: dict[str, str]) -> bool:
    haystacks = [
        image_url.lower(),
        image_meta.get("alt", "").lower(),
        image_meta.get("class", "").lower(),
        image_meta.get("id", "").lower(),
    ]
    return any(hint in haystack for haystack in haystacks for hint in IGNORE_IMAGE_HINTS)


def _parse_dimension(value: str) -> int | None:
    match = re.search(r"\d+", value or "")
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _probe_image_size(image_url: str) -> tuple[int | None, int | None]:
    if not PIL_AVAILABLE:
        return None, None
    response = _request_with_retries(
        "GET",
        image_url,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        headers={"Range": "bytes=0-65535", "User-Agent": "Mozilla/5.0"},
        stream=True,
    )
    if response is None:
        return None, None

    try:
        content = b"".join(response.iter_content(chunk_size=8192))
        with Image.open(io.BytesIO(content)) as img:
            return img.size
    except Exception:
        return None, None
    finally:
        response.close()


def _passes_size_filter(image_url: str, image_meta: dict[str, str]) -> bool:
    width = _parse_dimension(image_meta.get("width", ""))
    height = _parse_dimension(image_meta.get("height", ""))

    if width is not None and height is not None:
        return width >= MIN_IMAGE_WIDTH and height >= MIN_IMAGE_HEIGHT

    probed_width, probed_height = _probe_image_size(image_url)
    if probed_width is None or probed_height is None:
        return False
    return probed_width >= MIN_IMAGE_WIDTH and probed_height >= MIN_IMAGE_HEIGHT


def _fetch_page_html(page_url: str) -> str:
    response = _request_with_retries(
        "GET",
        page_url,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    if response is None:
        return ""
    content_type = response.headers.get("Content-Type", "").lower()
    if "html" not in content_type and response.text.lstrip()[:15].lower() != "<!doctype html":
        return ""
    return response.text


def _extract_images_from_page(page_url: str) -> list[dict[str, str]]:
    html = _fetch_page_html(page_url)
    if not html:
        return []
    parser = _ImageExtractor(page_url)
    try:
        parser.feed(html)
    except Exception:
        return []
    return parser.images[:DEFAULT_MAX_IMAGES_PER_PAGE]


def search_commoncrawl_images(query: str) -> list[dict[str, str]]:
    """
    Search Common Crawl for relevant pages, scrape images, and return up to 10 candidates.
    """
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return []

    candidates: list[dict[str, str]] = []
    seen_urls = set()

    for page_url in _get_candidate_pages(normalized_query):
        for image_meta in _extract_images_from_page(page_url):
            image_url = image_meta.get("src", "").strip()
            if not image_url.startswith(("http://", "https://")):
                continue
            if image_url in seen_urls:
                continue
            parsed = urlparse(image_url)
            if not parsed.path:
                continue
            if _looks_like_ignored_image(image_url, image_meta):
                continue
            if not _passes_size_filter(image_url, image_meta):
                continue

            seen_urls.add(image_url)
            candidates.append(
                {
                    "image_url": image_url,
                    "page_url": page_url,
                    "source": "common_crawl",
                }
            )
            if len(candidates) >= DEFAULT_MAX_RETURNED_IMAGES:
                return candidates

    return candidates

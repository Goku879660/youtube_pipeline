from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .image_downloader import ImageCache
from .image_ranker import rank_candidates
from .image_retriever import retrieve_candidates
from .overlay_scheduler import schedule_overlay
from .query_generator import build_queries, expand_retry_queries
from .scene_parser import parse_scene
from .source_selector import select_source_order


class DocumentaryImagePipeline:
    TIME_BUDGET_SECONDS = 18

    def __init__(self, cache_dir: Path, target_images_fn: Callable[[float | int | None], int]):
        self.cache = ImageCache(cache_dir)
        self.target_images_fn = target_images_fn
        self.query_cache: dict[tuple[str, str], list[dict]] = {}

    def _retrieve_candidates_cached(self, query: str, source: str) -> list[dict]:
        cache_key = (source, query.strip().lower())
        if cache_key not in self.query_cache:
            self.query_cache[cache_key] = retrieve_candidates(query, [source])
        return list(self.query_cache[cache_key])

    def process_scene(
        self,
        scene_num: int,
        scene: dict,
        *,
        status_cb: Callable | None = None,
    ) -> tuple[list[str], list[dict]]:
        scene_analysis = parse_scene(
            scene.get("narration", ""),
            scene.get("visual_search_terms", []),
        )
        queries = build_queries(scene_analysis, scene.get("visual_search_terms", []))
        source_order = select_source_order(scene_analysis)
        target_count = max(1, self.target_images_fn(scene.get("duration_seconds")))
        scene["target_image_count"] = target_count
        started_at = time.time()

        ranked_candidates: list[dict] = []
        attempted_queries: list[str] = []
        for query in queries:
            if (time.time() - started_at) >= self.TIME_BUDGET_SECONDS:
                break
            attempted_queries.append(query)
            for source in source_order:
                if source == "common_crawl" and (time.time() - started_at) > (self.TIME_BUDGET_SECONDS * 0.45):
                    continue
                candidates = self._retrieve_candidates_cached(query, source)
                ranked = rank_candidates(scene_analysis, candidates)
                if ranked:
                    ranked_candidates.extend(ranked)
                    if len(ranked_candidates) >= target_count:
                        break
            if len(ranked_candidates) >= target_count:
                break

        if len(ranked_candidates) < target_count:
            for retry_query in expand_retry_queries(scene_analysis, attempted_queries):
                if (time.time() - started_at) >= self.TIME_BUDGET_SECONDS:
                    break
                for source in source_order:
                    if source == "common_crawl" and (time.time() - started_at) > (self.TIME_BUDGET_SECONDS * 0.45):
                        continue
                    candidates = self._retrieve_candidates_cached(retry_query, source)
                    ranked = rank_candidates(scene_analysis, candidates)
                    if ranked:
                        ranked_candidates.extend(ranked)
                        if len(ranked_candidates) >= target_count:
                            break
                if len(ranked_candidates) >= target_count:
                    break

        deduped: list[dict] = []
        seen_urls = set()
        for candidate in sorted(ranked_candidates, key=lambda item: item.get("confidence_score", 0.0), reverse=True):
            image_url = candidate.get("image_url", "")
            if not image_url or image_url in seen_urls:
                continue
            deduped.append(candidate)
            seen_urls.add(image_url)
            if len(deduped) >= target_count:
                break

        overlays: list[dict] = []
        image_paths: list[str] = []
        for index, candidate in enumerate(deduped):
            image_path = self.cache.download(candidate.get("image_url", ""))
            if not image_path:
                continue
            image_paths.append(image_path)
            overlays.append(schedule_overlay(scene, scene_analysis, candidate, image_path, index=index))

        metadata_path = self.cache.cache_dir / f"scene_{scene_num:02d}_overlay_plan.json"
        metadata_path.write_text(json.dumps(overlays, ensure_ascii=False, indent=2), encoding="utf-8")

        if status_cb and overlays:
            status_cb(
                f"  Scene {scene_num} overlay planner selected {len(overlays)} relevant images",
                None,
            )
        return image_paths, overlays

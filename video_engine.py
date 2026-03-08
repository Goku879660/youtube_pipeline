"""
video_engine.py — De volledige documentaire-pipeline
Handles: AI scripting, TTS, Pexels download, MoviePy editing, cleanup
"""

import os
import json
import copy
import sqlite3
import asyncio
import shutil
import time
import threading
import subprocess
import requests
import re
import unicodedata
import random
import math
import textwrap
import warnings
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from dotenv import load_dotenv
from pipeline_status import STEP_RANGES, cancel_requested
from pipeline import DocumentaryImagePipeline

load_dotenv()

# ── Optionele imports met fallback-waarschuwingen ──────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    from moviepy import (
        VideoFileClip, AudioFileClip, CompositeVideoClip,
        CompositeAudioClip, concatenate_videoclips, TextClip,
        ImageClip,
        ColorClip, AudioClip
    )
    from moviepy.audio.AudioClip import AudioArrayClip
    from proglog import ProgressBarLogger
    import numpy as np
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

HTTP_SESSION = requests.Session()


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ══════════════════════════════════════════════════════════════════════════════


class CancellationRequested(Exception):
    pass


class ScriptValidationError(ValueError):
    pass

class ProjectDatabase:
    """SQLite-based project registry to avoid duplicate generation."""

    def __init__(self, db_path: str = "projects/registry.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT UNIQUE,
                theme_hash TEXT,
                status TEXT DEFAULT 'pending',
                output_path TEXT,
                created_at TEXT,
                duration_seconds REAL,
                file_size_mb REAL
            )
        """)
        self.conn.commit()

    def project_exists(self, theme_hash: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM projects WHERE theme_hash=? AND status='completed'",
            (theme_hash,)
        ).fetchone()
        if row:
            cols = [d[1] for d in self.conn.execute("PRAGMA table_info(projects)").fetchall()]
            return dict(zip(cols, row))
        return None

    def register_project(self, project_name: str, theme_hash: str) -> int:
        cursor = self.conn.execute(
            "INSERT OR REPLACE INTO projects (project_name, theme_hash, status, created_at) VALUES (?,?,?,?)",
            (project_name, theme_hash, "in_progress", datetime.now().isoformat())
        )
        self.conn.commit()
        return cursor.lastrowid

    def complete_project(self, project_name: str, output_path: str,
                         duration: float = 0, size_mb: float = 0):
        self.conn.execute(
            """UPDATE projects SET status='completed', output_path=?, 
               duration_seconds=?, file_size_mb=? WHERE project_name=?""",
            (output_path, duration, size_mb, project_name)
        )
        self.conn.commit()

    def fail_project(self, project_name: str, error: str):
        self.conn.execute(
            "UPDATE projects SET status='failed' WHERE project_name=?",
            (project_name,)
        )
        self.conn.commit()

    def cancel_project(self, project_name: str):
        self.conn.execute(
            "UPDATE projects SET status='cancelled' WHERE project_name=?",
            (project_name,)
        )
        self.conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# AI SCRIPT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class ScriptGenerator:
    """Generates a documentary script as structured JSON."""

    SYSTEM_PROMPT = """You are a documentary scriptwriter.
Return ONLY a JSON array. No explanation, no markdown, no text outside the JSON.

Format per element:
{
  "scene_number": 1,
  "narration": "The full narration text in English (80-120 words)",
  "duration_seconds": 65,
  "visual_search_terms": ["english term 1", "english term 2", "english term 3"],
  "is_hook": false
}

Requirements:
- Scene 1 is always a dramatic scroll-stopping hook (is_hook: true), max 20 seconds
- visual_search_terms: always 3 specific English search terms for stock video
- Write vivid, atmospheric, engaging English
- Make the opening hook unmissable and immediately curiosity-inducing
- No text outside the JSON array"""

    DURATION_PRESETS = {
        2: {
            "scene_range": (5, 6),
            "duration_range": (165, 195),
            "target_seconds": 180,
            "word_range": "420-540",
            "prompt_label": "2-minute",
            "hook_max_seconds": 18,
            "hook_max_ratio": 0.15,
        },
        5: {
            "scene_range": (6, 8),
            "duration_range": (240, 390),
            "target_seconds": 300,
            "word_range": "700-900",
            "prompt_label": "5-minute",
            "hook_max_seconds": 20,
            "hook_max_ratio": 0.12,
        },
        10: {
            "scene_range": (10, 12),
            "duration_range": (540, 720),
            "target_seconds": 600,
            "word_range": "1400-1700",
            "prompt_label": "10-minute",
            "hook_max_seconds": 20,
            "hook_max_ratio": 0.10,
        },
        15: {
            "scene_range": (14, 18),
            "duration_range": (840, 1080),
            "target_seconds": 900,
            "word_range": "2200-2600",
            "prompt_label": "15-minute",
            "hook_max_seconds": 20,
            "hook_max_ratio": 0.09,
        },
    }

    def __init__(self, target_duration_minutes: int = 10):
        self.api_type = os.getenv("AI_PROVIDER", "openai").lower()
        self.target_duration_minutes = target_duration_minutes if target_duration_minutes in self.DURATION_PRESETS else 10

        if self.api_type == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.api_type = "openai"
        else:
            raise RuntimeError("Geen AI-library gevonden. Installeer openai of anthropic.")

    def _compute_hook_max_seconds(self, preset: dict) -> int:
        target_seconds = int(preset.get("target_seconds", 0) or 0)
        hook_floor = int(preset.get("hook_max_seconds", 20) or 20)
        hook_ratio = float(preset.get("hook_max_ratio", 0.0) or 0.0)
        relative_limit = int(round(target_seconds * hook_ratio)) if target_seconds and hook_ratio else 0
        return max(12, hook_floor, relative_limit)

    def _build_user_message(self, theme: str, *, validation_feedback: str | None = None) -> str:
        preset = self.DURATION_PRESETS[self.target_duration_minutes]
        min_scenes, max_scenes = preset["scene_range"]
        min_duration, max_duration = preset["duration_range"]
        hook_max_seconds = self._compute_hook_max_seconds(preset)
        user_message = (
            f"Write a {preset['prompt_label']} documentary script about: {theme}\n\n"
            f"Requirements:\n"
            f"- Start with a breathtaking hook that grabs attention instantly\n"
            f"- The opening hook must be the strongest moment in the script\n"
            f"- Return exactly {min_scenes}-{max_scenes} scenes\n"
            f"- {preset['word_range']} words total spread across {min_scenes}-{max_scenes} scenes\n"
            f"- Total runtime around {preset['target_seconds']} seconds\n"
            f"- The sum of all duration_seconds values must be between {min_duration} and {max_duration}\n"
            f"- Every scene duration_seconds must be an integer between 12 and 120\n"
            f"- Scene 1 must be the hook and must be {hook_max_seconds} seconds or less\n"
            f"- Keep duration_seconds realistic for the narration length of each scene\n"
            f"- In-depth, atmospheric, educational\n"
            f"- Return ONLY the JSON array"
        )
        if validation_feedback:
            user_message += (
                "\n\nPrevious attempt failed validation. Fix these issues and regenerate the full JSON array:\n"
                f"- {validation_feedback}"
            )
        return user_message

    def generate(self, theme: str, *, validation_feedback: str | None = None) -> list[dict]:
        user_message = self._build_user_message(theme, validation_feedback=validation_feedback)

        if self.api_type == "anthropic":
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}]
            )
            raw = response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.4
            )
            raw = response.choices[0].message.content

        # JSON extractie & validatie
        return self._parse_script(raw)

    def _parse_script(self, raw: str) -> list[dict]:
        raw = raw.strip()

        # Verwijder eventuele markdown code fences
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)

        # Zoek array
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]

        # Soms wrapped OpenAI in {"scenes": [...]}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                for key in ["scenes", "script", "data"]:
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
            if not isinstance(data, list):
                raise ValueError("Script is geen array")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Kon script JSON niet parsen: {e}\n\nRuwe output:\n{raw[:500]}")


# ══════════════════════════════════════════════════════════════════════════════
# VOICE-OVER GENERATOR (edge-tts)
# ══════════════════════════════════════════════════════════════════════════════

class VoiceOverGenerator:
    """Generates voice-over audio with edge-tts or OpenAI TTS."""

    OPENAI_TTS_MODEL = "gpt-4o-mini-tts"

    def __init__(self, voice: str = "nl-NL-MaartenNeural"):
        self.voice = voice
        self.openai_client = None
        if str(self.voice).startswith("openai:"):
            if not OPENAI_AVAILABLE:
                raise RuntimeError("openai library niet geïnstalleerd")
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY niet gevonden in .env")
            self.openai_client = OpenAI(api_key=api_key)
        elif not EDGE_TTS_AVAILABLE:
            raise RuntimeError("edge-tts niet geïnstalleerd: pip install edge-tts")

    async def _generate_async(self, text: str, output_path: str):
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)

    def _generate_openai(self, text: str, output_path: str):
        if self.openai_client is None:
            raise RuntimeError("OpenAI TTS client is niet beschikbaar")
        openai_voice = self.voice.split(":", 1)[1] if ":" in self.voice else self.voice
        response = self.openai_client.audio.speech.create(
            model=self.OPENAI_TTS_MODEL,
            voice=openai_voice,
            input=text,
            response_format="mp3",
        )
        response.write_to_file(output_path)

    def generate_scene(self, text: str, output_path: str):
        if str(self.voice).startswith("openai:"):
            self._generate_openai(text, output_path)
            return
        asyncio.run(self._generate_async(text, output_path))

    def generate_all(self, scenes: list[dict], audio_dir: Path,
                     status_cb: Callable = None,
                     progress_range: tuple[int, int] | None = None,
                     cancel_check: Callable[[], None] | None = None) -> list[str]:
        audio_files = []
        total = max(1, len(scenes))
        for i, scene in enumerate(scenes):
            if cancel_check:
                cancel_check()
            out_path = str(audio_dir / f"scene_{i+1:02d}.mp3")
            if Path(out_path).exists():
                if status_cb:
                    progress = None
                    if progress_range:
                        start, end = progress_range
                        progress = start + int(((i + 1) / total) * (end - start))
                    status_cb(f"  Voice-over scene {i+1}/{len(scenes)} skipped (already exists)", progress)
                audio_files.append(out_path)
                continue
            if status_cb:
                status_cb(f"  Generating voice-over scene {i+1}/{len(scenes)}...", None)
            self.generate_scene(scene["narration"], out_path)
            audio_files.append(out_path)
            if status_cb and progress_range:
                start, end = progress_range
                progress = start + int(((i + 1) / total) * (end - start))
                status_cb(f"  Voice-over scene {i+1}/{len(scenes)} complete", progress)
        return audio_files


# ══════════════════════════════════════════════════════════════════════════════
# PEXELS VIDEO DOWNLOADER
# ══════════════════════════════════════════════════════════════════════════════

class PexelsDownloader:
    """Download stockvideo's van Pexels op basis van zoektermen."""

    BASE_URL = "https://api.pexels.com/videos/search"
    MIN_VALID_VIDEO_BYTES = 256 * 1024
    DOWNLOAD_RETRIES = 3
    MODE_SETTINGS = {
        "light": {"max_clips": 4, "target_seconds": 14},
        "standard": {"max_clips": 8, "target_seconds": 9},
    }
    SEARCH_STAGE_LIMIT = 3
    FRAME_SAMPLE_RATIOS = (0.15, 0.5, 0.85)

    def __init__(self, quality: str = "hd", clips_per_scene: int = 4, visual_mode: str = "standard"):
        self.api_key = os.getenv("PEXELS_API_KEY", "")
        self.quality = quality  # "hd" of "sd"
        self.clips_per_scene = max(1, clips_per_scene)
        self.visual_mode = visual_mode if visual_mode in self.MODE_SETTINGS else "standard"
        self.used_video_ids: set[int] = set()
        self._used_ids_lock = threading.Lock()
        self._validated_video_cache: dict[str, tuple[int, int, bool]] = {}
        if not self.api_key:
            raise RuntimeError("PEXELS_API_KEY niet gevonden in .env")

    def target_clip_count(self, scene_duration: float | int | None) -> int:
        duration = max(1.0, float(scene_duration or 0))
        settings = self.MODE_SETTINGS[self.visual_mode]
        if self.clips_per_scene <= 1:
            return 1 if duration <= 75 else 2
        duration_target = math.ceil(duration / settings["target_seconds"])
        return max(
            self.clips_per_scene,
            min(settings["max_clips"], duration_target),
        )

    def search_and_download(self, search_terms: list[str],
                            output_dir: Path,
                            scene_num: int,
                            target_clips: int,
                            status_cb: Callable = None,
                            cancel_check: Callable[[], None] | None = None) -> list[str]:
        """Download meerdere unieke clips voor een scène."""
        seen_ids = set()
        downloaded: list[str] = []
        staged_terms = self._expand_search_terms(search_terms)

        for stage_index, terms in enumerate(staged_terms):
            if stage_index > 0 and status_cb:
                status_cb(
                    f"  Broadening search for scene {scene_num} (stage {stage_index + 1}/{len(staged_terms)})...",
                    None,
                )
            for term in terms[:self.SEARCH_STAGE_LIMIT]:
                if cancel_check:
                    cancel_check()
                try:
                    candidates = self._search_candidates(term)
                except Exception as e:
                    if status_cb:
                        status_cb(f"  Search term '{term}' failed: {e}", None)
                    continue
                for video in candidates:
                    if cancel_check:
                        cancel_check()
                    video_id = video.get("id")
                    with self._used_ids_lock:
                        if video_id in seen_ids or video_id in self.used_video_ids:
                            continue
                        seen_ids.add(video_id)
                        if video_id is not None:
                            self.used_video_ids.add(video_id)
                    try:
                        video_path = self._download_video(
                            video,
                            output_dir,
                            scene_num,
                            len(downloaded) + 1,
                            cancel_check=cancel_check,
                        )
                    except Exception as e:
                        with self._used_ids_lock:
                            if video_id is not None:
                                self.used_video_ids.discard(video_id)
                        if status_cb:
                            status_cb(f"  Clip download failed for '{term}': {e}", None)
                        continue
                    if video_path:
                        downloaded.append(video_path)
                    if len(downloaded) >= target_clips:
                        return downloaded
        return downloaded

    def _expand_search_terms(self, search_terms: list[str]) -> list[list[str]]:
        base_terms = [term.strip() for term in search_terms if term and term.strip()]
        if not base_terms:
            return [["cinematic landscape", "atmospheric nature", "moody environment"]]

        staged_terms: list[list[str]] = [base_terms]
        simplified = []
        broad = []
        contextual = []
        mood_terms = []
        keyword_pool: list[str] = []

        for term in base_terms:
            words = [
                word for word in re.split(r"\s+", term)
                if word and re.search(r"[A-Za-z]", word)
            ]
            if len(words) >= 4:
                simplified.append(" ".join(words[:4]))
            if len(words) >= 2:
                broad.append(" ".join(words[:2]))
            cleaned_words = [
                re.sub(r"[^A-Za-z0-9-]", "", word).lower()
                for word in words
            ]
            keyword_pool.extend(word for word in cleaned_words if len(word) >= 4)

        unique_keywords = []
        seen_keywords = set()
        for word in keyword_pool:
            if word not in seen_keywords:
                seen_keywords.add(word)
                unique_keywords.append(word)

        for keyword in unique_keywords[:6]:
            contextual.append(f"{keyword} documentary")
            contextual.append(f"{keyword} cinematic")
            contextual.append(f"{keyword} landscape")

        for base in base_terms[:3]:
            mood_terms.append(f"{base} atmospheric")
            mood_terms.append(f"{base} cinematic")
            mood_terms.append(f"{base} dramatic")

        generic = [
            "documentary footage",
            "cinematic landscape",
            "atmospheric environment",
            "historical archive style",
            "moody wide shot",
        ]

        for group in (simplified, broad, contextual, mood_terms, generic):
            deduped = []
            seen = set()
            for term in group:
                normalized = term.strip().lower()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    deduped.append(term)
            if deduped and deduped not in staged_terms:
                staged_terms.append(deduped)

        return staged_terms

    def _search_candidates(self, query: str) -> list[dict]:
        headers = {"Authorization": self.api_key}
        params = {"query": query, "per_page": 30, "orientation": "landscape"}
        response = HTTP_SESSION.get(self.BASE_URL, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        videos = data.get("videos", [])
        return sorted(videos, key=lambda video: abs(video.get("duration", 8) - 8))

    def _is_usable_video_file(self, path: Path) -> bool:
        try:
            stat = path.stat()
            if not path.exists() or stat.st_size < self.MIN_VALID_VIDEO_BYTES:
                return False
        except OSError:
            return False

        cache_key = str(path.resolve())
        cache_value = self._validated_video_cache.get(cache_key)
        cache_token = (stat.st_size, int(stat.st_mtime_ns))
        if cache_value and cache_value[:2] == cache_token:
            return cache_value[2]

        if not MOVIEPY_AVAILABLE:
            self._validated_video_cache[cache_key] = (*cache_token, True)
            return True

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                clip = VideoFileClip(str(path))
                duration = float(clip.duration or 0)
                if duration <= 0.5:
                    clip.close()
                    self._validated_video_cache[cache_key] = (*cache_token, False)
                    return False
                sample_points = sorted({
                    max(0.0, min(duration - 0.05, duration * ratio))
                    for ratio in self.FRAME_SAMPLE_RATIOS
                })
                for timestamp in sample_points:
                    clip.get_frame(timestamp)
                clip.close()
                self._validated_video_cache[cache_key] = (*cache_token, True)
                return True
        except Exception:
            self._validated_video_cache[cache_key] = (*cache_token, False)
            return False

    def _collect_existing_scene_clips(self, video_dir: Path, scene_num: int, target_clips: int) -> list[Path]:
        valid_paths: list[Path] = []
        for path in sorted(video_dir.glob(f"scene_{scene_num:02d}_clip_*.mp4")):
            if self._is_usable_video_file(path):
                valid_paths.append(path)
            else:
                try:
                    path.unlink()
                except OSError:
                    pass
        return valid_paths[:target_clips]

    def _download_video(self, video: dict, output_dir: Path, scene_num: int, clip_index: int,
                        cancel_check: Callable[[], None] | None = None) -> Optional[str]:
        video_file = self._select_quality(video.get("video_files", []))
        if not video_file:
            return None

        url = video_file["link"]
        output_path = output_dir / f"scene_{scene_num:02d}_clip_{clip_index:02d}.mp4"
        temp_output_path = output_path.with_suffix(".mp4.part")

        if output_path.exists() and self._is_usable_video_file(output_path):
            return str(output_path)

        last_error = None
        for attempt in range(1, self.DOWNLOAD_RETRIES + 1):
            try:
                if temp_output_path.exists():
                    temp_output_path.unlink()
                r = HTTP_SESSION.get(url, stream=True, timeout=60)
                r.raise_for_status()
                with open(temp_output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if cancel_check:
                            cancel_check()
                        if not chunk:
                            continue
                        f.write(chunk)
                if not self._is_usable_video_file(temp_output_path):
                    raise RuntimeError("Downloaded clip is incomplete or unreadable")
                temp_output_path.replace(output_path)
                return str(output_path)
            except Exception as exc:
                last_error = exc
                for candidate in (temp_output_path, output_path):
                    if candidate.exists():
                        try:
                            candidate.unlink()
                        except OSError:
                            pass
                if cancel_check:
                    cancel_check()

        if last_error:
            raise last_error

        return None

    def _select_quality(self, video_files: list) -> Optional[dict]:
        """Kiest 720p of 1080p, negeert 4K."""
        quality_map = {"hd": [1080, 720, 480], "sd": [480, 720, 360]}
        preferred = quality_map.get(self.quality, [720, 1080])

        for target_height in preferred:
            for vf in video_files:
                h = vf.get("height", 0)
                if abs(h - target_height) <= 50 and vf.get("file_type") == "video/mp4":
                    return vf

        # Fallback: eerste mp4
        for vf in video_files:
            if vf.get("file_type") == "video/mp4" and vf.get("height", 9999) <= 1080:
                return vf
        return None

    def download_all(self, scenes: list[dict], video_dir: Path,
                     status_cb: Callable = None,
                     progress_range: tuple[int, int] | None = None,
                     cancel_check: Callable[[], None] | None = None) -> list[list[str]]:
        video_files: list[list[str]] = []
        total = max(1, len(scenes))
        for i, scene in enumerate(scenes):
            if cancel_check:
                cancel_check()
            target_clips = self.target_clip_count(scene.get("duration_seconds"))
            existing = self._collect_existing_scene_clips(video_dir, i + 1, target_clips)
            legacy = video_dir / f"scene_{i+1:02d}_raw.mp4"
            if len(existing) >= target_clips:
                if status_cb:
                    progress = None
                    if progress_range:
                        start, end = progress_range
                        progress = start + int(((i + 1) / total) * (end - start))
                    status_cb(
                        f"  Video scene {i+1}/{len(scenes)} skipped ({len(existing)} unique clips already exist)",
                        progress,
                    )
                video_files.append([str(path) for path in existing[:target_clips]])
                continue
            terms = scene.get("visual_search_terms", ["nature landscape", "sky clouds", "forest"])
            if status_cb:
                status_cb(
                    f"  Downloading video scene {i+1}/{len(scenes)}: {terms[0]} ({target_clips} unique clips target)...",
                    None,
                )
            paths = [str(path) for path in existing]
            if len(paths) < target_clips:
                downloaded = self.search_and_download(
                    terms,
                    video_dir,
                    i + 1,
                    target_clips - len(paths),
                    status_cb,
                    cancel_check=cancel_check,
                )
                paths.extend(downloaded)
            if legacy.exists() and len(paths) >= target_clips:
                legacy.unlink()
            if legacy.exists() and not paths and self._is_usable_video_file(legacy):
                paths = [str(legacy)]
            elif legacy.exists() and not self._is_usable_video_file(legacy):
                try:
                    legacy.unlink()
                except OSError:
                    pass
            video_files.append(paths)
            if status_cb and progress_range:
                start, end = progress_range
                progress = start + int(((i + 1) / total) * (end - start))
                status_cb(f"  Video scene {i+1}/{len(scenes)} processed ({len(paths)} unique clips)", progress)
        return video_files


class OpenverseImageDownloader:
    """Download openly licensed still images for scene overlays."""

    BASE_URL = "https://api.openverse.org/v1/images/"
    TOKEN_URL = "https://api.openverse.org/v1/auth_tokens/token/"
    DOWNLOAD_RETRIES = 3
    MIN_VALID_IMAGE_BYTES = 64 * 1024
    MODE_SETTINGS = {
        "light": {"min_images": 1, "max_images": 4, "target_seconds": 8},
        "standard": {"min_images": 2, "max_images": 8, "target_seconds": 7},
    }

    def __init__(self, visual_mode: str = "standard"):
        self.visual_mode = visual_mode if visual_mode in self.MODE_SETTINGS else "standard"
        self.access_token = os.getenv("OPENVERSE_ACCESS_TOKEN", "").strip()
        self.client_id = os.getenv("OPENVERSE_CLIENT_ID", "").strip()
        self.client_secret = os.getenv("OPENVERSE_CLIENT_SECRET", "").strip()

    def target_image_count(self, scene_duration: float | int | None) -> int:
        duration = max(1.0, float(scene_duration or 0))
        settings = self.MODE_SETTINGS[self.visual_mode]
        duration_target = math.ceil(duration / settings["target_seconds"])
        return max(settings["min_images"], min(settings["max_images"], duration_target))

    def _headers(self) -> dict[str, str]:
        if not self.access_token and self.client_id and self.client_secret:
            self._refresh_access_token()
        if self.access_token:
            return {"Authorization": f"Bearer {self.access_token}"}
        return {}

    def _refresh_access_token(self):
        response = HTTP_SESSION.post(
            self.TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        self.access_token = payload.get("access_token", "").strip()

    def _is_usable_image_file(self, path: Path) -> bool:
        try:
            if not path.exists() or path.stat().st_size < self.MIN_VALID_IMAGE_BYTES:
                return False
        except OSError:
            return False

        if not PILLOW_AVAILABLE:
            return True

        try:
            with PILImage.open(path) as image:
                image.verify()
            with PILImage.open(path) as image:
                width, height = image.size
            return width >= 640 and height >= 360
        except Exception:
            return False

    def _collect_existing_scene_images(self, image_dir: Path, scene_num: int, target_images: int) -> list[Path]:
        valid_paths: list[Path] = []
        for path in sorted(image_dir.glob(f"scene_{scene_num:02d}_image_*")):
            if self._is_usable_image_file(path):
                valid_paths.append(path)
            else:
                try:
                    path.unlink()
                except OSError:
                    pass
        return valid_paths[:target_images]

    def _search_candidates(self, query: str) -> list[dict]:
        params = {
            "q": query,
            "page_size": 10,
            "license_type": "commercial,modification",
            "category": "digitized_artwork,illustration,photograph",
            "aspect_ratio": "wide",
            "size": "large",
            "filter_dead": "true",
            "unstable__authority": "true",
            "unstable__sort_by": "relevance",
        }
        response = HTTP_SESSION.get(self.BASE_URL, headers=self._headers(), params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])

    def _download_image(self, item: dict, output_dir: Path, scene_num: int, image_index: int) -> tuple[Optional[str], Optional[dict]]:
        url = item.get("url")
        if not url:
            return None, None
        extension = item.get("filetype") or Path(url).suffix.lstrip(".") or "jpg"
        extension = extension.lower().strip(".")
        if extension not in {"jpg", "jpeg", "png", "webp"}:
            extension = "jpg"

        output_path = output_dir / f"scene_{scene_num:02d}_image_{image_index:02d}.{extension}"
        temp_output_path = output_path.with_suffix(f".{extension}.part")

        if output_path.exists() and self._is_usable_image_file(output_path):
            return str(output_path), {
                "title": item.get("title"),
                "creator": item.get("creator"),
                "license": item.get("license"),
                "license_url": item.get("license_url"),
                "attribution": item.get("attribution"),
                "source": item.get("source"),
                "foreign_landing_url": item.get("foreign_landing_url"),
                "url": url,
            }

        last_error = None
        for _ in range(self.DOWNLOAD_RETRIES):
            try:
                if temp_output_path.exists():
                    temp_output_path.unlink()
                response = HTTP_SESSION.get(url, stream=True, timeout=60)
                response.raise_for_status()
                with open(temp_output_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        handle.write(chunk)
                if not self._is_usable_image_file(temp_output_path):
                    raise RuntimeError("Downloaded image is incomplete or unreadable")
                temp_output_path.replace(output_path)
                return str(output_path), {
                    "title": item.get("title"),
                    "creator": item.get("creator"),
                    "license": item.get("license"),
                    "license_url": item.get("license_url"),
                    "attribution": item.get("attribution"),
                    "source": item.get("source"),
                    "foreign_landing_url": item.get("foreign_landing_url"),
                    "url": url,
                }
            except Exception as exc:
                last_error = exc
                for candidate in (temp_output_path, output_path):
                    if candidate.exists():
                        try:
                            candidate.unlink()
                        except OSError:
                            pass
        if last_error:
            raise last_error
        return None, None

    def search_and_download(self, search_terms: list[str], output_dir: Path, scene_num: int,
                            target_images: int, status_cb: Callable | None = None) -> tuple[list[str], list[dict]]:
        existing = self._collect_existing_scene_images(output_dir, scene_num, target_images)
        paths = [str(path) for path in existing]
        metadata: list[dict] = []
        seen_urls = set()

        for term in search_terms[:5]:
            if len(paths) >= target_images:
                break
            try:
                candidates = self._search_candidates(term)
            except Exception as exc:
                if status_cb:
                    status_cb(f"  Openverse search failed for '{term}': {exc}", None)
                continue
            for item in candidates:
                if len(paths) >= target_images:
                    break
                candidate_url = item.get("url")
                if not candidate_url or candidate_url in seen_urls:
                    continue
                seen_urls.add(candidate_url)
                try:
                    image_path, item_metadata = self._download_image(item, output_dir, scene_num, len(paths) + 1)
                except Exception as exc:
                    if status_cb:
                        status_cb(f"  Openverse image download failed: {exc}", None)
                    continue
                if image_path:
                    paths.append(image_path)
                if item_metadata:
                    metadata.append(item_metadata)

        metadata_path = output_dir / f"scene_{scene_num:02d}_images.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return paths[:target_images], metadata


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO EDITOR (MoviePy)
# ══════════════════════════════════════════════════════════════════════════════

class VideoEditor:
    """Monteert audio + video + ondertitels tot een finale documentaire."""

    FONT = "DejaVu-Sans-Bold"
    SUBTITLE_COLOR = "white"
    SUBTITLE_STROKE = "black"
    SUBTITLE_SIZE = 36
    SUBTITLE_ACTIVE_COLOR = "#F8E45C"
    SUBTITLE_MAX_WIDTH = 1720
    SUBTITLE_BOTTOM_MARGIN = 52
    MUSIC_VOLUME = 0.12      # Achtergrondmuziek volume
    SPEECH_VOLUME = 1.0      # Spraak volume
    MAX_SOURCE_SEGMENT_SECONDS = 10
    MIN_SOURCE_SEGMENT_SECONDS = 2.5
    TARGET_SOURCE_SEGMENT_SECONDS = 5
    SCENE_RENDER_THREADS = 2
    FINAL_RENDER_THREADS = 2
    SCENE_RENDER_PRESET = "superfast"
    FINAL_RENDER_PRESET = "veryfast"
    SCENE_RENDER_RETRIES = 3
    FINAL_RENDER_RETRIES = 2
    MIN_RENDER_FILE_BYTES = 64 * 1024
    RENDER_VALIDATION_SAMPLE_RATIOS = (0.1, 0.5, 0.9, 0.98)
    RENDER_VALIDATION_TAIL_SECONDS = (0.2, 0.05, 0.01)
    OVERLAY_WIDTH_RATIOS = (0.38, 0.44, 0.5)
    ENTITY_HINTS = {
        "god", "gods", "demi-god", "demigod", "king", "queen", "emperor", "prince",
        "dynasty", "empire", "battle", "war", "temple", "map", "manuscript", "artifact",
        "myth", "legend", "deity", "goddess", "hero", "persia", "persian", "rome", "greek",
        "china", "egypt", "ottoman", "medieval", "ancient", "historian", "painting", "portrait",
    }

    class RenderLogger(ProgressBarLogger):
        def __init__(self, status_cb: Callable, progress_range: tuple[int, int],
                     cancel_check: Callable[[], None] | None = None):
            super().__init__(min_time_interval=0.5)
            self.status_cb = status_cb
            self.progress_range = progress_range
            self.started_at = time.time()
            self.cancel_check = cancel_check

        def bars_callback(self, bar, attr, value, old_value=None):
            if bar != "t" or attr != "index":
                return
            total = self.bars.get(bar, {}).get("total")
            if not total:
                return
            if self.cancel_check:
                self.cancel_check()
            current_frame = min(total, int(value) + 1)
            start, end = self.progress_range
            fraction = min(1.0, max(0.0, current_frame / total))
            progress = start + int(fraction * (end - start))
            elapsed_seconds = max(0.0, time.time() - self.started_at)
            eta_seconds = None
            if fraction > 0:
                eta_seconds = max(0, int((elapsed_seconds / fraction) - elapsed_seconds))
            details = {
                "export_status": {
                    "progress_percent": int(fraction * 100),
                    "current_frame": current_frame,
                    "total_frames": int(total),
                    "elapsed_seconds": int(elapsed_seconds),
                    "eta_seconds": eta_seconds,
                }
            }
            self.status_cb(f"  Export render {int(fraction * 100)}%", progress, "info", details)

    def __init__(self, project_dir: Path, music_dir: Optional[str] = None):
        self.project_dir = project_dir
        self.music_dir = Path(music_dir) if music_dir else None
        self._validated_render_cache: dict[str, tuple[int, int, bool]] = {}
        if not MOVIEPY_AVAILABLE:
            raise RuntimeError("moviepy niet geïnstalleerd: pip install moviepy")

    def _is_render_file_usable(self, path: Path) -> bool:
        try:
            stat = path.stat()
            if not path.exists() or stat.st_size < self.MIN_RENDER_FILE_BYTES:
                return False
        except Exception:
            return False
        cache_key = str(path.resolve())
        cache_value = self._validated_render_cache.get(cache_key)
        cache_token = (stat.st_size, int(stat.st_mtime_ns))
        if cache_value and cache_value[:2] == cache_token:
            return cache_value[2]
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", UserWarning)
                clip = VideoFileClip(str(path))
                duration = float(clip.duration or 0)
                if duration <= 1.0:
                    clip.close()
                    self._validated_render_cache[cache_key] = (*cache_token, False)
                    return False
                safe_margin = min(0.25, max(0.05, duration / 200))
                sample_points = sorted({
                    max(0.05, min(duration - safe_margin, duration * ratio))
                    for ratio in self.RENDER_VALIDATION_SAMPLE_RATIOS
                } | {
                    max(0.05, min(duration - safe_margin, duration - offset))
                    for offset in self.RENDER_VALIDATION_TAIL_SECONDS
                })
                for timestamp in sample_points:
                    clip.get_frame(min(duration - safe_margin, timestamp))
                clip.close()
                hard_warnings = [warning for warning in caught_warnings]
                if hard_warnings:
                    self._validated_render_cache[cache_key] = (*cache_token, False)
                    return False
                self._validated_render_cache[cache_key] = (*cache_token, True)
                return True
        except Exception:
            self._validated_render_cache[cache_key] = (*cache_token, False)
            return False

    def _scene_index_from_render_path(self, path: Path) -> int | None:
        match = re.search(r"scene_(\d+)\.mp4$", path.name)
        if not match:
            return None
        index = int(match.group(1)) - 1
        return index if index >= 0 else None

    def _extract_problematic_media_path(self, error: Exception) -> Path | None:
        match = re.search(r"(/[^\s,]+?\.(?:mp4|m4a|mp3))", str(error))
        if not match:
            return None
        path = Path(match.group(1))
        try:
            resolved = path.resolve()
        except OSError:
            return None
        try:
            if resolved.is_relative_to(self.project_dir.resolve()):
                return resolved
        except ValueError:
            return None
        return None

    def _remove_problematic_media(self, path: Path, status_cb: Callable | None = None) -> bool:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return False
        if status_cb:
            status_cb(f"  Removed suspicious media file and retrying: {path.name}", None)
        return True

    def _prune_scene_video_paths(self, scene_video_paths: list[str], keep_count: int) -> list[str]:
        if keep_count <= 0:
            return []
        existing = [path for path in scene_video_paths if path and os.path.exists(path)]
        if len(existing) <= keep_count:
            return existing
        return existing[:keep_count]

    def _render_emergency_scene_fallback(
        self,
        scene_index: int,
        total_scenes: int,
        audio_path: str,
        scene_output_path: Path,
        *,
        status_cb: Callable | None = None,
    ) -> Path:
        audio_clip = None
        scene_clip = None
        try:
            if not os.path.exists(audio_path):
                raise RuntimeError("Missing audio for emergency scene fallback")

            audio_clip = AudioFileClip(audio_path)
            duration = max(1.0, float(audio_clip.duration or 0))
            scene_clip = self._create_fallback_clip(duration).with_audio(audio_clip)
            scene_clip.write_videofile(
                str(scene_output_path),
                fps=24,
                codec="libx264",
                audio_codec="aac",
                preset=self.SCENE_RENDER_PRESET,
                threads=1,
                logger=None,
                temp_audiofile=str(scene_output_path.with_suffix(".m4a")),
                remove_temp=True,
                pixel_format="yuv420p",
                ffmpeg_params=["-movflags", "+faststart"],
            )
            if not self._is_render_file_usable(scene_output_path):
                raise RuntimeError("Emergency fallback scene render produced an unreadable output file")
            if status_cb:
                status_cb(
                    f"  Scene {scene_index + 1}/{total_scenes} emergency fallback rendered successfully",
                    None,
                    "warning",
                )
            return scene_output_path
        finally:
            if scene_clip is not None:
                scene_clip.close()
            if audio_clip is not None:
                audio_clip.close()

    def _render_ffmpeg_emergency_scene_fallback(
        self,
        scene_index: int,
        total_scenes: int,
        audio_path: str,
        scene_output_path: Path,
        *,
        status_cb: Callable | None = None,
    ) -> Path:
        if not os.path.exists(audio_path):
            raise RuntimeError("Missing audio for ffmpeg emergency scene fallback")

        temp_audio_path = scene_output_path.with_suffix(".fallback.m4a")
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0A0A14:s=1920x1080:r=24",
            "-i",
            audio_path,
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-threads",
            "1",
            "-tune",
            "stillimage",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            str(scene_output_path),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not self._is_render_file_usable(scene_output_path):
                raise RuntimeError("FFmpeg emergency fallback scene render produced an unreadable output file")
            if status_cb:
                status_cb(
                    f"  Scene {scene_index + 1}/{total_scenes} ffmpeg emergency fallback rendered successfully",
                    None,
                    "warning",
                )
            return scene_output_path
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(stderr or "ffmpeg emergency fallback failed") from exc
        finally:
            temp_audio_path.unlink(missing_ok=True)

    def build(self, scenes: list[dict], audio_files: list[str],
              video_files: list[list[str]], image_files: list[list[str]], output_path: str,
              status_cb: Callable = None,
              progress_ranges: dict[str, tuple[int, int]] | None = None,
              cancel_check: Callable[[], None] | None = None) -> str:

        output_path = str(Path(output_path).resolve())
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        temp_output_path = output_path_obj.with_name(f"{output_path_obj.stem}.rendering.mp4")
        temp_audio_path = self.project_dir / "render_audio_temp.m4a"
        scene_cache_dir = self.project_dir / "render_cache"
        scene_cache_dir.mkdir(parents=True, exist_ok=True)
        scene_output_paths: list[Path] = []

        scene_progress_range = (65, 85)
        render_progress_range = (85, 100)
        if progress_ranges:
            scene_progress_range = progress_ranges.get("montage", scene_progress_range)
            render_progress_range = progress_ranges.get("export", render_progress_range)

        def render_scene(scene_index: int, scene: dict, audio_path: str, scene_video_paths: list[str], scene_image_paths: list[str], *, force_render: bool = False):
            if cancel_check:
                cancel_check()

            scene_output_path = scene_cache_dir / f"scene_{scene_index + 1:02d}.mp4"
            if not force_render and scene_output_path.exists() and self._is_render_file_usable(scene_output_path):
                if status_cb:
                    start, end = scene_progress_range
                    total = max(1, len(scenes))
                    progress = start + int(((scene_index + 1) / total) * (end - start))
                    status_cb(f"  Scene {scene_index + 1}/{len(scenes)} render cache reused", progress)
                return scene_output_path
            if scene_output_path.exists():
                scene_output_path.unlink()

            if status_cb:
                prefix = "Rebuilding" if force_render else "Editing"
                status_cb(f"  {prefix} scene {scene_index + 1}/{len(scenes)}...", None)

            attempt = 0
            working_video_paths = [path for path in scene_video_paths if path]
            working_image_paths = [path for path in scene_image_paths if path]
            while attempt < self.SCENE_RENDER_RETRIES:
                attempt += 1
                audio_clip = None
                scene_clip = None
                vid = None
                loaded_sources = []
                created_segments = []
                subtitle_clips = []
                try:
                    if not os.path.exists(audio_path):
                        return scene_output_path

                    with warnings.catch_warnings():
                        warnings.simplefilter("error", UserWarning)
                        audio_clip = AudioFileClip(audio_path)
                        duration = audio_clip.duration

                        if attempt == 2:
                            pruned = self._prune_scene_video_paths(
                                working_video_paths,
                                max(1, len(working_video_paths) - 1),
                            )
                            if len(pruned) != len(working_video_paths):
                                working_video_paths = pruned
                                if status_cb:
                                    status_cb(
                                        f"  Scene {scene_index + 1}/{len(scenes)} retry 2/{self.SCENE_RENDER_RETRIES}: retrying with fewer source clips",
                                        None,
                                        "warning",
                                    )
                        elif attempt == 3:
                            if working_video_paths and status_cb:
                                status_cb(
                                    f"  Scene {scene_index + 1}/{len(scenes)} retry 3/{self.SCENE_RENDER_RETRIES}: retrying without source video",
                                    None,
                                    "warning",
                                )
                            working_video_paths = []
                            working_image_paths = []

                        vid, loaded_sources, created_segments = self._build_scene_visual(
                            scene,
                            working_video_paths,
                            working_image_paths,
                            duration
                        )
                        subtitle_clips = self._create_subtitles(
                            scene["narration"], duration, (1920, 1080)
                        )

                        if subtitle_clips:
                            scene_clip = CompositeVideoClip([vid] + subtitle_clips)
                        else:
                            scene_clip = vid

                        scene_clip = scene_clip.with_audio(audio_clip)
                        scene_clip.write_videofile(
                            str(scene_output_path),
                            fps=24,
                            codec="libx264",
                            audio_codec="aac",
                            preset=self.SCENE_RENDER_PRESET,
                            threads=self.SCENE_RENDER_THREADS,
                            logger=None,
                            temp_audiofile=str(scene_cache_dir / f"scene_{scene_index + 1:02d}.m4a"),
                            remove_temp=True,
                            pixel_format="yuv420p",
                            ffmpeg_params=["-movflags", "+faststart"],
                        )
                    if not self._is_render_file_usable(scene_output_path):
                        raise RuntimeError("Scene render produced an unreadable output file")
                    if status_cb:
                        start, end = scene_progress_range
                        total = max(1, len(scenes))
                        progress = start + int(((scene_index + 1) / total) * (end - start))
                        status_cb(f"  Scene {scene_index + 1}/{len(scenes)} edit complete", progress)
                    return scene_output_path
                except Exception as exc:
                    bad_media_path = self._extract_problematic_media_path(exc)
                    if scene_output_path.exists():
                        scene_output_path.unlink(missing_ok=True)
                    if bad_media_path and attempt < self.SCENE_RENDER_RETRIES:
                        if self._remove_problematic_media(bad_media_path, status_cb=status_cb):
                            working_video_paths = [
                                path for path in working_video_paths
                                if Path(path).resolve() != bad_media_path
                            ]
                            continue
                    if attempt < self.SCENE_RENDER_RETRIES:
                        if status_cb:
                            status_cb(
                                f"  Scene {scene_index + 1}/{len(scenes)} render attempt {attempt}/{self.SCENE_RENDER_RETRIES} failed: {exc}. Retrying...",
                                None,
                                "warning",
                            )
                        continue
                    if status_cb:
                        status_cb(
                            f"  Scene {scene_index + 1}/{len(scenes)} render failed after {self.SCENE_RENDER_RETRIES} attempts; switching to emergency fallback",
                            None,
                            "warning",
                        )
                    try:
                        return self._render_emergency_scene_fallback(
                            scene_index,
                            len(scenes),
                            audio_path,
                            scene_output_path,
                            status_cb=status_cb,
                        )
                    except Exception as fallback_exc:
                        if status_cb:
                            status_cb(
                                f"  Scene {scene_index + 1}/{len(scenes)} MoviePy emergency fallback failed; trying ffmpeg fallback",
                                None,
                                "warning",
                            )
                        try:
                            return self._render_ffmpeg_emergency_scene_fallback(
                                scene_index,
                                len(scenes),
                                audio_path,
                                scene_output_path,
                                status_cb=status_cb,
                            )
                        except Exception as ffmpeg_exc:
                            raise RuntimeError(
                                f"Scene render failed for scene {scene_index + 1}: {exc}; emergency fallback also failed: {fallback_exc}; ffmpeg fallback also failed: {ffmpeg_exc}"
                            ) from ffmpeg_exc
                finally:
                    if scene_clip is not None:
                        scene_clip.close()
                    if audio_clip is not None:
                        audio_clip.close()
                    for clip in subtitle_clips:
                        clip.close()
                    closed_ids = set()
                    for group in (created_segments, loaded_sources):
                        for clip in group:
                            if id(clip) in closed_ids:
                                continue
                            clip.close()
                            closed_ids.add(id(clip))

        try:
            if temp_output_path.exists():
                temp_output_path.unlink()
            if temp_audio_path.exists():
                temp_audio_path.unlink()

            for i, (scene, audio_path, scene_video_paths, scene_image_paths) in enumerate(
                    zip(scenes, audio_files, video_files, image_files)):
                scene_output_path = render_scene(i, scene, audio_path, scene_video_paths, scene_image_paths)
                scene_output_paths.append(scene_output_path)

            if not scene_output_paths:
                raise RuntimeError("Geen clips konden worden gemonteerd")

            cached_scene_paths = []
            for i, path in enumerate(scene_output_paths):
                if self._is_render_file_usable(path):
                    cached_scene_paths.append(path)
                    continue
                path.unlink(missing_ok=True)
                rebuilt_path = render_scene(i, scenes[i], audio_files[i], video_files[i], image_files[i], force_render=True)
                if not self._is_render_file_usable(rebuilt_path):
                    rebuilt_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Rendered scene cache became unreadable before final export and rebuild failed: {path.name}"
                    )
                cached_scene_paths.append(rebuilt_path)
            if not cached_scene_paths:
                raise RuntimeError("No rendered scene cache files were produced")

            if status_cb:
                status_cb("Combining all scenes...", None)

            final_render_attempt = 0
            while final_render_attempt < self.FINAL_RENDER_RETRIES:
                final_render_attempt += 1
                clips = []
                background_tracks = []
                final = None
                try:
                    clips = [VideoFileClip(str(path)) for path in cached_scene_paths]
                    final = concatenate_videoclips(clips, method="chain")

                    if self.music_dir and self.music_dir.exists():
                        final, background_tracks = self._add_background_music(final, status_cb)

                    if status_cb:
                        status_cb(
                            "Rendering final video (this takes 5-10 min)...",
                            None,
                            "info",
                            {"export_status": {
                                "progress_percent": 0,
                                "current_frame": 0,
                                "total_frames": None,
                                "elapsed_seconds": 0,
                                "eta_seconds": None,
                            }}
                        )

                    final.write_videofile(
                        str(temp_output_path),
                        fps=24,
                        codec="libx264",
                        audio_codec="aac",
                        preset=self.FINAL_RENDER_PRESET,
                        threads=self.FINAL_RENDER_THREADS,
                        logger=self.RenderLogger(status_cb, render_progress_range, cancel_check=cancel_check) if status_cb else None,
                        temp_audiofile=str(temp_audio_path),
                        remove_temp=True,
                        pixel_format="yuv420p",
                        ffmpeg_params=["-movflags", "+faststart"],
                    )
                    break
                except Exception as exc:
                    bad_media_path = self._extract_problematic_media_path(exc)
                    repaired = False
                    if (
                        bad_media_path
                        and bad_media_path.parent == scene_cache_dir.resolve()
                        and final_render_attempt < self.FINAL_RENDER_RETRIES
                    ):
                        scene_index = self._scene_index_from_render_path(bad_media_path)
                        if scene_index is not None and scene_index < len(scenes):
                            if status_cb:
                                status_cb(
                                    f"  Final render retry {final_render_attempt}/{self.FINAL_RENDER_RETRIES}: rebuilding unreadable cached scene {scene_index + 1}",
                                    None,
                                    "warning",
                                )
                            bad_media_path.unlink(missing_ok=True)
                            rebuilt_path = render_scene(
                                scene_index,
                                scenes[scene_index],
                                audio_files[scene_index],
                                video_files[scene_index],
                                image_files[scene_index],
                                force_render=True,
                            )
                            if self._is_render_file_usable(rebuilt_path):
                                cached_scene_paths[scene_index] = rebuilt_path
                                repaired = True
                    if not repaired:
                        raise RuntimeError(f"Final ffmpeg render failed: {exc}") from exc
                finally:
                    if final is not None:
                        final.close()
                    closed_ids = set()
                    for clip in clips:
                        if id(clip) in closed_ids:
                            continue
                        clip.close()
                        closed_ids.add(id(clip))
                    for track in background_tracks:
                        track.close()
                    if temp_audio_path.exists():
                        temp_audio_path.unlink()

            if final_render_attempt >= self.FINAL_RENDER_RETRIES and (
                not temp_output_path.exists() or temp_output_path.stat().st_size == 0
            ):
                raise RuntimeError("Final ffmpeg render failed after automatic scene cache recovery")

            if not temp_output_path.exists() or temp_output_path.stat().st_size == 0:
                raise RuntimeError("Render voltooid zonder geldig outputbestand")

            shutil.move(str(temp_output_path), output_path)
            return output_path
        finally:
            if temp_output_path.exists():
                temp_output_path.unlink()
            if temp_audio_path.exists():
                temp_audio_path.unlink()

    def _create_fallback_clip(self, duration: float) -> ColorClip:
        """Zwart scherm als fallback wanneer geen stockvideo beschikbaar is."""
        return ColorClip(size=(1920, 1080), color=(10, 10, 20), duration=duration)

    def _build_image_sequence(self, image_paths: list[str], duration: float):
        if not image_paths:
            fallback = self._create_fallback_clip(duration)
            return fallback, [fallback]

        image_segments = []
        segment_duration = max(2.0, duration / max(1, len(image_paths)))

        for path in image_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                clip = ImageClip(path).resized(height=1080)
                clip = clip.with_position("center").with_duration(segment_duration)
                image_segments.append(clip)
            except Exception:
                continue

        if not image_segments:
            fallback = self._create_fallback_clip(duration)
            return fallback, [fallback]

        total_duration = sum(segment.duration for segment in image_segments)
        if total_duration < duration:
            filler = image_segments[-1].with_duration(duration - total_duration)
            image_segments.append(filler)

        if len(image_segments) == 1:
            image_segments[0] = image_segments[0].with_duration(duration)
            return image_segments[0], image_segments

        sequence = concatenate_videoclips(image_segments, method="compose").subclipped(0, duration)
        return sequence, image_segments + [sequence]

    def _scene_overlay_density(self, scene: dict, duration: float) -> int:
        narration = str(scene.get("narration", ""))
        lower_text = narration.lower()
        keyword_hits = sum(1 for hint in self.ENTITY_HINTS if hint in lower_text)
        titlecase_hits = len(re.findall(r"\b[A-Z][a-z]{3,}\b", narration))
        base_density = max(1, math.ceil(duration / 7))
        if scene.get("is_hook"):
            return min(3, max(1, math.ceil(duration / 8)))
        if keyword_hits >= 3 or titlecase_hits >= 6:
            return min(8, max(3, base_density + 1))
        if keyword_hits >= 1 or titlecase_hits >= 3:
            return min(7, max(2, base_density))
        return min(6, max(2, base_density))

    def _build_image_overlays(self, scene: dict, image_paths: list[str], duration: float, size: tuple[int, int]):
        overlays = []
        overlay_plan = scene.get("image_overlay_plan")
        if isinstance(overlay_plan, list) and overlay_plan:
            try:
                from moviepy.video.fx.FadeIn import FadeIn
                from moviepy.video.fx.FadeOut import FadeOut
            except Exception:
                FadeIn = None
                FadeOut = None

            sorted_plan = sorted(
                [item for item in overlay_plan if isinstance(item, dict)],
                key=lambda item: float(item.get("overlay_start_time", 0.0) or 0.0),
            )
            previous_end = 0.0
            for item in sorted_plan:
                path = str(item.get("image_path", "")).strip()
                if not path or not os.path.exists(path):
                    continue
                try:
                    width_ratio = float(item.get("overlay_width_ratio", 0.34))
                    width_ratio = min(0.40, max(0.30, width_ratio))
                    image_clip = ImageClip(path).resized(width=int(size[0] * width_ratio))
                    overlay_duration = min(3.0, max(2.0, float(item.get("overlay_duration", 2.8))))
                    raw_start_time = float(item.get("overlay_start_time", 1.0))
                    fade_duration = float(item.get("overlay_fade_duration", 0.5))
                    max_start = max(0.0, duration - overlay_duration - 0.2)
                    start_time = max(previous_end + 0.15, min(max_start, raw_start_time))
                    image_clip = image_clip.with_duration(max(1.5, min(duration, overlay_duration)))
                    image_clip = image_clip.with_start(max(0.0, min(duration - 0.2, start_time)))
                    previous_end = start_time + overlay_duration

                    position = str(item.get("overlay_position", "bottom-right"))
                    if position == "center":
                        image_clip = image_clip.with_position("center")
                    else:
                        x_pos = size[0] - image_clip.w - 70
                        y_pos = size[1] - image_clip.h - 110
                        image_clip = image_clip.with_position((max(40, x_pos), max(80, y_pos)))

                    if FadeIn and FadeOut:
                        image_clip = image_clip.with_effects([FadeIn(fade_duration), FadeOut(fade_duration)])
                    overlays.append(image_clip)
                except Exception:
                    continue
            if overlays:
                return overlays

        valid_paths = [path for path in image_paths if path and os.path.exists(path)]
        if not valid_paths:
            return overlays

        overlay_count = min(len(valid_paths), self._scene_overlay_density(scene, duration))
        selected_paths = valid_paths[:overlay_count]
        segment_duration = max(2.2, min(3.0, duration / max(1, overlay_count + 1)))
        available_window = max(0.8, duration - segment_duration - 0.4)
        if overlay_count > 1:
            start_points = np.linspace(0.9, available_window, overlay_count)
        else:
            start_points = [min(1.2, available_window)]
        positions = [
            ("left", 90),
            ("right", 120),
            ("left", 170),
            ("right", 200),
        ]

        for index, path in enumerate(selected_paths):
            try:
                width_ratio = self.OVERLAY_WIDTH_RATIOS[index % len(self.OVERLAY_WIDTH_RATIOS)]
                image_clip = ImageClip(path).resized(width=int(size[0] * width_ratio))
                image_clip = image_clip.with_duration(segment_duration)
                image_clip = image_clip.with_start(float(start_points[index]))
                horiz, vertical = positions[index % len(positions)]
                x_pos = 70 if horiz == "left" else size[0] - int(size[0] * width_ratio) - 70
                image_clip = image_clip.with_position((x_pos, vertical))
                overlays.append(image_clip)
            except Exception:
                continue

        return overlays

    def _build_scene_visual(self, scene: dict, video_paths: list[str], image_paths: list[str], duration: float):
        loaded_sources = []
        created_segments = []
        image_clips = []

        try:
            for path in video_paths or []:
                if not path or not os.path.exists(path):
                    continue
                try:
                    clip = VideoFileClip(path).resized((1920, 1080)).without_audio()
                    loaded_sources.append(clip)
                except Exception:
                    continue

            if not loaded_sources:
                image_visual, image_components = self._build_image_sequence(image_paths, duration)
                created_segments.extend(image_components)
                return image_visual, loaded_sources, created_segments

            remaining = duration
            random.shuffle(loaded_sources)
            for source in loaded_sources:
                if remaining <= 0:
                    break
                segment_limit = min(source.duration, self.MAX_SOURCE_SEGMENT_SECONDS, remaining)
                if segment_limit <= 0:
                    continue
                if segment_limit <= self.MIN_SOURCE_SEGMENT_SECONDS:
                    segment_length = segment_limit
                else:
                    segment_length = random.uniform(
                        self.MIN_SOURCE_SEGMENT_SECONDS,
                        min(segment_limit, self.TARGET_SOURCE_SEGMENT_SECONDS),
                    )
                if source.duration <= segment_length:
                    segment = source.subclipped(0, source.duration)
                else:
                    max_start = max(0.0, source.duration - segment_length)
                    start_at = random.uniform(0.0, max_start) if max_start else 0.0
                    segment = source.subclipped(start_at, start_at + segment_length)
                created_segments.append(segment)
                remaining -= segment.duration

            if remaining > 0:
                filler = self._create_fallback_clip(remaining)
                created_segments.append(filler)

            if len(created_segments) == 1:
                base_visual = created_segments[0]
            else:
                base_visual = concatenate_videoclips(created_segments, method="compose")
                created_segments.append(base_visual)

            image_clips = self._build_image_overlays(scene, image_paths, duration, (1920, 1080))
            if image_clips:
                composite = CompositeVideoClip([base_visual] + image_clips, size=(1920, 1080)).with_duration(duration)
                created_segments.extend(image_clips)
                created_segments.append(composite)
                return composite, loaded_sources, created_segments
            return base_visual, loaded_sources, created_segments
        except Exception:
            image_visual, image_components = self._build_image_sequence(image_paths, duration)
            created_segments.extend(image_components)
            return image_visual, loaded_sources, created_segments

    def _create_subtitles(self, text: str, duration: float,
                          size: tuple) -> list:
        """Maakt ondertitels met witte basisregel en gele actieve highlight."""
        clips = []
        if not text.strip():
            return []

        sentence_groups = []
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            cleaned = sentence.strip()
            if not cleaned:
                continue
            wrapped = textwrap.wrap(cleaned, width=42, break_long_words=False, break_on_hyphens=False)
            if not wrapped:
                continue
            if len(wrapped) == 1:
                sentence_groups.append(wrapped[0])
                continue
            current = []
            current_len = 0
            for line in wrapped:
                line_words = len(line.split())
                if current and current_len + line_words > 12:
                    sentence_groups.append(" ".join(current))
                    current = [line]
                    current_len = line_words
                else:
                    current.append(line)
                    current_len += line_words
            if current:
                sentence_groups.append(" ".join(current))

        if not sentence_groups:
            return []

        total_words = sum(max(1, len(chunk.split())) for chunk in sentence_groups)
        cursor = 0.0

        for index, chunk in enumerate(sentence_groups):
            chunk_words = max(1, len(chunk.split()))
            if index == len(sentence_groups) - 1:
                chunk_duration = max(0.6, duration - cursor)
            else:
                chunk_duration = max(0.8, duration * (chunk_words / total_words))
            if PILLOW_AVAILABLE:
                chunk_clips = self._create_highlighted_subtitle_chunk(
                    chunk,
                    cursor,
                    chunk_duration,
                    size,
                )
                if chunk_clips:
                    clips.extend(chunk_clips)
                    cursor += chunk_duration
                    continue
            try:
                txt = TextClip(
                    chunk,
                    fontsize=self.SUBTITLE_SIZE,
                    color=self.SUBTITLE_COLOR,
                    font=self.FONT,
                    stroke_color=self.SUBTITLE_STROKE,
                    stroke_width=2,
                    method="caption",
                    size=(size[0] - 200, None),
                    align="center"
                )
                txt = txt.with_position(("center", size[1] - 120))
                txt = txt.with_start(cursor)
                txt = txt.with_duration(chunk_duration)
                clips.append(txt)
            except Exception:
                pass
            cursor += chunk_duration

        return clips

    def _create_highlighted_subtitle_chunk(
        self,
        chunk: str,
        start_time: float,
        chunk_duration: float,
        size: tuple[int, int],
    ) -> list:
        font = self._load_subtitle_font()
        if font is None:
            return []

        layout = self._layout_subtitle_words(chunk, font, min(self.SUBTITLE_MAX_WIDTH, size[0] - 160))
        if not layout:
            return []

        base_image = self._render_subtitle_layout(
            layout,
            active_word_indexes=None,
        )
        clips = [
            ImageClip(np.array(base_image))
            .with_position(("center", size[1] - base_image.height - self.SUBTITLE_BOTTOM_MARGIN))
            .with_start(start_time)
            .with_duration(chunk_duration)
        ]

        active_groups = self._build_highlight_groups(layout["words"])
        if not active_groups:
            return clips

        group_word_counts = [max(1, len(group)) for group in active_groups]
        total_group_words = sum(group_word_counts)
        group_cursor = start_time

        for index, group in enumerate(active_groups):
            if index == len(active_groups) - 1:
                group_duration = max(0.18, (start_time + chunk_duration) - group_cursor)
            else:
                group_duration = max(0.18, chunk_duration * (len(group) / max(1, total_group_words)))
            overlay_image = self._render_subtitle_layout(
                layout,
                active_word_indexes=set(group),
            )
            clips.append(
                ImageClip(np.array(overlay_image))
                .with_position(("center", size[1] - overlay_image.height - self.SUBTITLE_BOTTOM_MARGIN))
                .with_start(group_cursor)
                .with_duration(group_duration)
            )
            group_cursor += group_duration

        return clips

    def _load_subtitle_font(self):
        font_candidates = [
            self.FONT,
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        for candidate in font_candidates:
            try:
                return ImageFont.truetype(candidate, self.SUBTITLE_SIZE)
            except Exception:
                continue
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def _layout_subtitle_words(self, chunk: str, font, max_width: int) -> dict | None:
        words = [word for word in chunk.split() if word]
        if not words:
            return None

        probe = PILImage.new("RGBA", (max_width, 10), (0, 0, 0, 0))
        draw = ImageDraw.Draw(probe)
        space_width = self._measure_text(draw, " ", font)[0]
        line_height = int(self._measure_text(draw, "Ag", font)[1] * 1.25)
        lines: list[list[dict]] = []
        current_line: list[dict] = []
        current_width = 0

        for word_index, word in enumerate(words):
            word_width, word_height = self._measure_text(draw, word, font)
            candidate_width = word_width if not current_line else current_width + space_width + word_width
            if current_line and candidate_width > max_width:
                lines.append(current_line)
                current_line = []
                current_width = 0
            entry = {
                "index": word_index,
                "text": word,
                "width": word_width,
                "height": word_height,
            }
            current_line.append(entry)
            current_width = word_width if len(current_line) == 1 else current_width + space_width + word_width
        if current_line:
            lines.append(current_line)

        if not lines:
            return None

        top_padding = 12
        bottom_padding = 14
        rendered_lines = []
        max_line_width = 0
        for line_index, line in enumerate(lines):
            line_width = sum(item["width"] for item in line) + max(0, len(line) - 1) * space_width
            max_line_width = max(max_line_width, line_width)
            cursor_x = int((max_width - line_width) / 2)
            y = top_padding + (line_index * line_height)
            word_positions = []
            for item in line:
                word_positions.append({
                    **item,
                    "x": cursor_x,
                    "y": y,
                })
                cursor_x += item["width"] + space_width
            rendered_lines.append(word_positions)

        height = top_padding + (len(lines) * line_height) + bottom_padding
        return {
            "width": max_width,
            "height": height,
            "lines": rendered_lines,
            "words": words,
            "font": font,
        }

    def _render_subtitle_layout(self, layout: dict, active_word_indexes: set[int] | None):
        image = PILImage.new("RGBA", (layout["width"], layout["height"]), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        active_word_indexes = active_word_indexes or set()

        for line in layout["lines"]:
            for word in line:
                index = word["index"]
                fill = self.SUBTITLE_ACTIVE_COLOR if index in active_word_indexes else self.SUBTITLE_COLOR
                alpha = 255 if (not active_word_indexes or index in active_word_indexes) else 0
                draw.text(
                    (word["x"], word["y"]),
                    word["text"],
                    font=layout["font"],
                    fill=self._hex_to_rgba(fill, alpha),
                    stroke_width=2,
                    stroke_fill=self._hex_to_rgba(self.SUBTITLE_STROKE, alpha),
                )
        return image

    def _measure_text(self, draw, text: str, font) -> tuple[int, int]:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
        return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])

    def _build_highlight_groups(self, words: list[str]) -> list[list[int]]:
        count = len(words)
        if count <= 0:
            return []
        if count <= 4:
            group_size = 1
        elif count <= 8:
            group_size = 2
        else:
            group_size = 3
        groups = []
        for start in range(0, count, group_size):
            groups.append(list(range(start, min(count, start + group_size))))
        return groups

    def _hex_to_rgba(self, color: str, alpha: int) -> tuple[int, int, int, int]:
        if color == "white":
            return (255, 255, 255, alpha)
        if color == "black":
            return (0, 0, 0, alpha)
        stripped = color.lstrip("#")
        if len(stripped) == 6:
            return tuple(int(stripped[i:i + 2], 16) for i in (0, 2, 4)) + (alpha,)
        return (255, 255, 255, alpha)

    def _add_background_music(self, video_clip, status_cb: Callable = None):
        """Voegt ambient muziek toe met:
        - Willekeurige volgorde van de nummers (elke video anders)
        - Automatisch herhalen als alle nummers op zijn
        - Fade-in (3 seconden) aan het begin
        - Fade-out (4 seconden) aan het einde
        - Audio ducking (muziek op 12% volume)
        """
        import random
        from moviepy.audio.AudioClip import concatenate_audioclips

        # Verzamel alle muziekbestanden
        music_files = (
            list(self.music_dir.glob("*.mp3")) +
            list(self.music_dir.glob("*.wav")) +
            list(self.music_dir.glob("*.ogg"))
        )

        if not music_files:
            return video_clip, []

        # Schud de volgorde willekeurig door elkaar
        random.shuffle(music_files)

        if status_cb:
            n = len(music_files)
            namen = ", ".join(f.name for f in music_files)
            status_cb(f"Adding background music ({n} tracks in random order: {namen})...", None)

        video_duration = video_clip.duration
        FADE_IN  = 3.0   # seconden fade-in aan het begin
        FADE_OUT = 4.0   # seconden fade-out aan het einde

        # ── Bouw de afspeellijst op in willekeurige volgorde, herhaal tot video gevuld is ──
        playlist = []
        total_built = 0.0

        # Laad alle tracks één keer
        tracks = [AudioFileClip(str(f)) for f in music_files]

        while total_built < video_duration:
            # Elke nieuwe ronde opnieuw willekeurig schudden
            random.shuffle(tracks)
            for track in tracks:
                if total_built >= video_duration:
                    break
                playlist.append(track)
                total_built += track.duration

        # Samenvoegen tot één lange audioclip
        if len(playlist) == 1:
            music = playlist[0]
        else:
            music = concatenate_audioclips(playlist)

        # Knippen op exacte video-duur
        music = music.subclipped(0, video_duration)

        # ── Fade-in en fade-out ────────────────────────────────────────────────
        from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
        music = music.with_effects([AudioFadeIn(FADE_IN), AudioFadeOut(FADE_OUT)])

        # ── Volume verlagen (audio ducking) ───────────────────────────────────
        music = music.with_volume_scaled(self.MUSIC_VOLUME)

        # ── Mixen met de voice-over ───────────────────────────────────────────
        original_audio = video_clip.audio
        mixed = CompositeAudioClip([original_audio, music])
        return video_clip.with_audio(mixed), tracks


# ══════════════════════════════════════════════════════════════════════════════
# CLEANUP MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class CleanupManager:
    """Verwijdert tijdelijke bestanden na succesvolle render."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

    def cleanup(self, keep_script: bool = True):
        """Verwijdert raw video's en tijdelijke audio, behoudt script en finale video."""
        removed = 0

        # Verwijder raw stockvideo's
        videos_dir = self.project_dir / "videos"
        if videos_dir.exists():
            for f in videos_dir.glob("*_raw.mp4"):
                f.unlink()
                removed += 1
            for f in videos_dir.glob("*_clip_*.mp4"):
                f.unlink()
                removed += 1

        # Verwijder tijdelijke audio clips
        audio_dir = self.project_dir / "audio"
        if audio_dir.exists():
            for f in audio_dir.glob("scene_*.mp3"):
                f.unlink()
                removed += 1

        render_cache_dir = self.project_dir / "render_cache"
        if render_cache_dir.exists():
            shutil.rmtree(render_cache_dir, ignore_errors=True)
            removed += 1

        images_dir = self.project_dir / "images"
        if images_dir.exists():
            for f in images_dir.glob("scene_*_image_*"):
                f.unlink()
                removed += 1
            for f in images_dir.glob("scene_*_overlay_plan.json"):
                f.unlink()
                removed += 1

        image_cache_dir = self.project_dir / "image_cache"
        if image_cache_dir.exists():
            shutil.rmtree(image_cache_dir, ignore_errors=True)
            removed += 1

        return removed


# ══════════════════════════════════════════════════════════════════════════════
# HOOFD PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class DocumentaryPipeline:
    """Orchestreert de volledige pipeline van thema → video."""

    DEFAULT_SCENE_WORKERS = 3
    MIN_SCENE_WORKERS = 1
    MAX_SCENE_WORKERS = 4
    ASSET_STATUS_INTERVAL_SECONDS = 10
    MICRO_TEST_DURATION_SECONDS = 15
    SCRIPT_GENERATION_MAX_ATTEMPTS = 4
    DURATION_PRESETS = {
        2: {"scene_range": (5, 6), "duration_range": (165, 195), "target_seconds": 180, "hook_max_seconds": 18, "hook_max_ratio": 0.15},
        5: {"scene_range": (6, 8), "duration_range": (240, 390), "target_seconds": 300, "hook_max_seconds": 20, "hook_max_ratio": 0.12},
        10: {"scene_range": (10, 12), "duration_range": (540, 720), "target_seconds": 600, "hook_max_seconds": 20, "hook_max_ratio": 0.10},
        15: {"scene_range": (14, 18), "duration_range": (840, 1080), "target_seconds": 900, "hook_max_seconds": 20, "hook_max_ratio": 0.09},
    }

    def __init__(self, theme: str, voice: str = "nl-NL-MaartenNeural",
                 video_quality: str = "hd", cleanup: bool = True,
                 target_duration_minutes: int = 10,
                 clips_per_scene: int = 4,
                 visual_mode: str = "standard",
                 scene_workers: int = DEFAULT_SCENE_WORKERS,
                 context_image_overlays: bool = True,
                 micro_test_live: bool = False,
                 status_callback: Callable = None,
                 resume_project_dir: str = None):
        self.theme = theme
        self.voice = voice
        self.video_quality = video_quality
        self.target_duration_minutes = target_duration_minutes if target_duration_minutes in self.DURATION_PRESETS else 10
        self.cleanup = bool(cleanup)
        self.clips_per_scene = max(1, clips_per_scene)
        self.visual_mode = visual_mode if visual_mode in {"light", "standard"} else "standard"
        self.scene_workers = max(self.MIN_SCENE_WORKERS, min(self.MAX_SCENE_WORKERS, int(scene_workers)))
        self.context_image_overlays = bool(context_image_overlays)
        self.micro_test_live = bool(micro_test_live)
        if self.context_image_overlays:
            self.scene_workers = min(self.scene_workers, 2)
        self.status_cb = status_callback or (lambda msg, prog, level="info", details=None: None)

        self.theme_hash = self._hash_theme(theme)
        self.base_dir = Path(__file__).resolve().parent

        if resume_project_dir:
            self.project_dir = Path(resume_project_dir).resolve()
            self.project_name = self.project_dir.name
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            safe_theme = self._slugify_theme(theme)
            self.project_name = f"{timestamp}_{safe_theme}_{self.theme_hash[:8]}"
            self.project_dir = (self.base_dir / "projects" / self.project_name).resolve()

        self.project_dir.mkdir(parents=True, exist_ok=True)
        (self.project_dir / "audio").mkdir(exist_ok=True)
        (self.project_dir / "videos").mkdir(exist_ok=True)
        (self.project_dir / "images").mkdir(exist_ok=True)
        self._cleanup_stale_artifacts()

        self.db = ProjectDatabase()

    def _cleanup_stale_artifacts(self):
        for path in self.project_dir.glob("*.rendering.mp4"):
            path.unlink(missing_ok=True)
        for path in self.project_dir.glob("final_videoTEMP_MPY*"):
            path.unlink(missing_ok=True)
        for path in self.project_dir.glob("render_audio_temp.m4a"):
            path.unlink(missing_ok=True)
        for path in self.project_dir.glob("worker_crash.log"):
            try:
                if path.exists() and path.stat().st_size == 0:
                    path.unlink()
            except OSError:
                pass

        videos_dir = self.project_dir / "videos"
        if videos_dir.exists():
            for path in videos_dir.glob("*.part"):
                path.unlink(missing_ok=True)
            for path in videos_dir.glob("*.mp4.part"):
                path.unlink(missing_ok=True)

        render_cache_dir = self.project_dir / "render_cache"
        if render_cache_dir.exists():
            for path in render_cache_dir.glob("*.m4a"):
                path.unlink(missing_ok=True)
            for path in render_cache_dir.glob("*.tmp"):
                path.unlink(missing_ok=True)

        images_dir = self.project_dir / "images"
        if images_dir.exists():
            for path in images_dir.glob("*.part"):
                path.unlink(missing_ok=True)

    def _compute_hook_max_seconds(self, preset: dict) -> int:
        target_seconds = int(preset.get("target_seconds", 0) or 0)
        hook_floor = int(preset.get("hook_max_seconds", 20) or 20)
        hook_ratio = float(preset.get("hook_max_ratio", 0.0) or 0.0)
        relative_limit = int(round(target_seconds * hook_ratio)) if target_seconds and hook_ratio else 0
        return max(12, hook_floor, relative_limit)

    def _split_sentences(self, narration: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", narration.strip())
        return [part.strip() for part in parts if part.strip()]

    def _repair_hook_scene_payload(self, scenes: list[dict], hook_max_seconds: int) -> list[dict]:
        if len(scenes) < 2:
            return scenes

        repaired = copy.deepcopy(scenes)
        first_scene = repaired[0]
        second_scene = repaired[1]

        try:
            first_duration = int(round(float(first_scene.get("duration_seconds", 0))))
            second_duration = int(round(float(second_scene.get("duration_seconds", 0))))
        except (TypeError, ValueError):
            return scenes

        if first_duration <= hook_max_seconds:
            return scenes

        first_narration = str(first_scene.get("narration", "")).strip()
        second_narration = str(second_scene.get("narration", "")).strip()
        sentences = self._split_sentences(first_narration)
        if len(sentences) < 2:
            return scenes

        sentence_word_counts = [len(sentence.split()) for sentence in sentences]
        total_words = max(1, sum(sentence_word_counts))
        moved_sentences: list[str] = []
        moved_words = 0

        while len(sentences) > 1 and first_duration > hook_max_seconds:
            next_sentence = sentences[-1]
            next_words = len(next_sentence.split())
            remaining_words = sum(len(sentence.split()) for sentence in sentences[:-1])
            if remaining_words < 20:
                break
            moved_sentences.insert(0, sentences.pop())
            moved_words += next_words
            shifted_seconds = max(
                1,
                int(round(first_duration * (moved_words / total_words))),
            )
            first_duration = max(12, first_duration - shifted_seconds)
            second_duration = min(120, second_duration + shifted_seconds)

        if not moved_sentences or first_duration > hook_max_seconds:
            return scenes

        first_scene["narration"] = " ".join(sentences).strip()
        second_scene["narration"] = " ".join(moved_sentences + ([second_narration] if second_narration else [])).strip()
        first_scene["duration_seconds"] = first_duration
        second_scene["duration_seconds"] = second_duration
        second_terms = list(second_scene.get("visual_search_terms", [])) if isinstance(second_scene.get("visual_search_terms"), list) else []
        for term in first_scene.get("visual_search_terms", []) if isinstance(first_scene.get("visual_search_terms"), list) else []:
            cleaned = str(term).strip()
            if cleaned and cleaned not in second_terms:
                second_terms.append(cleaned)
        if second_terms:
            second_scene["visual_search_terms"] = second_terms[:5]

        self._update(
            f"Auto-repaired opening hook by moving overflow from scene 1 into scene 2 (new hook target <= {hook_max_seconds}s)",
            None,
            "warning",
        )
        return repaired

    def _validate_scene_payload(self, scenes: object) -> list[dict]:
        if self.micro_test_live:
            return self._validate_micro_test_scene_payload(scenes)
        preset = self.DURATION_PRESETS[self.target_duration_minutes]
        min_scenes, max_scenes = preset["scene_range"]
        min_duration, max_duration = preset["duration_range"]
        if not isinstance(scenes, list):
            raise ScriptValidationError("Script must be a JSON array of scenes")
        hook_max_seconds = self._compute_hook_max_seconds(preset)
        scenes = self._repair_hook_scene_payload(scenes, hook_max_seconds)
        if not min_scenes <= len(scenes) <= max_scenes:
            raise ScriptValidationError(f"Expected {min_scenes}-{max_scenes} scenes, got {len(scenes)}")

        validated: list[dict] = []
        total_duration = 0

        for index, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                raise ScriptValidationError(f"Scene {index} is not an object")

            narration = str(scene.get("narration", "")).strip()
            if len(narration.split()) < 20:
                raise ScriptValidationError(f"Scene {index} narration is too short")

            try:
                duration_seconds = int(round(float(scene.get("duration_seconds", 0))))
            except (TypeError, ValueError):
                raise ScriptValidationError(f"Scene {index} has invalid duration_seconds")

            if duration_seconds < 12 or duration_seconds > 120:
                raise ScriptValidationError(
                    f"Scene {index} duration_seconds must be between 12 and 120"
                )

            raw_terms = scene.get("visual_search_terms")
            if not isinstance(raw_terms, list):
                raise ScriptValidationError(f"Scene {index} visual_search_terms must be a list")

            visual_terms: list[str] = []
            seen_terms = set()
            for term in raw_terms:
                cleaned = str(term).strip()
                normalized = cleaned.lower()
                if cleaned and normalized not in seen_terms:
                    seen_terms.add(normalized)
                    visual_terms.append(cleaned)

            if len(visual_terms) < 3:
                raise ScriptValidationError(f"Scene {index} must have at least 3 visual search terms")

            is_hook = bool(scene.get("is_hook", index == 1))
            if index == 1:
                is_hook = True
                if duration_seconds > hook_max_seconds:
                    raise ScriptValidationError(
                        f"Scene 1 hook is too long; expected max {hook_max_seconds} seconds"
                    )

            validated_scene = {
                "scene_number": index,
                "narration": narration,
                "duration_seconds": duration_seconds,
                "visual_search_terms": visual_terms[:5],
                "is_hook": is_hook,
            }
            validated.append(validated_scene)
            total_duration += duration_seconds

        if total_duration < min_duration or total_duration > max_duration:
            raise ScriptValidationError(
                f"Total script duration must stay between {min_duration} and {max_duration} seconds, got {total_duration}"
            )

        return validated

    def _validate_micro_test_scene_payload(self, scenes: object) -> list[dict]:
        if not isinstance(scenes, list):
            raise ScriptValidationError("Micro test script must be a JSON array of scenes")
        if len(scenes) != 1:
            raise ScriptValidationError(f"Micro test expects exactly 1 scene, got {len(scenes)}")

        scene = scenes[0]
        if not isinstance(scene, dict):
            raise ScriptValidationError("Micro test scene is not an object")

        narration = str(scene.get("narration", "")).strip()
        if len(narration.split()) < 12:
            raise ScriptValidationError("Micro test narration is too short")

        try:
            duration_seconds = int(round(float(scene.get("duration_seconds", 0))))
        except (TypeError, ValueError):
            raise ScriptValidationError("Micro test scene has invalid duration_seconds")

        if duration_seconds != self.MICRO_TEST_DURATION_SECONDS:
            raise ScriptValidationError(
                f"Micro test duration_seconds must be {self.MICRO_TEST_DURATION_SECONDS}"
            )

        raw_terms = scene.get("visual_search_terms")
        if not isinstance(raw_terms, list):
            raise ScriptValidationError("Micro test visual_search_terms must be a list")

        visual_terms: list[str] = []
        seen_terms = set()
        for term in raw_terms:
            cleaned = str(term).strip()
            normalized = cleaned.lower()
            if cleaned and normalized not in seen_terms:
                seen_terms.add(normalized)
                visual_terms.append(cleaned)

        if len(visual_terms) < 3:
            raise ScriptValidationError("Micro test scene must have at least 3 visual search terms")

        return [{
            "scene_number": 1,
            "narration": narration,
            "duration_seconds": duration_seconds,
            "visual_search_terms": visual_terms[:5],
            "is_hook": False,
        }]

    def _build_micro_test_scene_payload(self) -> list[dict]:
        theme_fragment = re.sub(r"\s+", " ", self.theme.strip()) if self.theme.strip() else "stormy lighthouse coast"
        narration = (
            f"This live micro test follows {theme_fragment} in a single continuous scene. "
            "Waves hit dark rocks, seabirds circle overhead, and the camera holds on the coastline "
            "long enough to test live voice-over, stock footage retrieval, contextual image search, overlays, and final rendering."
        )
        return [{
            "scene_number": 1,
            "narration": narration,
            "duration_seconds": self.MICRO_TEST_DURATION_SECONDS,
            "visual_search_terms": [
                "stormy lighthouse coast",
                "ocean waves rocks",
                "seabirds over sea",
                "dramatic coastline aerial",
            ],
            "is_hook": False,
        }]

    def _generate_validated_script(self, script_path: Path) -> list[dict]:
        generator = ScriptGenerator(target_duration_minutes=self.target_duration_minutes)
        last_error: ScriptValidationError | None = None

        for attempt in range(1, self.SCRIPT_GENERATION_MAX_ATTEMPTS + 1):
            feedback = str(last_error) if last_error else None
            scenes = generator.generate(self.theme, validation_feedback=feedback)
            try:
                validated = self._validate_scene_payload(scenes)
            except ScriptValidationError as exc:
                last_error = exc
                if attempt < self.SCRIPT_GENERATION_MAX_ATTEMPTS:
                    self._update(
                        f"Script attempt {attempt}/{self.SCRIPT_GENERATION_MAX_ATTEMPTS} failed validation: {exc}. Retrying...",
                        None,
                        "warning",
                    )
                    continue
                raise

            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(validated, f, ensure_ascii=False, indent=2)
            return validated

        if last_error is not None:
            raise last_error
        raise ScriptValidationError("Script generation failed without a validation error")

    def _hash_theme(self, theme: str) -> str:
        import hashlib
        cache_key = json.dumps({
            "theme": theme,
            "voice": self.voice,
            "quality": self.video_quality,
            "target_duration_minutes": self.target_duration_minutes,
            "clips_per_scene": self.clips_per_scene,
            "visual_mode": self.visual_mode,
            "scene_workers": self.scene_workers,
            "context_image_overlays": self.context_image_overlays,
            "micro_test_live": self.micro_test_live,
        }, sort_keys=True)
        return hashlib.md5(cache_key.encode()).hexdigest()

    def _slugify_theme(self, theme: str) -> str:
        normalized = unicodedata.normalize("NFKD", theme).encode("ascii", "ignore").decode("ascii")
        normalized = re.sub(r"[^A-Za-z0-9\s-]", "", normalized).strip().lower()
        normalized = re.sub(r"[-\s]+", "_", normalized)
        return (normalized[:40] or "documentary")

    def _update(self, message: str, progress: int, level: str = "info", details: dict | None = None):
        self.status_cb(message, progress, level, details)

    def _ensure_not_cancelled(self):
        if cancel_requested(self.project_dir):
            raise CancellationRequested("Pipeline stopped by user")

    def _asset_manifest_path(self) -> Path:
        return self.project_dir / "scene_assets.json"

    def _read_asset_manifest(self) -> dict[str, dict]:
        path = self._asset_manifest_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(key): value for key, value in data.items() if isinstance(value, dict)}

    def _write_asset_manifest(self, manifest: dict[str, dict]):
        path = self._asset_manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            temp_path = Path(handle.name)
        temp_path.replace(path)

    def _overlay_plan_path(self, scene_num: int) -> Path:
        return self.project_dir / "image_cache" / f"scene_{scene_num:02d}_overlay_plan.json"

    def _load_overlay_plan(self, scene_num: int) -> list[dict]:
        for candidate in (
            self._overlay_plan_path(scene_num),
            self.project_dir / "images" / f"scene_{scene_num:02d}_overlay_plan.json",
        ):
            if not candidate.exists():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        return []

    def _discover_existing_scene_assets(
        self,
        scene_num: int,
        scene: dict,
        *,
        audio_dir: Path,
        video_dir: Path,
        image_dir: Path,
        downloader: PexelsDownloader,
        image_downloader: OpenverseImageDownloader,
        render_validator: "VideoEditor",
        manifest_entry: dict | None = None,
    ) -> tuple[str, list[str], list[str], list[dict], bool]:
        audio_path = str(audio_dir / f"scene_{scene_num:02d}.mp3")
        if not Path(audio_path).exists():
            audio_path = ""

        target_clips = downloader.target_clip_count(scene.get("duration_seconds"))
        video_paths = [str(path) for path in downloader._collect_existing_scene_clips(video_dir, scene_num, target_clips)]

        overlay_plan = self._load_overlay_plan(scene_num)
        image_paths: list[str] = []

        if manifest_entry:
            for path in manifest_entry.get("image_paths", []):
                if path and Path(path).exists():
                    image_paths.append(str(Path(path)))
            if not overlay_plan:
                cached_plan = manifest_entry.get("overlay_plan")
                if isinstance(cached_plan, list):
                    overlay_plan = [item for item in cached_plan if isinstance(item, dict)]

        if not image_paths:
            for item in overlay_plan:
                path = str(item.get("image_path", "")).strip()
                if path and Path(path).exists():
                    image_paths.append(path)

        if not image_paths:
            target_images = image_downloader.target_image_count(scene.get("duration_seconds"))
            image_paths = [
                str(path) for path in image_downloader._collect_existing_scene_images(image_dir, scene_num, target_images)
            ]

        deduped_image_paths: list[str] = []
        seen_images = set()
        for path in image_paths:
            resolved = str(Path(path))
            if resolved in seen_images:
                continue
            seen_images.add(resolved)
            deduped_image_paths.append(resolved)
        image_paths = deduped_image_paths

        render_cache_path = self.project_dir / "render_cache" / f"scene_{scene_num:02d}.mp4"
        render_cache_ready = render_cache_path.exists() and render_validator._is_render_file_usable(render_cache_path)

        if manifest_entry and manifest_entry.get("completed"):
            overlay_plan = overlay_plan or []
            return audio_path, video_paths, image_paths, overlay_plan, bool(audio_path and (video_paths or render_cache_ready))

        sufficient_visuals = len(video_paths) >= target_clips or render_cache_ready
        if self.context_image_overlays:
            return audio_path, video_paths, image_paths, overlay_plan, bool(audio_path and sufficient_visuals)
        return audio_path, video_paths, image_paths, overlay_plan, bool(audio_path and sufficient_visuals)

    def _prepare_scenes_parallel(self, scenes: list[dict], audio_dir: Path, video_dir: Path, image_dir: Path) -> tuple[list[str], list[list[str]], list[list[str]]]:
        total = max(1, len(scenes))
        audio_files: list[str | None] = [None] * len(scenes)
        video_files: list[list[str] | None] = [None] * len(scenes)
        image_files: list[list[str] | None] = [None] * len(scenes)
        voice_done = 0
        visual_done = 0
        clips_done = 0
        clips_target_total = 0
        images_done = 0
        images_target_total = 0
        progress_lock = threading.Lock()
        asset_started_at = time.time()
        last_asset_emit_at = 0.0
        tts = VoiceOverGenerator(voice=self.voice)
        downloader = PexelsDownloader(
            quality=self.video_quality,
            clips_per_scene=self.clips_per_scene,
            visual_mode=self.visual_mode,
        )
        image_downloader = OpenverseImageDownloader(visual_mode=self.visual_mode)
        image_pipeline = DocumentaryImagePipeline(
            cache_dir=self.project_dir / "image_cache",
            target_images_fn=image_downloader.target_image_count,
        )
        manifest_lock = threading.Lock()
        asset_manifest = self._read_asset_manifest()
        video_editor = VideoEditor(self.project_dir)

        for scene in scenes:
            clips_target_total += downloader.target_clip_count(scene.get("duration_seconds"))
            images_target_total += image_downloader.target_image_count(scene.get("duration_seconds"))

        def emit_progress(message: str, *, force: bool = False):
            nonlocal last_asset_emit_at
            with progress_lock:
                voice_progress = STEP_RANGES["voice_over"][0] + int((voice_done / total) * (STEP_RANGES["voice_over"][1] - STEP_RANGES["voice_over"][0]))
                visual_progress = STEP_RANGES["beelden"][0] + int((visual_done / total) * (STEP_RANGES["beelden"][1] - STEP_RANGES["beelden"][0]))
                progress = max(voice_progress, visual_progress)
                completed_units = voice_done + clips_done + images_done
                total_units = total + max(1, clips_target_total) + max(1, images_target_total)
                eta_seconds = None
                if completed_units > 0:
                    elapsed = max(0.0, time.time() - asset_started_at)
                    remaining_units = max(0, total_units - completed_units)
                    eta_seconds = int((elapsed / completed_units) * remaining_units)
                details = {
                    "asset_status": {
                        "scenes_total": total,
                        "voice_done": voice_done,
                        "visual_done": visual_done,
                        "clips_downloaded": clips_done,
                        "clips_target": clips_target_total,
                        "images_downloaded": images_done,
                        "images_target": images_target_total,
                        "workers": self.scene_workers,
                        "eta_seconds": eta_seconds,
                        "updated_at": datetime.now().isoformat(),
                    }
                }
            now = time.time()
            if not force and (now - last_asset_emit_at) < self.ASSET_STATUS_INTERVAL_SECONDS:
                return
            last_asset_emit_at = now
            self._update(message, progress, details=details)

        def process_scene(index: int, scene: dict) -> tuple[int, str, list[str], list[str]]:
            nonlocal voice_done, visual_done, clips_done, images_done
            self._ensure_not_cancelled()
            scene_num = index + 1
            with manifest_lock:
                manifest_entry = asset_manifest.get(str(scene_num))
            out_path, paths, image_paths, overlay_plan, reusable = self._discover_existing_scene_assets(
                scene_num,
                scene,
                audio_dir=audio_dir,
                video_dir=video_dir,
                image_dir=image_dir,
                downloader=downloader,
                image_downloader=image_downloader,
                render_validator=video_editor,
                manifest_entry=manifest_entry,
            )
            scene["image_overlay_plan"] = overlay_plan

            if reusable:
                with progress_lock:
                    voice_done += 1
                    clips_done += len(paths)
                    images_done += len(image_paths)
                    visual_done += 1
                emit_progress(
                    f"  Scene {scene_num}/{len(scenes)} assets reused from cache ({len(paths)} clips, {len(image_paths)} images)",
                    force=True,
                )
                return index, out_path, paths, image_paths

            out_path = str(audio_dir / f"scene_{scene_num:02d}.mp3")
            if not Path(out_path).exists():
                self._update(f"  Generating voice-over scene {scene_num}/{len(scenes)}...", None)
                tts.generate_scene(scene["narration"], out_path)
            with progress_lock:
                voice_done += 1
            emit_progress(f"  Voice-over scene {scene_num}/{len(scenes)} ready", force=True)

            self._ensure_not_cancelled()
            target_clips = downloader.target_clip_count(scene.get("duration_seconds"))
            existing = downloader._collect_existing_scene_clips(video_dir, scene_num, target_clips)
            legacy = video_dir / f"scene_{scene_num:02d}_raw.mp4"
            paths = [str(path) for path in existing[:target_clips]]
            if len(paths) < target_clips:
                self._update(
                    f"  Downloading video scene {scene_num}/{len(scenes)} ({target_clips} unique clips target)...",
                    None,
                )
                downloaded = downloader.search_and_download(
                    scene.get("visual_search_terms", ["nature landscape", "sky clouds", "forest"]),
                    video_dir,
                    scene_num,
                    target_clips - len(paths),
                    cancel_check=self._ensure_not_cancelled,
                )
                paths.extend(downloaded)
                with progress_lock:
                    clips_done += len(downloaded)
                emit_progress(f"  Downloading video scene {scene_num}/{len(scenes)}...", force=False)
            if legacy.exists() and len(paths) >= target_clips:
                legacy.unlink()
            if legacy.exists() and not paths:
                paths = [str(legacy)]

            image_paths = []
            scene["image_overlay_plan"] = []
            if self.context_image_overlays:
                try:
                    image_paths, overlay_plan = image_pipeline.process_scene(
                        scene_num,
                        scene,
                        status_cb=self._update,
                    )
                except Exception as exc:
                    overlay_plan = []
                    image_paths = []
                    self._update(
                        f"  Scene {scene_num}/{len(scenes)} smart image retrieval failed: {exc}",
                        None,
                        "warning",
                    )
                scene["image_overlay_plan"] = overlay_plan

                if not image_paths:
                    target_images = image_downloader.target_image_count(scene.get("duration_seconds"))
                    image_paths, _ = image_downloader.search_and_download(
                        scene.get("visual_search_terms", ["historical illustration", "archival image", "ancient artwork"]),
                        image_dir,
                        scene_num,
                        target_images,
                        status_cb=self._update,
                    )
            with progress_lock:
                images_done += len(image_paths)

            with progress_lock:
                visual_done += 1
            with manifest_lock:
                asset_manifest[str(scene_num)] = {
                    "completed": True,
                    "audio_path": out_path,
                    "video_paths": paths,
                    "image_paths": image_paths,
                    "overlay_plan": scene.get("image_overlay_plan", []),
                    "updated_at": datetime.now().isoformat(),
                }
                self._write_asset_manifest(asset_manifest)
            emit_progress(f"  Scene {scene_num}/{len(scenes)} assets ready ({len(paths)} clips, {len(image_paths)} images)", force=True)
            return index, out_path, paths, image_paths

        with ThreadPoolExecutor(max_workers=self.scene_workers) as executor:
            futures = [executor.submit(process_scene, index, scene) for index, scene in enumerate(scenes)]
            for future in as_completed(futures):
                self._ensure_not_cancelled()
                index, out_path, paths, image_paths = future.result()
                audio_files[index] = out_path
                video_files[index] = paths
                image_files[index] = image_paths

        return [path or "" for path in audio_files], [paths or [] for paths in video_files], [paths or [] for paths in image_files]

    def run(self) -> dict:
        try:
            script_range = STEP_RANGES["script"]
            voice_range = STEP_RANGES["voice_over"]
            beeld_range = STEP_RANGES["beelden"]
            montage_range = STEP_RANGES["montage"]
            export_range = STEP_RANGES["export"]

            # ── Check cache ────────────────────────────────────────────────────
            existing = self.db.project_exists(self.theme_hash)
            if existing and Path(existing["output_path"]).exists():
                self._update("Identical project found in cache", 100, "success")
                return {
                    "success": True,
                    "output_path": existing["output_path"],
                    "project_name": existing["project_name"],
                    "duration": f"{existing['duration_seconds']:.0f}s",
                    "file_size_mb": existing["file_size_mb"],
                    "from_cache": True,
                }

            self.db.register_project(self.project_name, self.theme_hash)

            # Sla config op zodat resume mogelijk is
            config_path = self.project_dir / "config.json"
            if not config_path.exists():
                with open(config_path, "w") as f:
                    json.dump({
                        "theme": self.theme,
                        "voice": self.voice,
                        "quality": self.video_quality,
                        "target_duration_minutes": self.target_duration_minutes,
                        "clips_per_scene": self.clips_per_scene,
                        "visual_mode": self.visual_mode,
                        "scene_workers": self.scene_workers,
                        "context_image_overlays": self.context_image_overlays,
                        "micro_test_live": self.micro_test_live,
                    }, f)

            # ── Stap 1: Script Genereren ───────────────────────────────────────
            self._ensure_not_cancelled()
            if self.micro_test_live:
                self._update("Preparing live micro test scene...", script_range[0] + 5)
            else:
                self._update("Generating script with AI...", script_range[0] + 5)
            script_path = self.project_dir / "script.json"

            if script_path.exists():
                try:
                    with open(script_path) as f:
                        scenes = self._validate_scene_payload(json.load(f))
                    self._update("Script loaded from cache", script_range[1], "success")
                except ScriptValidationError as exc:
                    script_path.unlink(missing_ok=True)
                    self._update(f"Cached script failed validation: {exc}. Regenerating...", None, "warning")
                    if self.micro_test_live:
                        scenes = self._validate_scene_payload(self._build_micro_test_scene_payload())
                        with open(script_path, "w", encoding="utf-8") as f:
                            json.dump(scenes, f, ensure_ascii=False, indent=2)
                        self._update("Live micro test scene prepared", script_range[1], "success")
                    else:
                        scenes = self._generate_validated_script(script_path)
                        self._update(f"Script generated: {len(scenes)} scenes", script_range[1], "success")
            else:
                if self.micro_test_live:
                    scenes = self._validate_scene_payload(self._build_micro_test_scene_payload())
                    with open(script_path, "w", encoding="utf-8") as f:
                        json.dump(scenes, f, ensure_ascii=False, indent=2)
                    self._update("Live micro test scene prepared", script_range[1], "success")
                else:
                    scenes = self._generate_validated_script(script_path)
                    self._update(f"Script generated: {len(scenes)} scenes", script_range[1], "success")
            total_duration = sum(s.get("duration_seconds", 60) for s in scenes)

            # ── Stap 2-3: Scene assets parallel voorbereiden ──────────────────
            self._ensure_not_cancelled()
            self._update(f"Preparing scene assets with {self.scene_workers} workers...", voice_range[0])
            audio_dir = self.project_dir / "audio"
            video_dir = self.project_dir / "videos"
            image_dir = self.project_dir / "images"
            audio_files, video_files, image_files = self._prepare_scenes_parallel(scenes, audio_dir, video_dir, image_dir)
            self._update(f"{len(audio_files)} voice-overs ready", voice_range[1], "success")
            found = sum(1 for v in video_files if v)
            self._update(f"{found}/{len(scenes)} scenes have source video", beeld_range[1], "success")

            # ── Stap 4: Video Monteren ─────────────────────────────────────────
            self._ensure_not_cancelled()
            self._update("Editing video...", montage_range[0])
            output_path = str(self.project_dir / "final_video.mp4")
            music_dir = os.getenv("MUSIC_DIR", None)
            if music_dir and not Path(music_dir).is_absolute():
                music_dir = str((self.base_dir / music_dir).resolve())

            editor = VideoEditor(self.project_dir, music_dir=music_dir)
            editor.build(
                scenes=scenes,
                audio_files=audio_files,
                video_files=video_files,
                image_files=image_files,
                output_path=output_path,
                status_cb=self._update,
                progress_ranges={
                    "montage": montage_range,
                    "export": export_range,
                },
                cancel_check=self._ensure_not_cancelled,
            )
            total_frames = int(total_duration * 24)
            self._update(
                "Video editing complete",
                export_range[1],
                "success",
                {"export_status": {
                    "progress_percent": 100,
                    "current_frame": total_frames,
                    "total_frames": total_frames,
                    "elapsed_seconds": None,
                    "eta_seconds": 0,
                }}
            )

            # ── Stap 5: Cleanup & Afronden ─────────────────────────────────────
            if self.cleanup:
                self._ensure_not_cancelled()
                self._update("Removing temporary files...", 96)
                cleaner = CleanupManager(self.project_dir)
                removed = cleaner.cleanup()
                self._update(f"Removed {removed} temporary files", 98, "success")

            # Metadata opslaan
            output_file = Path(output_path)
            size_mb = output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0

            self.db.complete_project(
                self.project_name, output_path,
                duration=total_duration, size_mb=size_mb
            )

            # Metadata JSON
            metadata = {
                "project_name": self.project_name,
                "theme": self.theme,
                "created_at": datetime.now().isoformat(),
                "scenes": len(scenes),
                "total_duration_seconds": total_duration,
                "output_path": output_path,
                "file_size_mb": round(size_mb, 2),
                "voice": self.voice,
                "target_duration_minutes": self.target_duration_minutes,
                "micro_test_live": self.micro_test_live,
            }
            with open(self.project_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            self._update("Documentary generated successfully", 100, "success")

            return {
                "success": True,
                "output_path": output_path,
                "project_name": self.project_name,
                "duration": f"{int(total_duration // 60)}m {int(total_duration % 60)}s",
                "file_size_mb": size_mb,
                "script_path": str(script_path),
            }

        except CancellationRequested as e:
            self.db.cancel_project(self.project_name)
            self._update_no_progress(str(e), level="info")
            return {
                "success": False,
                "cancelled": True,
                "error": str(e),
            }
        except Exception as e:
            self.db.fail_project(self.project_name, str(e))
            self._update_no_progress(f"Error: {str(e)}", level="error")
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _update_no_progress(self, message: str, _progress=None, level: str = "info", details: dict | None = None):
        """Status update zonder voortgangsbalk te wijzigen."""
        self.status_cb(message, None, level, details)

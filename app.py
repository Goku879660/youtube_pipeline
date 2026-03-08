"""
app.py — Streamlit UI voor de YouTube Documentary Pipeline
Lanceer met: streamlit run app.py
"""

import streamlit as st
import json
import os
import hashlib
import inspect
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path
from dotenv import load_dotenv, set_key
from video_engine import DocumentaryPipeline
from pipeline_status import (
    STEP_KEYS,
    STEP_LABELS,
    claim_lock,
    compute_step_progress,
    is_lock_active,
    normalize_runtime_status,
    request_cancel,
    read_status,
    release_lock,
    terminate_process,
    write_status,
)

load_dotenv()

# ── Step definitions ───────────────────────────────────────────────────────────
PIPELINE_STEPS = ["Script", "Voice-over", "Visuals", "Editing", "Export"]


def display_project_name(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 4 and len(parts[0]) == 10 and len(parts[1]) == 6:
        return "_".join(parts[2:-1]) or name
    if len(name) > 11 and name[4] == "-" and name[7] == "-":
        return name[11:]
    return name


def slugify_theme(theme: str) -> str:
    normalized = unicodedata.normalize("NFKD", theme).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^A-Za-z0-9\s-]", "", normalized).strip().lower()
    normalized = re.sub(r"[-\s]+", "_", normalized)
    return normalized[:40] or "documentary"


def is_openai_tts_voice(voice: str) -> bool:
    return str(voice).startswith("openai:")


def display_tts_voice(voice: str) -> str:
    if is_openai_tts_voice(voice):
        return f"OpenAI {voice.split(':', 1)[1]}"
    return voice


def compute_theme_hash(theme: str, voice: str, quality: str, target_duration_minutes: int, clips_per_scene: int, visual_mode: str, scene_workers: int, context_image_overlays: bool, micro_test_live: bool) -> str:
    payload = json.dumps({
        "theme": theme,
        "voice": voice,
        "quality": quality,
        "target_duration_minutes": target_duration_minutes,
        "clips_per_scene": max(1, clips_per_scene),
        "visual_mode": visual_mode,
        "scene_workers": clamp_scene_workers(scene_workers),
        "context_image_overlays": bool(context_image_overlays),
        "micro_test_live": bool(micro_test_live),
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def is_meaningful_project_dir(project_dir: Path) -> bool:
    return any((project_dir / name).exists() for name in ("status.json", "config.json", "script.json", "final_video.mp4", "worker.log"))


def find_matching_resume_project(theme: str, voice: str, quality: str, target_duration_minutes: int, clips_per_scene: int, visual_mode: str, scene_workers: int, context_image_overlays: bool, micro_test_live: bool) -> Path | None:
    if not PROJECTS_DIR.exists():
        return None
    expected_hash = compute_theme_hash(theme, voice, quality, target_duration_minutes, clips_per_scene, visual_mode, scene_workers, context_image_overlays, micro_test_live)[:8]
    expected_slug = slugify_theme(theme)
    for project_dir in sorted(PROJECTS_DIR.iterdir(), reverse=True):
        if not project_dir.is_dir() or not is_meaningful_project_dir(project_dir):
            continue
        if (project_dir / "final_video.mp4").exists():
            continue
        if not project_dir.name.endswith(expected_hash):
            continue
        name_parts = project_dir.name.split("_")
        project_slug = "_".join(name_parts[2:-1]) if len(name_parts) >= 4 else ""
        if project_slug == expected_slug:
            return project_dir
    return None


def delete_project_dir(project_dir: str | Path):
    path = Path(project_dir)
    if not path.exists():
        return
    status = normalize_runtime_status(path, read_status(path))
    if status and status.get("status") in {"queued", "running"}:
        request_cancel(path, "Delete requested")
        terminate_process(status.get("pid"))
    shutil.rmtree(path, ignore_errors=True)


def cleanup_empty_project_dirs() -> int:
    removed = 0
    if not PROJECTS_DIR.exists():
        return removed
    for project_dir in PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        if is_meaningful_project_dir(project_dir):
            continue
        shutil.rmtree(project_dir, ignore_errors=True)
        removed += 1
    return removed


def get_resume_settings_mismatch(project: dict, *, voice: str, quality: str, target_duration_minutes: int, clips_per_scene: int, visual_mode: str, scene_workers: int, context_image_overlays: bool, micro_test_live: bool) -> str | None:
    comparisons = [
        ("voice", project.get("voice"), voice, "voice"),
        ("quality", project.get("quality"), quality, "quality"),
        ("target_duration_minutes", int(project.get("target_duration_minutes", 10)), int(target_duration_minutes), "video length"),
        ("clips_per_scene", int(project.get("clips_per_scene", 4)), int(clips_per_scene), "shot density"),
        ("visual_mode", project.get("visual_mode", "standard"), visual_mode, "storage mode"),
        ("scene_workers", clamp_scene_workers(project.get("scene_workers", 3)), clamp_scene_workers(scene_workers), "scene workers"),
        ("context_image_overlays", bool(project.get("context_image_overlays", True)), bool(context_image_overlays), "context image overlays"),
        ("micro_test_live", bool(project.get("micro_test_live", False)), bool(micro_test_live), "micro test mode"),
    ]
    mismatches = [label for _, original, current, label in comparisons if original != current]
    if not mismatches:
        return None
    return ", ".join(mismatches)

def detect_incomplete_projects() -> list[dict]:
    projects_dir = Path("projects")
    incomplete = []
    if not projects_dir.exists():
        return incomplete
    for p in sorted(projects_dir.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        if not is_meaningful_project_dir(p):
            continue
        status = normalize_runtime_status(p, read_status(p))
        if status and status.get("status") in {"queued", "running"}:
            continue
        script = p / "script.json"
        final = p / "final_video.mp4"
        if not script.exists() or final.exists():
            continue
        theme = None
        voice = "en-US-GuyNeural"
        quality = "hd"
        target_duration_minutes = 10
        clips_per_scene = 4
        visual_mode = "standard"
        scene_workers = 3
        context_image_overlays = True
        micro_test_live = False
        config = p / "config.json"
        if config.exists():
            try:
                cfg = json.loads(config.read_text())
                theme = cfg.get("theme")
                voice = cfg.get("voice", voice)
                quality = cfg.get("quality", quality)
                target_duration_minutes = int(cfg.get("target_duration_minutes", target_duration_minutes))
                clips_per_scene = int(cfg.get("clips_per_scene", clips_per_scene))
                visual_mode = cfg.get("visual_mode", visual_mode)
                scene_workers = clamp_scene_workers(cfg.get("scene_workers", scene_workers))
                context_image_overlays = bool(cfg.get("context_image_overlays", context_image_overlays))
                micro_test_live = bool(cfg.get("micro_test_live", micro_test_live))
            except Exception:
                pass
        audio_dir = p / "audio"
        video_dir = p / "videos"
        audio_count = len(list(audio_dir.glob("scene_*.mp3"))) if audio_dir.exists() else 0
        if video_dir.exists():
            clip_scenes = {path.name.split("_clip_")[0] for path in video_dir.glob("scene_*_clip_*.mp4")}
            raw_scenes = {path.name.replace("_raw.mp4", "") for path in video_dir.glob("scene_*_raw.mp4")}
            video_count = len(clip_scenes | raw_scenes)
        else:
            video_count = 0
        try:
            scenes = json.loads(script.read_text())
            total = len(scenes)
        except Exception:
            total = "?"
        if video_count and isinstance(total, int) and video_count >= total:
            last_step, step_num = "Footage downloaded", 3
        elif audio_count and isinstance(total, int) and audio_count >= total:
            last_step, step_num = "Voice-overs ready", 2
        elif audio_count:
            last_step, step_num = f"Voice-over {audio_count}/{total}", 2
        else:
            last_step, step_num = "Script generated", 1
        incomplete.append({
            "project_dir": str(p),
            "name": p.name,
            "theme": theme,
            "voice": voice,
            "quality": quality,
            "target_duration_minutes": target_duration_minutes,
            "clips_per_scene": clips_per_scene,
            "visual_mode": visual_mode,
            "scene_workers": scene_workers,
            "context_image_overlays": context_image_overlays,
            "micro_test_live": micro_test_live,
            "last_step": last_step,
            "step_num": step_num,
            "audio_count": audio_count,
            "video_count": video_count,
            "total": total,
        })
    return incomplete

st.set_page_config(
    page_title="Documentary Generator",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp { background-color: #0a0a0a; }
section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #1e1e1e; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 1100px; }

/* Typography */
h1, h2, h3 { font-weight: 500 !important; letter-spacing: -0.02em; color: #f5f5f5 !important; }
p, li, span, label { color: #888 !important; }

/* Sidebar labels */
.stSidebar label { color: #555 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background-color: #141414 !important;
    color: #e8e8e8 !important;
    border: 1px solid #222 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #333 !important;
    box-shadow: none !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #141414 !important;
    border: 1px solid #222 !important;
    border-radius: 8px !important;
    color: #e8e8e8 !important;
}

/* Primary button */
.stButton > button[kind="primary"], .stButton > button {
    background: #f5f5f5 !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.6rem 1.5rem !important;
    letter-spacing: 0.01em;
    transition: opacity 0.15s ease !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Divider */
hr { border-color: #1a1a1a !important; margin: 1.5rem 0 !important; }

/* Log cards */
.log-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 6px;
    margin: 5px 0;
    font-size: 13px;
    font-family: 'Inter', monospace;
    background: #111;
    border: 1px solid #1e1e1e;
    color: #999 !important;
}
.log-item.success { border-color: #1a3a2a; color: #4ade80 !important; background: #0d1f16; }
.log-item.error   { border-color: #3a1a1a; color: #f87171 !important; background: #1f0d0d; }
.log-dot { width: 6px; height: 6px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; background: #444; }
.log-dot.success { background: #4ade80; }
.log-dot.error   { background: #f87171; }

/* Metric cards */
.metric-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: left;
}
.metric-label { font-size: 11px; color: #555 !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.metric-value { font-size: 22px; font-weight: 600; color: #f5f5f5 !important; }

/* API key status dot */
.key-status { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }
.key-ok  { background: #4ade80; }
.key-err { background: #f87171; }

/* Progress bar */
.stProgress > div > div { background: #f5f5f5 !important; border-radius: 4px; }
.stProgress > div { background: #1e1e1e !important; border-radius: 4px; }

/* Expander */
.streamlit-expanderHeader { background: #111 !important; border: 1px solid #1e1e1e !important; border-radius: 8px !important; color: #888 !important; }
details { border: none !important; }

/* Checkbox */
.stCheckbox label { color: #666 !important; font-size: 13px !important; }

/* Pulse animation for the active step */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Resume card */
.resume-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
.resume-card:hover { border-color: #2a2a2a; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
ENV_PATH = Path("/home/florens/youtube_pipeline/.env")
BASE_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = BASE_DIR / "projects"
DEFAULT_POLL_INTERVAL_SECONDS = 5
UI_SECRET_EDITING_ALLOWED = os.getenv("ALLOW_UI_SECRET_EDITING", "1") == "1"


def clamp_scene_workers(value: int | str | None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 3
    return max(1, min(4, parsed))

def save_env_key(key: str, value: str):
    set_key(str(ENV_PATH), key, value)
    os.environ[key] = value

def key_set(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def sidebar_secret_input(env_key: str, label: str, placeholder: str):
    ok = key_set(env_key)
    dot = f'<span class="key-status key-{"ok" if ok else "err"}"></span>'
    st.markdown(
        f"<p style='font-size:11px;color:#555;margin:6px 0 4px'>{dot}{label}</p>",
        unsafe_allow_html=True,
    )
    current_value = os.getenv(env_key, "")
    if UI_SECRET_EDITING_ALLOWED:
        new_value = st.text_input(
            env_key.lower(),
            value=current_value,
            type="password",
            label_visibility="collapsed",
            placeholder=placeholder,
        )
        if new_value and new_value != current_value:
            save_env_key(env_key, new_value)
            st.rerun()
    else:
        masked = "Configured" if current_value else "Not configured"
        st.text_input(
            env_key.lower(),
            value=masked,
            disabled=True,
            label_visibility="collapsed",
        )


def format_seconds(seconds: int | None) -> str:
    if seconds is None:
        return "Calculating..."
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def launch_pipeline_job(theme: str, voice: str, quality: str, target_duration_minutes: int, cleanup: bool, clips_per_scene: int, visual_mode: str, scene_workers: int, context_image_overlays: bool, micro_test_live: bool, resume_dir: str | None = None) -> Path:
    active_status = get_active_pipeline_status()
    if active_status:
        active_project_dir = str(active_status.get("project_dir", ""))
        if not resume_dir or Path(resume_dir).resolve() != Path(active_project_dir).resolve():
            active_name = display_project_name(active_status.get("project_name", "active project"))
            raise RuntimeError(f"Another pipeline is already running: {active_name}")
    if resume_dir is None:
        match = find_matching_resume_project(theme, voice, quality, target_duration_minutes, clips_per_scene, visual_mode, scene_workers, context_image_overlays, micro_test_live)
        if match is not None:
            resume_dir = str(match)
    pipeline_kwargs = {
        "theme": theme,
        "voice": voice,
        "video_quality": quality,
        "target_duration_minutes": target_duration_minutes,
        "cleanup": cleanup,
        "clips_per_scene": clips_per_scene,
        "visual_mode": visual_mode,
        "scene_workers": scene_workers,
        "resume_project_dir": resume_dir,
        "micro_test_live": micro_test_live,
    }
    if "context_image_overlays" in inspect.signature(DocumentaryPipeline).parameters:
        pipeline_kwargs["context_image_overlays"] = context_image_overlays
    bootstrap = DocumentaryPipeline(**pipeline_kwargs)
    project_dir = bootstrap.project_dir
    bootstrap.db.conn.close()
    lock_data = claim_lock(project_dir, owner="launcher")

    write_status(project_dir, {
        "project_dir": str(project_dir),
        "project_name": project_dir.name,
        "theme": theme,
        "voice": voice,
        "quality": quality,
        "target_duration_minutes": target_duration_minutes,
        "clips_per_scene": clips_per_scene,
        "visual_mode": visual_mode,
        "scene_workers": clamp_scene_workers(scene_workers),
        "context_image_overlays": bool(context_image_overlays),
        "micro_test_live": bool(micro_test_live),
        "status": "queued",
        "progress": 0,
        "active_step": STEP_KEYS[0],
        "step_progress": compute_step_progress(0),
        "last_message": "Waiting for worker start",
        "logs": [],
        "cancel_requested": False,
    })

    cmd = [
        sys.executable,
        str(BASE_DIR / "pipeline_worker.py"),
        "--theme", theme,
        "--voice", voice,
        "--quality", quality,
        "--target-duration-minutes", str(target_duration_minutes),
        "--project-dir", str(project_dir),
        "--lock-token", lock_data["token"],
        "--cleanup", "1" if cleanup else "0",
        "--clips-per-scene", str(clips_per_scene),
        "--visual-mode", visual_mode,
        "--scene-workers", str(clamp_scene_workers(scene_workers)),
        "--context-image-overlays", "1" if context_image_overlays else "0",
        "--micro-test-live", "1" if micro_test_live else "0",
    ]
    log_path = project_dir / "worker.log"
    try:
        with log_path.open("ab") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
        threading.Thread(
            target=process.wait,
            name=f"worker-reaper-{project_dir.name}",
            daemon=True,
        ).start()
    except Exception:
        release_lock(project_dir, lock_data["token"])
        raise
    return project_dir


def get_project_statuses() -> list[dict]:
    statuses = []
    if not PROJECTS_DIR.exists():
        return statuses
    for project_dir in sorted(PROJECTS_DIR.iterdir(), reverse=True):
        if not project_dir.is_dir():
            continue
        if not is_meaningful_project_dir(project_dir):
            continue
        status = normalize_runtime_status(project_dir, read_status(project_dir))
        if not status:
            continue
        status["project_dir"] = str(project_dir)
        if status != read_status(project_dir):
            write_status(project_dir, status)
        statuses.append(status)
    statuses.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return statuses


def get_focus_status() -> dict | None:
    active_dir = st.session_state.get("active_project_dir")
    if active_dir:
        status = normalize_runtime_status(active_dir, read_status(active_dir))
        if status:
            status["project_dir"] = active_dir
            if status != read_status(active_dir):
                write_status(active_dir, status)
            return status
    statuses = get_project_statuses()
    if not statuses:
        return None
    for status in statuses:
        if status.get("status") in {"queued", "running"}:
            return status
    return statuses[0]


def get_active_pipeline_status() -> dict | None:
    for status in get_project_statuses():
        if status.get("status") in {"queued", "running"}:
            return status
    return None


def render_job_panel(status: dict):
    running = status.get("status") in {"queued", "running"}
    progress = int(status.get("progress", 0))
    step_progress = status.get("step_progress") or compute_step_progress(progress)
    display_name = display_project_name(status.get("project_name", "onbekend"))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#444;margin-bottom:10px'>Pipeline status</p>", unsafe_allow_html=True)
    st.markdown(f"**{display_name}**")
    st.caption(status.get("last_message", ""))
    st.progress(progress / 100)
    st.caption(f"Total: {progress}%")

    export_status = status.get("export_status") or {}
    if status.get("active_step") == "export" or export_status:
        export_pct = int(export_status.get("progress_percent", 0))
        current_frame = export_status.get("current_frame")
        total_frames = export_status.get("total_frames")
        eta_seconds = export_status.get("eta_seconds")
        st.markdown("<p style='font-size:12px;color:#f5f5f5;margin:14px 0 4px'>Export</p>", unsafe_allow_html=True)
        st.progress(export_pct / 100 if export_pct else 0)
        frame_text = "Frame -- / --"
        if current_frame is not None and total_frames:
            frame_text = f"Frame {current_frame} / {total_frames}"
        eta_text = format_seconds(eta_seconds)
        st.caption(f"Export: {export_pct}% · {frame_text} · Estimated time remaining: {eta_text}")

    asset_status = status.get("asset_status") or {}
    if asset_status and status.get("active_step") in {"voice_over", "beelden"}:
        voice_done = int(asset_status.get("voice_done", 0))
        visual_done = int(asset_status.get("visual_done", 0))
        scenes_total = max(1, int(asset_status.get("scenes_total", 1)))
        clips_downloaded = int(asset_status.get("clips_downloaded", 0))
        clips_target = max(1, int(asset_status.get("clips_target", 1)))
        images_downloaded = int(asset_status.get("images_downloaded", 0))
        images_target_raw = int(asset_status.get("images_target", 0))
        workers = int(asset_status.get("workers", 0))
        eta_seconds = asset_status.get("eta_seconds")
        prep_pct = max(
            int((voice_done / scenes_total) * 50),
            50 + int((visual_done / scenes_total) * 50),
        )
        st.markdown("<p style='font-size:12px;color:#f5f5f5;margin:14px 0 4px'>Scene Prep</p>", unsafe_allow_html=True)
        st.progress(prep_pct / 100)
        image_text = f" · Images: {images_downloaded}/{images_target_raw}" if images_target_raw > 0 else ""
        st.caption(
            f"Voice-overs: {voice_done}/{scenes_total} · "
            f"Visuals: {visual_done}/{scenes_total} · "
            f"Clips: {clips_downloaded}/{clips_target}"
            f"{image_text} · "
            f"Workers: {workers} · "
            f"Estimated time remaining: {format_seconds(eta_seconds)}"
        )

    for key in STEP_KEYS:
        pct = int(step_progress.get(key, 0))
        st.markdown(f"<p style='font-size:12px;color:#777;margin:10px 0 4px'>{STEP_LABELS[key]} · {pct}%</p>", unsafe_allow_html=True)
        st.progress(pct / 100)

    logs = status.get("logs", [])[-12:]
    if logs:
        html = ""
        for item in logs:
            level = item.get("level", "info")
            dot_cls = "success" if level == "success" else ("error" if level == "error" else "")
            html += (
                f'<div class="log-item {dot_cls}">'
                f'<span class="log-dot {dot_cls}"></span>'
                f'<span>{item.get("message", "")}</span>'
                f'</div>'
            )
        st.markdown(html, unsafe_allow_html=True)

    result = status.get("result") or {}
    if status.get("status") == "completed" and result.get("success"):
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#444;margin-bottom:16px'>Resultaat</p>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, label, value in [
            (c1, "Project", result.get("project_name", status.get("project_name", "—"))),
            (c2, "Duration", result.get("duration", "—")),
            (c3, "File size", f"{result.get('file_size_mb', 0):.1f} MB"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        st.markdown(f"<p style='font-size:12px;color:#444;margin-top:12px'>Saved to <code style='background:#111;padding:2px 6px;border-radius:4px;color:#888'>{result.get('output_path', '—')}</code></p>", unsafe_allow_html=True)
    elif status.get("status") == "cancelled":
        st.markdown(f"<p style='color:#facc15;font-size:13px'>Pipeline stopped: {status.get('last_message', 'stop requested')}</p>", unsafe_allow_html=True)
    elif status.get("status") == "failed":
        st.markdown(f"<p style='color:#f87171;font-size:13px'>Pipeline failed: {status.get('last_message', 'unknown error')}</p>", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='padding: 8px 0 20px'><span style='font-size:13px;font-weight:600;color:#f5f5f5;letter-spacing:0.02em'>DOCUMENTARY GENERATOR</span></div>", unsafe_allow_html=True)

    # ── API Keys ──────────────────────────────────────────────────────────────
    st.markdown("<p style='font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#444;margin-bottom:10px'>API Keys</p>", unsafe_allow_html=True)
    st.caption("This sidebar writes secrets to the local `.env` file. Use only on a single-user local machine.")
    if not UI_SECRET_EDITING_ALLOWED:
        st.caption("Sidebar secret editing is disabled. Set `ALLOW_UI_SECRET_EDITING=1` only for local-only use.")

    # AI provider toggle
    current_provider = os.getenv("AI_PROVIDER", "openai").lower()
    provider = st.radio(
        "AI Provider",
        ["openai", "anthropic"],
        index=0 if current_provider == "openai" else 1,
        horizontal=True,
        label_visibility="collapsed",
        disabled=not UI_SECRET_EDITING_ALLOWED,
    )
    if provider != current_provider and UI_SECRET_EDITING_ALLOWED:
        save_env_key("AI_PROVIDER", provider)

    if provider == "openai":
        sidebar_secret_input("OPENAI_API_KEY", "OpenAI API Key", "sk-proj-...")
    else:
        sidebar_secret_input("ANTHROPIC_API_KEY", "Anthropic API Key", "sk-ant-...")

    sidebar_secret_input("PEXELS_API_KEY", "Pexels API Key", "Pexels key...")
    sidebar_secret_input("OPENVERSE_ACCESS_TOKEN", "Openverse Access Token", "Openverse bearer token...")
    sidebar_secret_input("OPENVERSE_CLIENT_ID", "Openverse Client ID", "Openverse client id...")
    sidebar_secret_input("OPENVERSE_CLIENT_SECRET", "Openverse Client Secret", "Openverse client secret...")
    st.caption("Openverse image search works anonymously too; add a token only if you need higher limits.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Output settings ───────────────────────────────────────────────────────
    st.markdown("<p style='font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#444;margin-bottom:10px'>Settings</p>", unsafe_allow_html=True)

    tts_engine = st.selectbox(
        "Voice engine",
        ["edge-tts", "OpenAI TTS"],
        index=0,
    )

    if tts_engine == "OpenAI TTS":
        openai_voice = st.selectbox(
            "OpenAI voice",
            ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"],
            index=0,
        )
        voice_option = f"openai:{openai_voice}"
        st.caption("Uses the OpenAI audio speech API with the selected built-in voice.")
    else:
        voice_option = st.selectbox(
            "Voice",
            ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-GB-RyanNeural", "en-AU-WilliamNeural"],
            index=0,
        )

    video_quality = st.selectbox(
        "Quality",
        ["hd", "sd"],
        index=0,
    )

    duration_label = st.selectbox(
        "Video length",
        ["3 minutes (test)", "5 minutes", "10 minutes", "15 minutes"],
        index=1,
    )
    target_duration_minutes = {
        "3 minutes (test)": 2,
        "5 minutes": 5,
        "10 minutes": 10,
        "15 minutes": 15,
    }[duration_label]

    shot_density_label = st.selectbox(
        "Shot density",
        ["Smart dynamic (4 clips/scene)", "Max dynamic (6 clips/scene)", "Calm (1 clip/scene)"],
        index=0,
    )
    clips_per_scene = {
        "Smart dynamic (4 clips/scene)": 4,
        "Max dynamic (6 clips/scene)": 6,
        "Calm (1 clip/scene)": 1,
    }[shot_density_label]

    visual_mode_label = st.selectbox(
        "Storage mode",
        ["Standard variety", "Light mode"],
        index=0,
    )
    visual_mode = {
        "Standard variety": "standard",
        "Light mode": "light",
    }[visual_mode_label]

    scene_worker_options = [1, 2, 3, 4]
    default_scene_workers = clamp_scene_workers(os.getenv("SCENE_WORKERS", 3))
    scene_workers = st.selectbox(
        "Scene workers",
        scene_worker_options,
        index=scene_worker_options.index(default_scene_workers),
    )
    context_image_overlays = st.checkbox(
        "Context image overlays",
        value=True,
        help="Show contextual still images over the video when the narration mentions specific people, places, objects, maps or historical concepts.",
    )
    micro_test_live = st.checkbox(
        "Micro test live",
        value=False,
        help="Run a real 15-second single-scene integration test with live TTS, stock video and image retrieval.",
    )

    st.caption("Set `Scene workers` to 1 to disable parallel asset preparation and run in a safer single-worker mode.")
    st.caption("Context image overlays place Openverse images over roughly 20-40% of the frame in content-heavy scenes.")
    st.caption("Micro test live bypasses AI script generation and uses one fixed 15-second scene while still calling the live voice, video and image providers.")
    st.caption("Temporary stock videos, scene audio and downloaded still images are always removed after a successful render.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Recent projects ───────────────────────────────────────────────────────
    st.markdown("<p style='font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#444;margin-bottom:10px'>Recent projects</p>", unsafe_allow_html=True)

    projects_dir = Path("projects")
    if projects_dir.exists():
        projects = [p for p in sorted(projects_dir.iterdir(), reverse=True) if p.is_dir() and is_meaningful_project_dir(p)][:5]
        if projects:
            for p in projects:
                final = p / "final_video.mp4"
                dot_color = "#4ade80" if final.exists() else "#555"
                name = display_project_name(p.name)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid #1a1a1a">'
                    f'<span style="width:6px;height:6px;border-radius:50%;background:{dot_color};flex-shrink:0;display:inline-block"></span>'
                    f'<span style="font-size:12px;color:#555;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{name}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown("<p style='font-size:12px;color:#333'>No projects</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size:12px;color:#333'>No projects</p>", unsafe_allow_html=True)

    empty_project_count = 0
    if projects_dir.exists():
        empty_project_count = sum(
            1 for p in projects_dir.iterdir()
            if p.is_dir() and not is_meaningful_project_dir(p)
        )
    if empty_project_count:
        st.caption(f"{empty_project_count} empty project folders detected.")
        if st.button("Cleanup empty folders", use_container_width=True):
            cleanup_empty_project_dirs()
            st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size:28px;margin-bottom:4px'>Documentary Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#444;font-size:14px;margin-bottom:2rem'>From one prompt to a full 9-minute documentary</p>", unsafe_allow_html=True)

# ── Prompt input ───────────────────────────────────────────────────────────────
theme_prompt = st.text_area(
    label="Theme",
    placeholder=(
        "Describe your documentary topic...\n\n"
        "Example: '5 forgotten battles of the Hundred Years' War, told from the perspective of ordinary soldiers "
        "with dark, atmospheric historical detail'"
    ),
    height=140,
    label_visibility="collapsed",
)

# ── Pipeline stappen preview ───────────────────────────────────────────────────
steps_html = ""
steps = [
    ("Script", "AI generates ~1400 words"),
    ("Voice-over", "edge-tts voice generation"),
    ("Visuals", "Pexels stock footage"),
    ("Montage", "MoviePy video editing"),
    ("Export", "Final MP4 render"),
]
for i, (title, desc) in enumerate(steps, 1):
    steps_html += (
        f'<div style="display:flex;align-items:center;gap:14px;padding:10px 0;border-bottom:1px solid #151515">'
        f'<span style="font-size:11px;color:#333;width:16px;text-align:right;flex-shrink:0">{i:02d}</span>'
        f'<span style="font-size:13px;color:#888;font-weight:500;width:80px;flex-shrink:0">{title}</span>'
        f'<span style="font-size:12px;color:#444">{desc}</span>'
        f'</div>'
    )
st.markdown(f'<div style="margin:1.5rem 0;padding:0 4px">{steps_html}</div>', unsafe_allow_html=True)

# ── Interrupted projects ──────────────────────────────────────────────────────
incomplete = detect_incomplete_projects()
if incomplete:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#444;margin-bottom:12px'>Interrupted projects</p>", unsafe_allow_html=True)
    active_pipeline_status = get_active_pipeline_status()
    active_pipeline_dir = str(active_pipeline_status.get("project_dir", "")) if active_pipeline_status else ""
    for proj in incomplete:
        mini_tracker = ""
        for i, name in enumerate(PIPELINE_STEPS):
            if i < proj["step_num"]:
                dot = f'<span style="color:#4ade80;font-size:10px">✓</span>'
            elif i == proj["step_num"]:
                dot = f'<span style="color:#f5f5f5;font-size:10px">●</span>'
            else:
                dot = f'<span style="color:#2a2a2a;font-size:10px">○</span>'
            mini_tracker += f'{dot}<span style="font-size:10px;color:#444;margin:0 4px">{name}</span>'

        display_name = display_project_name(proj["name"])
        theme_preview = (proj["theme"][:60] + "…") if proj["theme"] and len(proj["theme"]) > 60 else (proj["theme"] or "—")

        col_info, col_resume, col_delete = st.columns([4, 1, 1])
        with col_info:
            st.markdown(
                f'<div class="resume-card">'
                f'<div>'
                f'<div style="font-size:13px;color:#ccc;font-weight:500;margin-bottom:4px">{display_name}</div>'
                f'<div style="font-size:11px;color:#555;margin-bottom:8px">{theme_preview}</div>'
                f'<div style="display:flex;align-items:center;gap:2px">{mini_tracker}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_resume:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            resume_locked = is_lock_active(proj["project_dir"])
            settings_mismatch = get_resume_settings_mismatch(
                proj,
                voice=voice_option,
                quality=video_quality,
                target_duration_minutes=target_duration_minutes,
                clips_per_scene=clips_per_scene,
                visual_mode=visual_mode,
                scene_workers=scene_workers,
                context_image_overlays=context_image_overlays,
                micro_test_live=micro_test_live,
            )
            resume_disabled = resume_locked or (
                active_pipeline_status is not None and Path(proj["project_dir"]).resolve() != Path(active_pipeline_dir).resolve()
            ) or settings_mismatch is not None
            if st.button("Resume", key=f"resume_{proj['project_dir']}", use_container_width=True, disabled=resume_disabled):
                try:
                    project_dir = launch_pipeline_job(
                        theme=proj["theme"] or theme_prompt.strip() or "untitled theme",
                        voice=proj.get("voice", voice_option),
                        quality=proj.get("quality", video_quality),
                        target_duration_minutes=proj.get("target_duration_minutes", target_duration_minutes),
                        cleanup=True,
                        clips_per_scene=proj.get("clips_per_scene", clips_per_scene),
                        visual_mode=proj.get("visual_mode", visual_mode),
                        scene_workers=proj.get("scene_workers", scene_workers),
                        context_image_overlays=proj.get("context_image_overlays", context_image_overlays),
                        micro_test_live=proj.get("micro_test_live", micro_test_live),
                        resume_dir=proj["project_dir"],
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["active_project_dir"] = str(project_dir)
                    st.rerun()
            if resume_locked:
                st.caption("Locked by an active worker")
            elif settings_mismatch is not None:
                st.caption(f"Resume blocked: change UI settings back to match the original project ({settings_mismatch})")
            elif active_pipeline_status is not None and Path(proj["project_dir"]).resolve() != Path(active_pipeline_dir).resolve():
                st.caption("Another pipeline is already running")
        with col_delete:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            if st.button("Delete", key=f"delete_{proj['project_dir']}", use_container_width=True, disabled=resume_locked):
                delete_project_dir(proj["project_dir"])
                if st.session_state.get("active_project_dir") == proj["project_dir"]:
                    st.session_state.pop("active_project_dir", None)
                st.rerun()

# ── Active pipeline ────────────────────────────────────────────────────────────
focus_status = get_focus_status()
active_running = bool(focus_status and focus_status.get("status") in {"queued", "running"})
new_project_locked = False
resume_candidate_dir = None
if theme_prompt.strip():
    resume_candidate_dir = find_matching_resume_project(
        theme_prompt.strip(),
        voice_option,
        video_quality,
        target_duration_minutes,
        clips_per_scene,
        visual_mode,
        scene_workers,
        context_image_overlays,
        micro_test_live,
    )
    if resume_candidate_dir is not None:
        new_project_locked = is_lock_active(resume_candidate_dir)

# ── Start / stop buttons ──────────────────────────────────────────────────────
btn_col, stop_col, status_col = st.columns([1, 1, 3])

with btn_col:
    start_btn = st.button("Start", use_container_width=True, disabled=active_running or new_project_locked)

with stop_col:
    stop_btn = st.button("Stop", use_container_width=True, disabled=not active_running)

with status_col:
    if active_running:
        if focus_status.get("cancel_requested"):
            st.markdown("<p style='color:#facc15;font-size:13px;padding-top:10px'>Stop requested. Waiting for the worker to exit cleanly.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#4ade80;font-size:13px;padding-top:10px'>A pipeline is already running. Status stays visible after refresh.</p>", unsafe_allow_html=True)
    elif not theme_prompt.strip():
        st.markdown("<p style='color:#333;font-size:13px;padding-top:10px'>Enter a theme to begin</p>", unsafe_allow_html=True)
    else:
        words = len(theme_prompt.split())
        ai_ok = key_set("OPENAI_API_KEY") if provider == "openai" else key_set("ANTHROPIC_API_KEY")
        tts_ok = (not is_openai_tts_voice(voice_option)) or key_set("OPENAI_API_KEY")
        pexels_ready = key_set("PEXELS_API_KEY")
        if not ai_ok:
            st.markdown("<p style='color:#f87171;font-size:13px;padding-top:10px'>Missing AI API key. Add it in the sidebar.</p>", unsafe_allow_html=True)
        elif not tts_ok:
            st.markdown("<p style='color:#f87171;font-size:13px;padding-top:10px'>Missing OpenAI API key for OpenAI TTS.</p>", unsafe_allow_html=True)
        elif not pexels_ready:
            st.markdown("<p style='color:#f87171;font-size:13px;padding-top:10px'>Missing Pexels API key. Add it in the sidebar.</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:#4ade80;font-size:13px;padding-top:10px'>{words} words · {display_tts_voice(voice_option)} · {video_quality.upper()}</p>", unsafe_allow_html=True)

if stop_btn and active_running and focus_status:
    request_cancel(focus_status["project_dir"])
    terminate_process(focus_status.get("pid"))
    st.session_state["active_project_dir"] = focus_status["project_dir"]
    st.rerun()

if new_project_locked:
    st.markdown("<p style='color:#facc15;font-size:13px'>This project is already locked by an active worker.</p>", unsafe_allow_html=True)

if start_btn and not active_running and not new_project_locked:
    ai_ok = key_set("OPENAI_API_KEY") if provider == "openai" else key_set("ANTHROPIC_API_KEY")
    tts_ok = (not is_openai_tts_voice(voice_option)) or key_set("OPENAI_API_KEY")
    pexels_ready = key_set("PEXELS_API_KEY")
    if not theme_prompt.strip():
        st.markdown("<p style='color:#f87171;font-size:13px'>Enter a theme first</p>", unsafe_allow_html=True)
    elif not ai_ok:
        st.markdown("<p style='color:#f87171;font-size:13px'>Enter a valid AI API key first</p>", unsafe_allow_html=True)
    elif not tts_ok:
        st.markdown("<p style='color:#f87171;font-size:13px'>Enter a valid OpenAI API key to use OpenAI TTS</p>", unsafe_allow_html=True)
    elif not pexels_ready:
        st.markdown("<p style='color:#f87171;font-size:13px'>Enter a valid Pexels API key first</p>", unsafe_allow_html=True)
    else:
        try:
            project_dir = launch_pipeline_job(
                theme=theme_prompt.strip(),
                voice=voice_option,
                quality=video_quality,
                target_duration_minutes=target_duration_minutes,
                cleanup=True,
                clips_per_scene=clips_per_scene,
                visual_mode=visual_mode,
                scene_workers=scene_workers,
                context_image_overlays=context_image_overlays,
                micro_test_live=micro_test_live,
                resume_dir=str(resume_candidate_dir) if resume_candidate_dir else None,
            )
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.session_state["active_project_dir"] = str(project_dir)
            st.rerun()

if focus_status:
    st.session_state["active_project_dir"] = focus_status["project_dir"]
    render_job_panel(focus_status)
    if focus_status.get("status") in {"queued", "running"}:
        time.sleep(DEFAULT_POLL_INTERVAL_SECONDS)
        st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<p style='font-size:11px;color:#222;margin-top:3rem;text-align:center'>edge-tts · Pexels · MoviePy · OpenAI / Anthropic</p>", unsafe_allow_html=True)

import argparse
import faulthandler
import inspect
import logging
import os
import signal
import threading
from pathlib import Path

from pipeline_status import (
    compute_step_progress,
    refresh_lock,
    infer_active_step,
    now_iso,
    read_status,
    read_lock,
    release_lock,
    write_status,
)
from video_engine import CancellationRequested, DocumentaryPipeline


def configure_logging(project_dir: Path) -> logging.Logger:
    log_path = project_dir / "worker.log"
    logger = logging.getLogger(f"pipeline_worker.{project_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--target-duration-minutes", required=True)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--lock-token", required=True)
    parser.add_argument("--cleanup", choices=["0", "1"], default="1")
    parser.add_argument("--clips-per-scene", default="4")
    parser.add_argument("--visual-mode", default="standard")
    parser.add_argument("--scene-workers", default="3")
    parser.add_argument("--context-image-overlays", choices=["0", "1"], default="1")
    parser.add_argument("--micro-test-live", choices=["0", "1"], default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    logger = configure_logging(project_dir)
    crash_log_handle = (project_dir / "worker_crash.log").open("a", encoding="utf-8")
    faulthandler.enable(file=crash_log_handle)
    lock_token = args.lock_token
    cleanup = args.cleanup == "1"
    clips_per_scene = max(1, int(args.clips_per_scene))
    target_duration_minutes = int(args.target_duration_minutes)
    visual_mode = args.visual_mode
    scene_workers = max(1, min(4, int(args.scene_workers)))
    context_image_overlays = args.context_image_overlays == "1"
    micro_test_live = args.micro_test_live == "1"
    existing_status = read_status(project_dir) or {}
    existing_lock = read_lock(project_dir)

    if not existing_lock or existing_lock.get("token") != lock_token:
        logger.error("Worker lock mismatch for %s", project_dir)
        state = {
            "project_dir": str(project_dir),
            "project_name": project_dir.name,
            "theme": args.theme,
            "voice": args.voice,
            "quality": args.quality,
            "target_duration_minutes": target_duration_minutes,
            "clips_per_scene": clips_per_scene,
            "visual_mode": visual_mode,
            "scene_workers": scene_workers,
            "context_image_overlays": context_image_overlays,
            "micro_test_live": micro_test_live,
            "status": "failed",
            "progress": int(existing_status.get("progress", 0) or 0),
            "active_step": existing_status.get("active_step", "script"),
            "step_progress": existing_status.get("step_progress") or compute_step_progress(existing_status.get("progress", 0)),
            "last_message": "Worker lock mismatch",
            "logs": existing_status.get("logs", []),
            "export_status": existing_status.get("export_status"),
            "asset_status": existing_status.get("asset_status"),
            "pid": os.getpid(),
            "started_at": existing_status.get("started_at", now_iso()),
            "updated_at": now_iso(),
            "result": None,
            "cancel_requested": bool(existing_status.get("cancel_requested")),
        }
        write_status(project_dir, state)
        crash_log_handle.close()
        return

    state = {
        "project_dir": str(project_dir),
        "project_name": project_dir.name,
        "theme": args.theme,
        "voice": args.voice,
        "quality": args.quality,
        "target_duration_minutes": target_duration_minutes,
        "clips_per_scene": clips_per_scene,
        "visual_mode": visual_mode,
        "scene_workers": scene_workers,
        "context_image_overlays": context_image_overlays,
        "micro_test_live": micro_test_live,
        "status": "running",
        "progress": 0,
        "active_step": "script",
        "step_progress": compute_step_progress(0),
        "last_message": "Worker started",
        "logs": [],
        "export_status": None,
        "asset_status": existing_status.get("asset_status"),
        "pid": os.getpid(),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "result": None,
        "cancel_requested": bool(existing_status.get("cancel_requested")),
    }
    refresh_lock(project_dir, lock_token, pid=os.getpid(), owner="worker")
    write_status(project_dir, state)
    logger.info(
        "Worker started pid=%s project=%s voice=%s quality=%s clips_per_scene=%s visual_mode=%s scene_workers=%s cleanup=%s",
        os.getpid(),
        project_dir.name,
        args.voice,
        args.quality,
        clips_per_scene,
        visual_mode,
        scene_workers,
        cleanup,
    )

    if state["cancel_requested"]:
        state["status"] = "cancelled"
        state["last_message"] = "Stop requested before worker start"
        write_status(project_dir, state)
        logger.info("Stop requested before worker start")
        crash_log_handle.close()
        return

    stop_heartbeat = threading.Event()
    state_lock = threading.Lock()

    def flush_state():
        with state_lock:
            state["updated_at"] = now_iso()
            snapshot = dict(state)
        write_status(project_dir, snapshot)
        refresh_lock(project_dir, lock_token, pid=os.getpid(), owner="worker")

    def heartbeat():
        while not stop_heartbeat.wait(2):
            flush_state()

    def update_status(message: str, progress: int | None = None, level: str = "info", details: dict | None = None):
        with state_lock:
            if progress is not None:
                state["progress"] = max(0, min(100, int(progress)))
            state["active_step"] = infer_active_step(state.get("progress"))
            state["step_progress"] = compute_step_progress(state.get("progress"))
            state["last_message"] = message
            if details:
                if "export_status" in details:
                    state["export_status"] = details["export_status"]
                if "asset_status" in details:
                    state["asset_status"] = details["asset_status"]
            state["logs"] = (state["logs"] + [{
                "timestamp": now_iso(),
                "level": level,
                "message": message,
                "progress": progress,
            }])[-200:]
        log_method = logger.error if level == "error" else logger.info
        if progress is None:
            log_method("%s", message)
        else:
            log_method("%s [progress=%s]", message, progress)
        flush_state()

    def handle_stop_signal(_signum, _frame):
        with state_lock:
            state["cancel_requested"] = True
            state["status"] = "cancelled"
            state["last_message"] = "Stop requested"
            state["active_step"] = infer_active_step(state.get("progress"))
            state["step_progress"] = compute_step_progress(state.get("progress"))
            state["logs"] = (state["logs"] + [{
                "timestamp": now_iso(),
                "level": "info",
                "message": "Stop requested",
                "progress": state.get("progress"),
            }])[-200:]
        logger.info("Received stop signal")
        flush_state()
        raise CancellationRequested("Stop requested")

    signal.signal(signal.SIGTERM, handle_stop_signal)
    signal.signal(signal.SIGINT, handle_stop_signal)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()

    try:
        pipeline_kwargs = {
            "theme": args.theme,
            "voice": args.voice,
            "video_quality": args.quality,
            "target_duration_minutes": target_duration_minutes,
            "cleanup": cleanup,
            "clips_per_scene": clips_per_scene,
            "visual_mode": visual_mode,
            "scene_workers": scene_workers,
            "status_callback": update_status,
            "resume_project_dir": str(project_dir),
            "micro_test_live": micro_test_live,
        }
        if "context_image_overlays" in inspect.signature(DocumentaryPipeline).parameters:
            pipeline_kwargs["context_image_overlays"] = context_image_overlays
        pipeline = DocumentaryPipeline(**pipeline_kwargs)
        result = pipeline.run()
        with state_lock:
            state["result"] = result
        if result.get("success"):
            logger.info("Pipeline completed successfully")
            with state_lock:
                state["status"] = "completed"
                state["progress"] = 100
                state["last_message"] = "Pipeline complete"
        elif result.get("cancelled"):
            logger.info("Pipeline cancelled: %s", result.get("error", "Pipeline stopped"))
            with state_lock:
                state["status"] = "cancelled"
                state["cancel_requested"] = True
                state["last_message"] = result.get("error", "Pipeline stopped")
        else:
            logger.error("Pipeline failed: %s", result.get("error", "Pipeline failed"))
            with state_lock:
                state["status"] = "failed"
                state["last_message"] = result.get("error", "Pipeline failed")
        with state_lock:
            state["active_step"] = infer_active_step(state.get("progress"))
            state["step_progress"] = compute_step_progress(state.get("progress"))
        flush_state()
    except CancellationRequested as exc:
        logger.info("Worker cancelled: %s", exc)
        with state_lock:
            state["status"] = "cancelled"
            state["cancel_requested"] = True
            state["last_message"] = str(exc)
            state["active_step"] = infer_active_step(state.get("progress"))
            state["step_progress"] = compute_step_progress(state.get("progress"))
            state["logs"] = (state["logs"] + [{
                "timestamp": now_iso(),
                "level": "info",
                "message": str(exc),
                "progress": state.get("progress"),
            }])[-200:]
        flush_state()
    except Exception as exc:
        logger.exception("Unhandled worker exception")
        with state_lock:
            state["status"] = "failed"
            state["last_message"] = str(exc)
            state["active_step"] = infer_active_step(state.get("progress"))
            state["step_progress"] = compute_step_progress(state.get("progress"))
            state["logs"] = (state["logs"] + [{
                "timestamp": now_iso(),
                "level": "error",
                "message": str(exc),
                "progress": state.get("progress"),
            }])[-200:]
        flush_state()
        raise
    finally:
        stop_heartbeat.set()
        logger.info("Releasing worker lock")
        release_lock(project_dir, lock_token)
        crash_log_handle.close()


if __name__ == "__main__":
    main()

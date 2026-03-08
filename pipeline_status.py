import json
import os
import signal
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

STATUS_FILE = "status.json"
LOCK_FILE = ".run.lock"
LOCK_STALE_SECONDS = 30
STEP_KEYS = ["script", "voice_over", "beelden", "montage", "export"]
STEP_LABELS = {
    "script": "Script",
    "voice_over": "Voice-over",
    "beelden": "Visuals",
    "montage": "Editing",
    "export": "Export",
}
STEP_RANGES = {
    "script": (0, 20),
    "voice_over": (20, 45),
    "beelden": (45, 65),
    "montage": (65, 85),
    "export": (85, 100),
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def status_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / STATUS_FILE


def lock_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / LOCK_FILE


def compute_step_progress(progress: int | float | None) -> dict[str, int]:
    value = 0 if progress is None else max(0, min(100, int(progress)))
    result: dict[str, int] = {}
    for key, (start, end) in STEP_RANGES.items():
        if value <= start:
            result[key] = 0
        elif value >= end:
            result[key] = 100
        else:
            span = end - start
            result[key] = int(((value - start) / span) * 100)
    return result


def infer_active_step(progress: int | float | None) -> str:
    steps = compute_step_progress(progress)
    for key in STEP_KEYS:
        if steps[key] < 100:
            return key
    return STEP_KEYS[-1]


def read_status(project_dir: str | Path) -> dict | None:
    path = status_path(project_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_status(project_dir: str | Path, data: dict) -> Path:
    path = status_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload.setdefault("updated_at", now_iso())
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        tmp_file.write(serialized)
        temp_path = Path(tmp_file.name)
    temp_path.replace(path)
    return path


def read_lock(project_dir: str | Path) -> dict | None:
    path = lock_path(project_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {"invalid": True, "path": str(path)}
    data.setdefault("path", str(path))
    return data


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def process_state(pid: int | None) -> str:
    if not pid:
        return "missing"
    try:
        os.kill(pid, 0)
    except OSError:
        return "dead"
    proc_status = Path(f"/proc/{pid}/status")
    if proc_status.exists():
        try:
            for line in proc_status.read_text().splitlines():
                if line.startswith("State:"):
                    if "\tZ" in line or " zombie" in line.lower():
                        return "zombie"
                    break
        except OSError:
            pass
    return "alive"


def lock_is_stale(lock_data: dict | None) -> bool:
    if not lock_data:
        return False
    if lock_data.get("invalid"):
        return True
    pid = lock_data.get("pid")
    if pid and process_state(pid) != "alive":
        return True
    last_seen = _parse_iso8601(lock_data.get("updated_at")) or _parse_iso8601(lock_data.get("created_at"))
    if pid is None and last_seen:
        age = (datetime.now(timezone.utc) - last_seen).total_seconds()
        if age > LOCK_STALE_SECONDS:
            return True
    return False


def is_lock_active(project_dir: str | Path) -> bool:
    lock_data = read_lock(project_dir)
    if not lock_data:
        return False
    if lock_is_stale(lock_data):
        try:
            lock_path(project_dir).unlink()
        except FileNotFoundError:
            pass
        except OSError:
            return True
        return False
    return True


def claim_lock(project_dir: str | Path, *, owner: str, pid: int | None = None) -> dict:
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    path = lock_path(project_dir)
    token = uuid.uuid4().hex
    payload = {
        "token": token,
        "owner": owner,
        "pid": pid,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            existing = read_lock(project_dir)
            if attempt == 0 and lock_is_stale(existing):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    break
                continue
            raise RuntimeError("Project is already running")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return payload
    raise RuntimeError("Project is already running")


def refresh_lock(project_dir: str | Path, token: str, *, pid: int | None = None, owner: str | None = None) -> bool:
    path = lock_path(project_dir)
    current = read_lock(project_dir)
    if not current or current.get("token") != token:
        return False
    current["updated_at"] = now_iso()
    if pid is not None:
        current["pid"] = pid
    if owner is not None:
        current["owner"] = owner
    serialized = json.dumps(current, ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        tmp_file.write(serialized)
        temp_path = Path(tmp_file.name)
    temp_path.replace(path)
    return True


def release_lock(project_dir: str | Path, token: str) -> bool:
    path = lock_path(project_dir)
    current = read_lock(project_dir)
    if not current or current.get("token") != token:
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def normalize_runtime_status(project_dir: str | Path, status: dict | None) -> dict | None:
    if not status:
        return None
    normalized = dict(status)
    runtime_status = normalized.get("status")
    if runtime_status not in {"queued", "running"}:
        return normalized
    if is_lock_active(project_dir):
        return normalized
    pid = normalized.get("pid")
    state = process_state(pid)
    normalized["status"] = "cancelled" if normalized.get("cancel_requested") else "failed"
    if normalized.get("cancel_requested"):
        normalized["last_message"] = "Pipeline stopped"
    elif state == "zombie":
        normalized["last_message"] = "Worker stopped unexpectedly (zombie process)"
    elif state == "dead":
        normalized["last_message"] = "Worker stopped unexpectedly"
    else:
        normalized["last_message"] = "Worker lock expired"
    normalized["updated_at"] = now_iso()
    return normalized


def process_is_alive(pid: int | None) -> bool:
    return process_state(pid) == "alive"


def cancel_requested(project_dir: str | Path) -> bool:
    status = read_status(project_dir) or {}
    return bool(status.get("cancel_requested"))


def request_cancel(project_dir: str | Path, message: str = "Stop requested") -> bool:
    status = read_status(project_dir) or {}
    status["cancel_requested"] = True
    status["last_message"] = message
    status.setdefault("updated_at", now_iso())
    write_status(project_dir, status)
    return True


def clear_cancel_request(project_dir: str | Path) -> bool:
    status = read_status(project_dir) or {}
    if status.get("cancel_requested"):
        status["cancel_requested"] = False
        write_status(project_dir, status)
    return True


def terminate_process(pid: int | None) -> bool:
    if not process_is_alive(pid):
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except OSError:
            return False

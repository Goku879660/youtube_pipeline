from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

HTTP_SESSION = requests.Session()


class ImageCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _filename_for_url(self, image_url: str) -> str:
        parsed = urlparse(image_url)
        suffix = Path(parsed.path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        digest = hashlib.md5(image_url.encode("utf-8")).hexdigest()[:16]
        return f"{digest}{suffix}"

    def _is_valid_image(self, path: Path) -> bool:
        try:
            if not path.exists() or path.stat().st_size < 32 * 1024:
                return False
        except OSError:
            return False
        if not PIL_AVAILABLE:
            return True
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            return width >= 400 and height >= 250
        except Exception:
            return False

    def download(self, image_url: str) -> str | None:
        target = self.cache_dir / self._filename_for_url(image_url)
        temp = target.with_suffix(target.suffix + ".part")

        if self._is_valid_image(target):
            return str(target)

        for _ in range(3):
            try:
                if temp.exists():
                    temp.unlink()
                response = HTTP_SESSION.get(
                    image_url,
                    stream=True,
                    timeout=20,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                response.raise_for_status()
                with open(temp, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            handle.write(chunk)
                if not self._is_valid_image(temp):
                    raise RuntimeError("downloaded image is invalid")
                temp.replace(target)
                return str(target)
            except Exception:
                for candidate in (temp, target):
                    if candidate.exists():
                        try:
                            candidate.unlink()
                        except OSError:
                            pass
        return None

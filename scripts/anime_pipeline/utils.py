from __future__ import annotations

import re
import sys
from pathlib import Path

# Workspace root is two levels up from scripts/anime_pipeline/
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def normalize_path(path_str: str) -> Path:
    """Normalize Windows paths (e.g. C:\\... or C:/...) to WSL Linux paths (/mnt/c/...)."""
    p = path_str.strip().strip('"').strip("'")
    match = re.match(r"^([a-zA-Z]):[/\\](.*)", p)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(p).resolve()


def collect_images(paths_or_dirs: list[str]) -> list[Path]:
    """Collect valid image paths from a list of files or directories."""
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    collected: list[Path] = []
    for raw in paths_or_dirs:
        p = normalize_path(raw)
        if p.is_file():
            if p.suffix.lower() in valid_exts:
                collected.append(p)
            else:
                print(f"[Warning] Skipping non-image file: {p}", file=sys.stderr)
        elif p.is_dir():
            for item in sorted(p.iterdir()):
                if item.is_file() and item.suffix.lower() in valid_exts:
                    collected.append(item)
        else:
            print(f"[Warning] Path not found: {p}", file=sys.stderr)
    return collected


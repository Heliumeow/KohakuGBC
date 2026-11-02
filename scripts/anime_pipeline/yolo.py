from __future__ import annotations

import fnmatch
import json
import os
import sys
from pathlib import Path

from .utils import WORKSPACE_ROOT

DEFAULT_PRESETS: dict[str, dict] = {
    "anime-sfw": {
        "description": "Anime general/all-ages: eyes, face, mouth, jewelry, hands, hair, person/objects",
        "models": [
            "Eyes.pt",
            "adetailer2dMouth_v10.pt",
            "jewelryDetector_v021.pt",
            "yolo26s-face.pt",
            "yolo26s-person.pt",
            "yolo26x.pt",
        ],
        "include_patterns": [
            "*eye*",
            "*mouth*",
            "*face*",
            "*jewelry*",
            "*hair*",
            "*hand*",
            "*accessory*",
            "*person*",
            "*head*",
            "*yolo26x*",
        ],
        "exclude_patterns": [
            "*cock*",
            "*ball*",
            "*penis*",
            "*pussy*",
            "*vagina*",
            "*nipple*",
            "*cum*",
            "*nude*",
            "*nsfw*",
        ],
    },
    "anime-nsfw": {
        "description": "Anime adult/NSFW: SFW features plus breasts, nipples, penis, genitalia, person/objects",
        "models": [
            "Eyes.pt",
            "adetailer2dMouth_v10.pt",
            "jewelryDetector_v021.pt",
            "breasts_seg.pt",
            "nipples_v2_yolov11s-seg.pt",
            "cockAndBallDetection2D_v20.pt",
            "yolo26s-face.pt",
            "yolo26s-person.pt",
            "yolo26x.pt",
        ],
        "include_patterns": ["*"],
        "exclude_patterns": ["*real*", "*photo*"],
    },
    "real": {
        "description": "Real-world photography & realistic human subjects (excludes 2D anime models)",
        "models": ["yolov8n-face.pt", "yolo11n-face.pt"],
        "include_patterns": ["*real*", "*photo*", "*face*"],
        "exclude_patterns": ["*adetailer2d*", "*2d*"],
    },
    "all": {
        "description": "Load all available Ultralytics YOLO models in models/ultralytics",
        "models": ["*"],
        "include_patterns": ["*"],
        "exclude_patterns": [],
    },
}

NSFW_TAG_KEYWORDS = {
    "nsfw",
    "nude",
    "naked",
    "penis",
    "pussy",
    "vagina",
    "nipples",
    "areola",
    "sex",
    "fellatio",
    "cunnilingus",
    "cum",
    "ejaculation",
    "masturbation",
    "topless",
    "bottomless",
    "uncensored",
    "genitals",
    "anus",
}


def load_presets(model_dir: Path) -> dict[str, dict]:
    """Load presets configuration from presets.json if available, merged over defaults."""
    presets = dict(DEFAULT_PRESETS)
    candidates = [
        model_dir / "presets.json",
        WORKSPACE_ROOT / "models" / "ultralytics" / "presets.json",
        WORKSPACE_ROOT / "configs" / "yolo_presets.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                data = json.loads(c.read_text(encoding="utf-8"))
                custom_presets = data.get("presets", data)
                if isinstance(custom_presets, dict):
                    for k, v in custom_presets.items():
                        if isinstance(v, dict):
                            presets[k] = v
                        elif isinstance(v, list):
                            presets[k] = {"models": v, "include_patterns": v, "exclude_patterns": []}
                    break
            except Exception as e:
                print(f"[Warning] Failed to read presets from {c}: {e}", file=sys.stderr)
    return presets


def resolve_auto_preset(tag_results: list[dict] | dict | None) -> tuple[str, str]:
    """Inspect CLTagger tags to determine if an image contains explicit NSFW tags."""
    if not tag_results:
        return "anime-sfw", "default (no tags provided)"

    items = tag_results if isinstance(tag_results, list) else [tag_results]
    for item in items:
        gen_tags = item.get("tags", {}).get("general", {})
        for tag, score in gen_tags.items():
            clean_tag = tag.lower().replace("_", " ")
            if score >= 0.35 and any(k in clean_tag for k in NSFW_TAG_KEYWORDS):
                return "anime-nsfw", f"detected NSFW tag '{tag}' ({score:.2f})"

    return "anime-sfw", "no explicit NSFW tags detected"


def resolve_preset_models(
    preset_name: str,
    available_files: list[Path],
    presets_dict: dict[str, dict],
) -> list[Path]:
    """Resolve which model files match the requested preset."""
    if preset_name not in presets_dict:
        # Fuzzy match alias
        alias_map = {
            "sfw": "anime-sfw",
            "nsfw": "anime-nsfw",
            "r18": "anime-nsfw",
            "photo": "real",
            "realistic": "real",
        }
        preset_name = alias_map.get(preset_name.lower(), preset_name)

    if preset_name not in presets_dict:
        print(f"[Warning] Unknown YOLO preset '{preset_name}'. Falling back to 'anime-sfw'.", file=sys.stderr)
        preset_name = "anime-sfw"

    cfg = presets_dict[preset_name]
    explicit_models = {m.lower() for m in cfg.get("models", [])}
    inc_patterns = cfg.get("include_patterns", ["*"])
    exc_patterns = cfg.get("exclude_patterns", [])

    matched: list[Path] = []
    for pt in available_files:
        pt_name = pt.name.lower()
        pt_stem = pt.stem.lower()

        # Wildcard all
        if "*" in explicit_models:
            is_inc = any(fnmatch.fnmatch(pt_name, pat.lower()) for pat in inc_patterns)
            is_exc = any(fnmatch.fnmatch(pt_name, pat.lower()) for pat in exc_patterns)
            if is_inc and not is_exc:
                matched.append(pt)
            continue

        # Explicit name match
        if pt_name in explicit_models or pt_stem in explicit_models:
            matched.append(pt)
            continue

        # Pattern match
        is_inc = any(fnmatch.fnmatch(pt_name, pat.lower()) for pat in inc_patterns)
        is_exc = any(fnmatch.fnmatch(pt_name, pat.lower()) for pat in exc_patterns)
        if is_inc and not is_exc:
            matched.append(pt)

    return matched


class UltralyticsEnsemble:
    """Ensemble runner for Ultralytics YOLO models (detect & segment) in models/ultralytics.

    Supports configurable presets:
    - 'anime-sfw': eyes, mouth, jewelry, face, hair, hands
    - 'anime-nsfw': SFW features plus breasts, nipples, penis, genitalia
    - 'real': real-world human photography
    - 'all': all available models
    - 'auto': automatically selects anime-sfw or anime-nsfw based on CLTagger tags
    """

    def __init__(
        self,
        model_dir: Path | str | None = None,
        preset: str = "anime-sfw",
        model_names: list[str] | None = None,
        tag_hint: list[dict] | dict | None = None,
        device: str = "auto",
        conf: float = 0.20,
        imgsz: int = 1280,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[Warning] ultralytics is not installed. Skipping YOLO spatial priors.", file=sys.stderr)
            self.models = {}
            return

        import torch

        if model_dir is None:
            model_dir = WORKSPACE_ROOT / "models" / "ultralytics"
        self.model_dir = Path(model_dir)

        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.conf = conf
        self.imgsz = imgsz
        self.models: dict[str, YOLO] = {}
        self.preset = preset

        if not self.model_dir.exists():
            return

        all_pts = sorted(list(self.model_dir.glob("*.pt")) + list(self.model_dir.glob("*.onnx")))
        presets_dict = load_presets(self.model_dir)

        # 1. Resolve preset name if auto
        active_preset = preset
        if preset.lower() == "auto":
            active_preset, reason = resolve_auto_preset(tag_hint)
            print(f"[*] YOLO Preset 'auto' resolved to '{active_preset}' ({reason})")

        # 2. Select model files
        selected_pts: list[Path] = []
        if model_names:
            names_set = {n.lower() for n in model_names}
            for pt in all_pts:
                if pt.name.lower() in names_set or pt.stem.lower() in names_set:
                    selected_pts.append(pt)
            print(f"[*] Explicit YOLO models requested: {', '.join(p.name for p in selected_pts)}")
        else:
            selected_pts = resolve_preset_models(active_preset, all_pts, presets_dict)

        # 3. Load selected models
        for pt in selected_pts:
            try:
                m = YOLO(str(pt))
                self.models[pt.name] = m
            except Exception as e:
                print(f"[Warning] Failed to load YOLO model {pt.name}: {e}", file=sys.stderr)

        if self.models:
            print(f"[*] Loaded {len(self.models)} Ultralytics YOLO model(s) [Preset: {active_preset}]:")
            for name, m in self.models.items():
                classes = list(m.names.values())
                cls_prev = ", ".join(classes[:4]) + ("..." if len(classes) > 4 else "")
                print(f"    • {name} [{m.task}]: {cls_prev}")
        elif not model_names and selected_pts:
            print(f"[Warning] No models were successfully loaded for preset '{active_preset}'.", file=sys.stderr)

    @property
    def is_available(self) -> bool:
        return len(self.models) > 0

    def detect(self, img_path: Path) -> list[dict]:
        from PIL import Image

        if not self.models:
            return []

        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            print(f"[Warning] Failed to open image for YOLO: {img_path} ({e})", file=sys.stderr)
            return []

        results = []
        for model_name, model in self.models.items():
            try:
                out = model(str(img_path), imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
                for b in out.boxes:
                    cls_id = int(b.cls[0])
                    cls_name = model.names.get(cls_id, str(cls_id))
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    results.append({
                        "class": cls_name,
                        "conf": round(conf, 4),
                        "model": model_name,
                        "bbox": {
                            "left": round(max(0.0, x1 / w), 4),
                            "top": round(max(0.0, y1 / h), 4),
                            "right": round(min(1.0, x2 / w), 4),
                            "bottom": round(min(1.0, y2 / h), 4),
                            "confidence": round(conf, 4),
                        },
                    })
            except Exception as e:
                print(f"[Warning] Model {model_name} failed on {img_path.name}: {e}", file=sys.stderr)

        return self._nms_dedup(results)

    @staticmethod
    def _nms_dedup(detections: list[dict], iou_thresh: float = 0.6) -> list[dict]:
        if len(detections) <= 1:
            return detections
        dets = sorted(detections, key=lambda d: -d["conf"])
        keep = []
        for d in dets:
            overlap = False
            b1 = d["bbox"]
            for k in keep:
                if k["class"] == d["class"]:
                    b2 = k["bbox"]
                    ix1 = max(b1["left"], b2["left"])
                    iy1 = max(b1["top"], b2["top"])
                    ix2 = min(b1["right"], b2["right"])
                    iy2 = min(b1["bottom"], b2["bottom"])
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        area1 = (b1["right"] - b1["left"]) * (b1["bottom"] - b1["top"])
                        area2 = (b2["right"] - b2["left"]) * (b2["bottom"] - b2["top"])
                        union = area1 + area2 - inter
                        if union > 0 and (inter / union) > iou_thresh:
                            overlap = True
                            break
            if not overlap:
                keep.append(d)
        return keep

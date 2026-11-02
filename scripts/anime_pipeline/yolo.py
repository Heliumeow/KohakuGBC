from __future__ import annotations

import sys
from pathlib import Path

from .utils import WORKSPACE_ROOT


class UltralyticsEnsemble:
    """Ensemble runner for Ultralytics YOLO models (detect & segment) in models/ultralytics.

    Automatically discovers and runs all available models in the folder (e.g. eyes, mouth,
    breasts, jewelry, accessories) to provide high-precision anime spatial priors.
    """

    def __init__(
        self,
        model_dir: Path | str | None = None,
        model_names: list[str] | None = None,
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

        if not self.model_dir.exists():
            return

        all_pts = sorted(list(self.model_dir.glob("*.pt")) + list(self.model_dir.glob("*.onnx")))
        selected_pts: list[Path] = []
        if model_names:
            names_set = {n.lower() for n in model_names}
            for pt in all_pts:
                if pt.name.lower() in names_set or pt.stem.lower() in names_set:
                    selected_pts.append(pt)
        else:
            # Default: use all models found in the directory
            selected_pts = all_pts

        for pt in selected_pts:
            try:
                m = YOLO(str(pt))
                self.models[pt.name] = m
            except Exception as e:
                print(f"[Warning] Failed to load YOLO model {pt.name}: {e}", file=sys.stderr)

        if self.models:
            print(f"[*] Loaded {len(self.models)} Ultralytics YOLO model(s) from {self.model_dir}:")
            for name, m in self.models.items():
                classes = list(m.names.values())
                cls_prev = ", ".join(classes[:4]) + ("..." if len(classes) > 4 else "")
                print(f"    • {name} [{m.task}]: {cls_prev}")

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

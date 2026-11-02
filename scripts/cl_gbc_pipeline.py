#!/usr/bin/env python3
"""CLTagger + KohakuGBC Automated Image Annotation Pipeline.

This script executes a two-stage image annotation workflow:
1. CLTagger (Anime ONNX Classifier) extracts preliminary prior tags
   (character, copyright/franchise, and visual general tags with confidence).
2. KohakuGBC ingests the visual prior tags, queries VLM (via llama-server)
   and fine-grained detectors (SAM3 / OWLv2), generating structured scene
   graphs and detailed captions.
3. Automatically generates an interactive HTML viewer (viewer.html).

Usage:
    python scripts/cl_gbc_pipeline.py <IMAGE_PATH_OR_DIR> [OPTIONS]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure KohakuGBC and scripts are in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(WORKSPACE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))


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


def extract_copyright_from_character(char_tag: str) -> str | None:
    """Extract copyright series from character tag format like 'name (series)'."""
    m = re.search(r"\(([^)]+)\)$", char_tag)
    if m:
        return m.group(1).strip()
    return None


def resolve_cltagger_model(custom_dir: Path | str | None = None) -> tuple[Path, Path]:
    """Find CLTagger model files, with auto-download fallback from HuggingFace."""
    candidates: list[Path] = []
    if custom_dir:
        candidates.append(Path(custom_dir))
    if os.environ.get("CLTAGGER_DIR"):
        candidates.append(Path(os.environ["CLTAGGER_DIR"]))
    candidates.append(WORKSPACE_ROOT / "models" / "cltagger" / "v2_01a")
    candidates.append(WORKSPACE_ROOT / "models" / "cltagger")
    candidates.append(Path.home() / ".cache" / "cltagger" / "v2_01a")

    for c in candidates:
        model_p = c / "model.onnx"
        vocab_p = c / "model_vocabulary.json"
        if model_p.exists() and vocab_p.exists():
            return model_p, vocab_p

    # If not found locally, auto-download from HuggingFace
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "CLTagger model not found locally and huggingface_hub is not installed. "
            "Please specify --cltagger-dir or install huggingface_hub."
        )

    print("[*] CLTagger model not found locally. Auto-downloading from cella110n/cl_tagger_v2...")
    target_dir = WORKSPACE_ROOT / "models" / "cltagger" / "v2_01a"
    target_dir.mkdir(parents=True, exist_ok=True)
    repo = "cella110n/cl_tagger_v2"
    for filename in ["v2_01a/model.onnx", "v2_01a/model.onnx.data", "v2_01a/model_vocabulary.json"]:
        dest = target_dir / Path(filename).name
        if not dest.exists():
            dl_path = hf_hub_download(repo_id=repo, filename=filename)
            shutil.copy2(dl_path, dest)
    return target_dir / "model.onnx", target_dir / "model_vocabulary.json"


class StandaloneCLTagger:
    """Self-contained CLTagger inference engine using ONNX Runtime.

    Zero external dependencies: only requires onnxruntime, numpy, and PIL.
    """

    def __init__(self, model_dir: Path | str | None = None) -> None:
        import numpy as np
        import onnxruntime as ort

        model_path, vocab_path = resolve_cltagger_model(model_dir)
        avail = ort.get_available_providers()
        selected = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in avail]
        self.session = ort.InferenceSession(str(model_path), providers=selected)

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        shape = input_meta.shape
        self.input_h = shape[2] if len(shape) >= 4 and isinstance(shape[2], int) else 384
        self.input_w = shape[3] if len(shape) >= 4 and isinstance(shape[3], int) else 384

        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.idx_to_tag = {int(k): v for k, v in vocab.get("idx_to_tag", {}).items()}
        self.tag_to_cat = vocab.get("tag_to_category", {})

    def tag_image(self, img_path: Path) -> dict[str, float]:
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img) or img
            img = img.convert("RGB")
            side = max(img.width, img.height)
            canvas = Image.new("RGB", (side, side), (255, 255, 255))
            canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
            canvas = canvas.resize((self.input_w, self.input_h), Image.Resampling.BICUBIC)
            arr = (np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None, ...]

        logits = self.session.run(None, {self.input_name: arr})[0][0]
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

        raw_scores: dict[str, float] = {}
        for idx, p in enumerate(scores):
            if idx in self.idx_to_tag:
                raw_scores[self.idx_to_tag[idx]] = float(p)
        return raw_scores


def run_cl_tagger(
    images: list[Path],
    threshold_general: float = 0.35,
    threshold_character: float = 0.40,
    model_dir: Path | str | None = None,
) -> list[dict]:
    """Run CLTagger to extract character, copyright, and general tags with confidences."""
    tagger = StandaloneCLTagger(model_dir=model_dir)

    results: list[dict] = []
    print(f"[*] Running CLTagger on {len(images)} image(s)...")
    for img_p in images:
        raw_scores = tagger.tag_image(img_p)

        char_candidates: dict[str, float] = {}
        copy_bucket: dict[str, float] = {}
        gen_bucket: dict[str, float] = {}

        for tag, score in raw_scores.items():
            category = tagger.tag_to_cat.get(tag, "General").lower()
            if category == "character":
                char_candidates[tag] = round(float(score), 4)
            elif category == "copyright":
                if score >= 0.25:
                    copy_bucket[tag] = round(float(score), 4)
            elif category == "general":
                if score >= threshold_general:
                    gen_bucket[tag] = round(float(score), 4)

        char_bucket: dict[str, float] = {}
        for tag, score in char_candidates.items():
            if score >= threshold_character:
                char_bucket[tag] = score

        # Fallback: if no character exceeded threshold_character, pick top candidate(s) if >= 0.25
        if not char_bucket and char_candidates:
            sorted_chars = sorted(char_candidates.items(), key=lambda kv: -kv[1])
            if sorted_chars and sorted_chars[0][1] >= 0.25:
                char_bucket[sorted_chars[0][0]] = sorted_chars[0][1]
                if len(sorted_chars) > 1 and sorted_chars[1][1] >= sorted_chars[0][1] * 0.8:
                    char_bucket[sorted_chars[1][0]] = sorted_chars[1][1]

        # Infer copyright if missing from explicit copyright tags
        if not copy_bucket and char_bucket:
            top_char = max(char_bucket.items(), key=lambda kv: kv[1])[0]
            series = extract_copyright_from_character(top_char)
            if series:
                copy_bucket[series] = char_bucket[top_char]

        entry = {
            "image": str(img_p),
            "tags": {
                "character": char_bucket,
                "copyright": copy_bucket,
                "general": gen_bucket,
            },
        }
        results.append(entry)

        # Log preview
        print(f"\n[CLTagger] Image: {img_p.name}")
        if char_bucket:
            chars_str = ", ".join(f"{k} ({v:.2f})" for k, v in sorted(char_bucket.items(), key=lambda x: -x[1])[:3])
            print(f"  Character: {chars_str}")
        if copy_bucket:
            copy_str = ", ".join(f"{k} ({v:.2f})" for k, v in sorted(copy_bucket.items(), key=lambda x: -x[1])[:3])
            print(f"  Copyright: {copy_str}")
        top_gen = sorted(gen_bucket.items(), key=lambda x: -x[1])[:8]
        top_gen_str = ", ".join(f"{k} ({v:.2f})" for k, v in top_gen)
        print(f"  Top Visual Tags: {top_gen_str}")

    return results


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


def merge_yolo_to_graph(graph_data: dict, yolo_detections: list[dict]) -> dict:
    """Merge YOLO spatial prior detections into scene graph vertices and edges."""
    if not yolo_detections:
        return graph_data

    vertices = graph_data.setdefault("vertices", [])
    edges = graph_data.setdefault("edges", [])

    # Find character or girl/person vertex to act as parent
    parent_vid = None
    for v in vertices:
        v_label = v.get("label")
        vid = v.get("vertex_id", "")
        if v_label == "entity" and any(k in vid.lower() for k in ["girl", "boy", "character", "person", "woman", "man"]):
            parent_vid = vid
            break
    if not parent_vid and vertices:
        parent_vid = vertices[0].get("vertex_id", "")

    existing_vids = {v.get("vertex_id") for v in vertices}
    class_counts: dict[str, int] = {}

    for det in yolo_detections:
        cls_name = det["class"]
        idx = class_counts.get(cls_name, 0)
        class_counts[cls_name] = idx + 1

        clean_cls = re.sub(r"[^\w\-]", "_", cls_name)
        if parent_vid and parent_vid != clean_cls:
            vid = f"{parent_vid}_{clean_cls}_{idx}"
        else:
            vid = f"{clean_cls}_{idx}"

        while vid in existing_vids:
            idx += 1
            vid = f"{clean_cls}_{idx}"
        existing_vids.add(vid)

        new_v = {
            "vertex_id": vid,
            "label": "entity",
            "bbox": det["bbox"],
            "descs": [
                {
                    "text": f"Character {cls_name} detected via {det['model']} (confidence: {det['conf']:.2f}).",
                    "type": "yolo_prior",
                }
            ],
        }
        vertices.append(new_v)

        if parent_vid:
            edges.append({
                "source": parent_vid,
                "target": vid,
                "relation": "part of",
            })

    return graph_data


def build_fast_yolo_graph(img_path: Path, tag_item: dict, yolo_dets: list[dict]) -> dict:
    """Build a lightweight scene graph from CLTagger tags and YOLO detections without running VLM/GBC."""
    tags_info = tag_item.get("tags", {})
    char_dict = tags_info.get("character", {})
    copy_dict = tags_info.get("copyright", {})
    gen_dict = tags_info.get("general", {})

    sorted_chars = [t for t, _ in sorted(char_dict.items(), key=lambda kv: -kv[1])]
    sorted_copys = [t for t, _ in sorted(copy_dict.items(), key=lambda kv: -kv[1])]
    sorted_gens = [t for t, _ in sorted(gen_dict.items(), key=lambda kv: -kv[1])][:15]

    char_str = ", ".join(t.replace("_", " ") for t in sorted_chars[:3]) if sorted_chars else "Anime character"
    copy_str = f" from {', '.join(t.replace('_', ' ') for t in sorted_copys[:2])}" if sorted_copys else ""
    top_tags_str = ", ".join(t.replace("_", " ") for t in sorted_gens)

    detail_cap = f"{char_str}{copy_str}. Visual features: {top_tags_str}."
    short_cap = char_str

    root_vertex = {
        "vertex_id": "",
        "label": "image",
        "bbox": {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0, "confidence": None},
        "descs": [{"text": detail_cap}],
    }

    graph = {
        "img_path": str(img_path),
        "detail_caption": detail_cap,
        "short_caption": short_cap,
        "vertices": [root_vertex],
        "edges": [],
    }

    char_vid = "character"
    if sorted_chars:
        clean_char = re.sub(r"[^\w\-]", "_", sorted_chars[0].split("(")[0].strip())[:25]
        char_vid = clean_char
    elif "1girl" in sorted_gens:
        char_vid = "girl"
    elif "1boy" in sorted_gens:
        char_vid = "boy"

    if yolo_dets:
        xs = [d["bbox"]["left"] for d in yolo_dets] + [d["bbox"]["right"] for d in yolo_dets]
        ys = [d["bbox"]["top"] for d in yolo_dets] + [d["bbox"]["bottom"] for d in yolo_dets]
        pad = 0.05
        char_bbox = {
            "left": round(max(0.0, min(xs) - pad), 4),
            "top": round(max(0.0, min(ys) - pad), 4),
            "right": round(min(1.0, max(xs) + pad), 4),
            "bottom": round(min(1.0, max(ys) + pad), 4),
            "confidence": 0.90,
        }
    else:
        char_bbox = {"left": 0.1, "top": 0.05, "right": 0.9, "bottom": 0.95, "confidence": 0.85}

    graph["vertices"].append({
        "vertex_id": char_vid,
        "label": "entity",
        "bbox": char_bbox,
        "descs": [{"text": f"Character {char_str}{copy_str}."}],
    })
    graph["edges"].append({"source": "", "target": char_vid, "relation": "contains"})

    merge_yolo_to_graph(graph, yolo_dets)
    return graph


def run_gbc(
    input_file: Path,
    output_file: Path,
    base_config: Path,
    report_dir: Path | None = None,
) -> None:
    """Run KohakuGBC scene graph refinement."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Clean old sink files to prevent append mode accumulation
    stem = output_file.with_suffix("")
    for f in [output_file, stem.with_name(stem.name + ".stats.jsonl"), stem.with_name(stem.name + ".failures.jsonl")]:
        if f.exists():
            f.unlink()

    # Generate layer override config
    override_config_path = output_file.parent / "pipeline_override.py"
    report_path_str = str(report_dir) if report_dir else str(output_file.parent / "report")
    override_content = f"""# Auto-generated pipeline config override
source = {{
    "kind": "jsonl",
    "path": {json.dumps(str(input_file))},
}}

sink = {{
    "kind": "graphs",
    "path": {json.dumps(str(output_file))},
}}

machine = {{
    "report": {json.dumps(report_path_str)},
}}

controls = {{
    "tag_hint": True,
}}
"""
    override_config_path.write_text(override_content, encoding="utf-8")

    cmd = [
        str(WORKSPACE_ROOT / ".venv" / "bin" / "gbc"),
        "run",
        str(base_config),
        str(override_config_path),
    ]

    print(f"\n[*] Starting KohakuGBC refinement...")
    print(f"    Base config: {base_config}")
    print(f"    Input: {input_file}")
    print(f"    Output: {output_file}")
    start_t = time.time()
    res = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"KohakuGBC failed with return code {res.returncode}")
    print(f"[*] KohakuGBC finished in {time.time() - start_t:.1f}s")


def print_summary(graphs_file: Path) -> None:
    """Print readable summary of the generated scene graphs."""
    if not graphs_file.exists():
        print(f"[Error] Output file {graphs_file} not found.", file=sys.stderr)
        return

    print("\n" + "=" * 70)
    print("                     PIPELINE RUN SUMMARY")
    print("=" * 70)

    count = 0
    with open(graphs_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            count += 1
            img_path = data.get("img_path")
            detail = data.get("detail_caption")
            short = data.get("short_caption")
            vertices = data.get("vertices", [])

            entities = [v for v in vertices if v.get("label") == "entity"]
            compositions = [v for v in vertices if v.get("label") == "composition"]
            relations = [v for v in vertices if v.get("label") == "relation"]

            print(f"\n[{count}] Image: {img_path}")
            print(f"--- Detail Caption ---")
            print(detail)
            print(f"\n--- Short Caption ---")
            print(short)
            print(f"\n--- Scene Graph Statistics ---")
            print(f"Total Vertices: {len(vertices)} | Entities: {len(entities)} | Compositions: {len(compositions)} | Relations: {len(relations)}")

            if entities:
                print(f"\nDetected Entities:")
                for e in entities[:6]:
                    vid = e.get("vertex_id")
                    bbox = e.get("bbox", {})
                    box_str = f"[{bbox.get('left', 0):.2f}, {bbox.get('top', 0):.2f}, {bbox.get('right', 0):.2f}, {bbox.get('bottom', 0):.2f}]"
                    descs = e.get("descs", [])
                    desc_text = descs[0].get("text") if descs else ""
                    print(f"  • {vid:15s} bbox={box_str}: {desc_text[:90]}...")

            if relations:
                print(f"\nKey Relations:")
                for r in relations[:4]:
                    vid = r.get("vertex_id")
                    descs = r.get("descs", [])
                    desc_text = descs[0].get("text") if descs else ""
                    print(f"  • {vid:20s}: {desc_text}")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLTagger + Ultralytics YOLO + KohakuGBC Automated Pipeline")
    parser.add_argument("images", nargs="+", help="Image file path(s) or directory containing images")
    parser.add_argument("--config", default="configs/machine/llama_test.py", help="Base machine config (default: configs/machine/llama_test.py)")
    parser.add_argument("--cltagger-dir", default=None, help="Directory containing CLTagger model.onnx and model_vocabulary.json")
    parser.add_argument("--output-dir", default=None, help="Explicit output directory (default: auto-create timestamped folder in runs/)")
    parser.add_argument("--name", default=None, help="Optional custom name prefix for auto-created run directory")
    parser.add_argument("--threshold-general", type=float, default=0.35, help="Confidence threshold for general tags")
    parser.add_argument("--threshold-character", type=float, default=0.40, help="Confidence threshold for character tags")
    parser.add_argument("--no-gbc", action="store_true", help="Only run CLTagger and YOLO priors without running gbc")
    parser.add_argument("--yolo-dir", default=None, help="Directory containing Ultralytics YOLO models (default: models/ultralytics)")
    parser.add_argument("--yolo-models", nargs="*", default=None, help="Specific YOLO model names to use (default: use all models in yolo-dir)")
    parser.add_argument("--no-yolo", action="store_true", help="Disable Ultralytics YOLO spatial priors")
    parser.add_argument("--yolo-conf", type=float, default=0.20, help="Confidence threshold for YOLO detections (default: 0.20)")
    parser.add_argument("--yolo-imgsz", type=int, default=1280, help="Inference resolution for YOLO models (default: 1280)")
    parser.add_argument("--yolo-device", default="auto", help="Device for YOLO models: 'auto', 'cuda:0', or 'cpu' (default: auto)")
    args = parser.parse_args()

    # Collect images
    images = collect_images(args.images)
    if not images:
        print("[Error] No valid images found to process.", file=sys.stderr)
        sys.exit(1)

    # Determine unique output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        if args.name:
            folder_name = f"{args.name}_{timestamp}"
        elif len(images) == 1:
            safe_stem = re.sub(r"[^\w\-.]", "_", images[0].stem)[:40]
            folder_name = f"{safe_stem}_{timestamp}"
        else:
            folder_name = f"run_{timestamp}"
        output_dir = (WORKSPACE_ROOT / "runs" / folder_name).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    input_file = output_dir / "input.jsonl"
    graphs_file = output_dir / "graphs.jsonl"
    yolo_file = output_dir / "yolo_priors.json"
    base_config = (WORKSPACE_ROOT / args.config).resolve()

    # Update runs/latest symlink for convenience
    latest_symlink = WORKSPACE_ROOT / "runs" / "latest"
    try:
        if latest_symlink.is_symlink() or latest_symlink.exists():
            latest_symlink.unlink()
        latest_symlink.symlink_to(output_dir, target_is_directory=True)
    except Exception:
        pass

    # Stage 1a: Standalone CLTagger
    tag_results = run_cl_tagger(
        images,
        threshold_general=args.threshold_general,
        threshold_character=args.threshold_character,
        model_dir=args.cltagger_dir,
    )

    # Stage 1b: Ultralytics YOLO Spatial Priors
    yolo_results_by_img: dict[str, list[dict]] = {}
    if not args.no_yolo:
        yolo_ensemble = UltralyticsEnsemble(
            model_dir=args.yolo_dir,
            model_names=args.yolo_models,
            device=args.yolo_device,
            conf=args.yolo_conf,
            imgsz=args.yolo_imgsz,
        )
        if yolo_ensemble.is_available:
            print(f"\n[*] Running Ultralytics YOLO ensemble on {len(images)} image(s)...")
            for img_p in images:
                dets = yolo_ensemble.detect(img_p)
                yolo_results_by_img[str(img_p)] = dets
                det_summary = ", ".join(f"{d['class']} ({d['conf']:.2f})" for d in dets[:6])
                print(f"[YOLO] Image: {img_p.name} -> {len(dets)} parts detected: {det_summary or 'None'}")
            yolo_file.write_text(json.dumps(yolo_results_by_img, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[+] Saved YOLO spatial priors to: {yolo_file}")

    # Inject spatial priors into tag_results
    for item in tag_results:
        img_key = item.get("image", "")
        item["spatial_priors"] = yolo_results_by_img.get(img_key, [])

    with open(input_file, "w", encoding="utf-8") as f:
        for item in tag_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n[+] Saved preliminary tags & priors to: {input_file}")

    if args.no_gbc:
        print("\n[*] --no-gbc specified. Generating fast scene graph from priors...")
        fast_graphs = []
        for item in tag_results:
            img_p = Path(item["image"])
            fast_g = build_fast_yolo_graph(img_p, item, item.get("spatial_priors", []))
            fast_graphs.append(fast_g)

        with open(graphs_file, "w", encoding="utf-8") as f:
            for g in fast_graphs:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")

        print_summary(graphs_file)
        print(f"[+] Output directory: {output_dir}")
        print(f"[+] Fast Graphs:      {graphs_file}")
        print(f"[+] Latest symlink:   {latest_symlink / 'graphs.jsonl'}")

        # Auto-generate interactive HTML viewer
        try:
            from visualize_graph import generate_html, load_image_base64
            with open(graphs_file, "r", encoding="utf-8") as f:
                first_data = json.loads(f.readline().strip())
            img_b64, orig_w, orig_h = load_image_base64(first_data.get("img_path", ""))
            html_code = generate_html(first_data, img_b64, orig_w, orig_h)
            viewer_path = output_dir / "viewer.html"
            viewer_path.write_text(html_code, encoding="utf-8")
            latest_html = latest_symlink / "viewer.html"
            if latest_html.parent.exists() and viewer_path != latest_html:
                latest_html.write_text(html_code, encoding="utf-8")
            print(f"[+] Visualizer HTML:  {viewer_path}")
            print(f"[+] Latest HTML:      {latest_html}")
        except Exception as e:
            print(f"[Warning] Failed to generate viewer.html: {e}", file=sys.stderr)
        return

    # Stage 2: KohakuGBC
    run_gbc(
        input_file=input_file,
        output_file=graphs_file,
        base_config=base_config,
        report_dir=output_dir / "report",
    )

    # Post-process: merge YOLO detections into KohakuGBC graphs
    if yolo_results_by_img:
        merged_lines = []
        with open(graphs_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                g_data = json.loads(line)
                img_key = g_data.get("img_path", "")
                y_dets = yolo_results_by_img.get(img_key, [])
                merged_g = merge_yolo_to_graph(g_data, y_dets)
                merged_lines.append(json.dumps(merged_g, ensure_ascii=False) + "\n")
        with open(graphs_file, "w", encoding="utf-8") as f:
            f.writelines(merged_lines)

    # Stage 3: Summary
    print_summary(graphs_file)
    print(f"[+] Output directory: {output_dir}")
    print(f"[+] Graphs:           {graphs_file}")
    print(f"[+] Latest symlink:   {latest_symlink / 'graphs.jsonl'}")

    # Stage 4: Auto-generate interactive HTML viewer
    try:
        try:
            from scripts.visualize_graph import generate_html, load_image_base64
        except ImportError:
            from visualize_graph import generate_html, load_image_base64
        with open(graphs_file, "r", encoding="utf-8") as f:
            first_data = json.loads(f.readline().strip())
        img_b64, orig_w, orig_h = load_image_base64(first_data.get("img_path", ""))
        html_code = generate_html(first_data, img_b64, orig_w, orig_h)
        viewer_path = output_dir / "viewer.html"
        viewer_path.write_text(html_code, encoding="utf-8")
        latest_html = latest_symlink / "viewer.html"
        if latest_html.parent.exists() and viewer_path != latest_html:
            latest_html.write_text(html_code, encoding="utf-8")
        print(f"[+] Visualizer HTML:  {viewer_path}")
        print(f"[+] Latest HTML:      {latest_html}")
    except Exception as e:
        print(f"[Warning] Failed to generate viewer.html: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

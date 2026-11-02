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
import base64
import concurrent.futures
import io
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


def parse_vlm_json(content: str) -> dict:
    """Robustly parse JSON object from VLM string response."""
    if not content:
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def extract_caption_from_reasoning(reasoning: str, cls_name: str) -> tuple[str, str]:
    """Fallback extraction of visual features from VLM reasoning tokens when JSON format was interrupted."""
    if not reasoning:
        return f"Anime character {cls_name}.", f"anime {cls_name}"
    clean = re.sub(r"[\*\#\`]", "", reasoning)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if len(s.strip()) > 20]
    good_s = [
        s for s in sentences
        if not s.lower().startswith("the user")
        and not s.lower().startswith("let's")
        and not s.lower().startswith("we need")
    ]
    if good_s:
        detailed = " ".join(good_s[-2:])
    elif sentences:
        detailed = sentences[-1]
    else:
        detailed = f"Anime character {cls_name} with distinct styling."
    short = f"anime {cls_name}"
    return detailed[:300], short


def describe_yolo_detections(
    image_path: Path,
    detections: list[dict],
    endpoint: str = "http://127.0.0.1:8080/v1",
    model_name: str | None = None,
    max_workers: int = 4,
    char_context: str | None = None,
) -> list[dict]:
    """Use the active VLM server to generate detailed and short captions for YOLO-detected bounding boxes."""
    if not detections:
        return detections

    try:
        from PIL import Image
    except ImportError:
        return detections

    import httpx

    if not check_vlm_reachability(endpoint):
        return detections

    model = model_name or os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[Warning] Could not load image for VLM crop description: {e}", file=sys.stderr)
        return detections

    w, h = img.size

    def _describe_one(det: dict) -> dict:
        cls_name = det.get("class", "object")
        box = det.get("bbox", {})
        l, t, r, b = box.get("left", 0.0), box.get("top", 0.0), box.get("right", 1.0), box.get("bottom", 1.0)
        bw, bh = max(0.001, r - l), max(0.001, b - t)

        # 15% contextual padding
        pad_x, pad_y = bw * 0.15, bh * 0.15
        x1 = max(0, int((l - pad_x) * w))
        y1 = max(0, int((t - pad_y) * h))
        x2 = min(w, int((r + pad_x) * w))
        y2 = min(h, int((b + pad_y) * h))

        crop = img.crop((x1, y1, x2, y2))
        if crop.width < 224 or crop.height < 224:
            scale = max(224 / max(1, crop.width), 224 / max(1, crop.height))
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt_focus = f"{cls_name}"
        if char_context:
            prompt_focus += f" of character '{char_context}'"

        system_msg = (
            "You are an expert anime character and visual art analyzer. "
            "When given a cropped image patch, describe the specific visual feature in concise, precise detail. "
            "Output JSON only."
        )
        user_msg = (
            f"Focus on the {prompt_focus} shown in this cropped image patch.\n"
            "Describe its detailed visual appearance (color, shape, styling, texture, highlights, expression or decorations).\n"
            "Return a valid JSON object with exactly two keys:\n"
            "- \"detailed_caption\": 1-2 detailed sentences describing the visual features.\n"
            "- \"short_caption\": A concise phrase (3-8 words) summarizing the appearance."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_msg},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 1800,
            "temperature": 0.2,
        }

        det_copy = dict(det)
        try:
            resp = httpx.post(f"{endpoint.rstrip('/')}/chat/completions", json=payload, timeout=90.0)
            if resp.status_code == 200:
                c = resp.json()["choices"][0]["message"]
                raw_content = c.get("content", "")
                parsed = parse_vlm_json(raw_content)

                detailed = parsed.get("detailed_caption")
                short = parsed.get("short_caption")

                # Fallback to reasoning extraction if JSON was incomplete
                if not detailed and c.get("reasoning_content"):
                    detailed, short = extract_caption_from_reasoning(c["reasoning_content"], cls_name)

                if not detailed:
                    detailed = f"Anime character {cls_name} detected with confidence {det.get('conf', 0):.2f}."
                if not short:
                    short = f"anime {cls_name}"

                det_copy["descs"] = [
                    {
                        "label": "detail",
                        "text": detailed,
                        "type": "vlm_yolo_desc",
                    },
                    {
                        "label": "short",
                        "text": short,
                        "type": "vlm_yolo_desc",
                    },
                    {
                        "label": "source",
                        "text": f"Character {cls_name} detected via {det.get('model', 'yolo')} (confidence: {det.get('conf', 0):.2f}).",
                        "type": "yolo_prior",
                    },
                ]
            else:
                det_copy["descs"] = [
                    {
                        "label": "detail",
                        "text": f"Character {cls_name} detected via {det.get('model')} (confidence: {det.get('conf', 0):.2f}).",
                        "type": "yolo_prior",
                    }
                ]
        except Exception as e:
            det_copy["descs"] = [
                {
                    "label": "detail",
                    "text": f"Character {cls_name} detected via {det.get('model')} (confidence: {det.get('conf', 0):.2f}).",
                    "type": "yolo_prior",
                }
            ]
        return det_copy

    print(f"[*] Generating VLM visual descriptions for {len(detections)} YOLO detection(s) on {image_path.name}...")
    start_t = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(detections))) as executor:
        results = list(executor.map(_describe_one, detections))
    print(f"[+] Described {len(results)} YOLO detection(s) in {time.time() - start_t:.1f}s")
    return results


def merge_yolo_to_graph(graph_data: dict, yolo_detections: list[dict]) -> dict:
    """Merge YOLO spatial prior detections into scene graph vertices and edges."""
    if not yolo_detections:
        return graph_data

    vertices = graph_data.setdefault("vertices", [])
    edges = graph_data.setdefault("edges", [])

    # Find character or girl/person vertex to act as parent
    parent_vid = None
    # 1. Look for entity matching character keywords
    for v in vertices:
        v_label = v.get("label")
        vid = v.get("vertex_id", "")
        if v_label == "entity" and any(k in vid.lower() for k in ["girl", "boy", "character", "person", "woman", "man"]):
            parent_vid = vid
            break

    # 2. Look for largest entity by bounding box area (e.g. named character entity like 'angelina')
    if not parent_vid:
        candidate_entities = []
        for v in vertices:
            vid = v.get("vertex_id", "")
            if v.get("label") == "entity" and vid and v.get("bbox"):
                b = v["bbox"]
                area = abs((b.get("right", 0) - b.get("left", 0)) * (b.get("bottom", 0) - b.get("top", 0)))
                candidate_entities.append((area, vid))
        if candidate_entities:
            candidate_entities.sort(key=lambda x: -x[0])
            parent_vid = candidate_entities[0][1]

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

        descs = det.get("descs")
        if not descs:
            descs = [
                {
                    "label": "detail",
                    "text": f"Character {cls_name} detected via {det['model']} (confidence: {det['conf']:.2f}).",
                    "type": "yolo_prior",
                }
            ]

        new_v = {
            "vertex_id": vid,
            "label": "entity",
            "bbox": det["bbox"],
            "descs": descs,
        }
        vertices.append(new_v)

        if parent_vid:
            edges.append({
                "source": parent_vid,
                "target": vid,
                "relation": "part of",
            })
            for v in vertices:
                if v.get("vertex_id") == parent_vid:
                    out_edges = v.setdefault("out_edges", [])
                    out_edges.append({
                        "source": parent_vid,
                        "text": cls_name,
                        "target": vid,
                    })
                    sub_masks = v.setdefault("sub_masks", [])
                    if vid not in sub_masks:
                        sub_masks.append(vid)
                    break

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


def check_vlm_reachability(endpoint: str = "http://127.0.0.1:8080/v1") -> bool:
    """Check if the VLM server endpoint is responding."""
    import httpx

    url = endpoint.rstrip("/")
    # Try models endpoint or root
    for test_path in ["/models", "", "/health"]:
        try:
            r = httpx.get(f"{url}{test_path}", timeout=1.5)
            if r.status_code < 500:
                return True
        except Exception:
            pass
    return False


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
    failures_file = stem.with_name(stem.name + ".failures.jsonl")
    for f in [output_file, stem.with_name(stem.name + ".stats.jsonl"), failures_file]:
        if f.exists():
            f.unlink()

    # Pre-flight VLM health check
    vlm_endpoint = os.environ.get("VLM_ENDPOINT", "http://127.0.0.1:8080/v1")
    if not check_vlm_reachability(vlm_endpoint):
        print("\n" + "!" * 70, file=sys.stderr)
        print(f"[Error] VLM server at '{vlm_endpoint}' is NOT reachable or offline!", file=sys.stderr)
        print("KohakuGBC requires an active VLM service (llama-server or vLLM) to generate scene graphs.", file=sys.stderr)
        print("\nTo start llama-server locally on RTX 5070 / CUDA:", file=sys.stderr)
        print("  /root/llama.cpp/build/bin/llama-server \\", file=sys.stderr)
        print("    -m /usr/local/gguf/qwen3.5-9b/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf \\", file=sys.stderr)
        print("    --mmproj /usr/local/gguf/qwen3.5-9b/mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf \\", file=sys.stderr)
        print("    --reasoning-budget 3500 --ctx-size 8000 --repeat-penalty 1.2 --port 8080\n", file=sys.stderr)
        print("Or if you only want fast prior tags & Ultralytics YOLO bounding boxes, run with '--no-gbc':", file=sys.stderr)
        print("  python scripts/cl_gbc_pipeline.py <IMAGE_PATH> --no-gbc", file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

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

    # Post-check: ensure output graphs.jsonl was generated
    if not output_file.exists() or output_file.stat().st_size == 0:
        fail_msg = "Unknown error"
        if failures_file.exists():
            try:
                fail_lines = [json.loads(l) for l in failures_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                fail_msg = "; ".join(f"{f.get('key')}: {f.get('kind')} - {f.get('message')}" for f in fail_lines)
            except Exception:
                fail_msg = failures_file.read_text(encoding="utf-8")[:300]
        print(f"\n[Error] KohakuGBC completed without generating graphs.jsonl!", file=sys.stderr)
        print(f"Recorded failures: {fail_msg}", file=sys.stderr)
        if "no described record" in fail_msg:
            print("[Hint] 'no described record' usually indicates the VLM server did not return a valid response or timed out.", file=sys.stderr)
        sys.exit(1)


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
    parser.add_argument("--describe-yolo", action=argparse.BooleanOptionalAction, default=True, help="Use VLM to describe YOLO-detected regions in detail (default: True)")
    parser.add_argument("--yolo-vlm-workers", type=int, default=4, help="Concurrent workers for VLM crop descriptions (default: 4)")
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

            # Stage 1c: VLM visual descriptions for YOLO crops
            if args.describe_yolo:
                vlm_endpoint = os.environ.get("VLM_ENDPOINT", "http://127.0.0.1:8080/v1")
                if check_vlm_reachability(vlm_endpoint):
                    for item in tag_results:
                        img_k = item.get("image", "")
                        p = Path(img_k)
                        dets = yolo_results_by_img.get(img_k, [])
                        if dets:
                            top_chars = list(item.get("tags", {}).get("character", {}).keys())
                            char_ctx = top_chars[0] if top_chars else None
                            described_dets = describe_yolo_detections(
                                image_path=p,
                                detections=dets,
                                endpoint=vlm_endpoint,
                                max_workers=args.yolo_vlm_workers,
                                char_context=char_ctx,
                            )
                            yolo_results_by_img[img_k] = described_dets
                    yolo_file.write_text(json.dumps(yolo_results_by_img, indent=2, ensure_ascii=False), encoding="utf-8")

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

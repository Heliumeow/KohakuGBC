from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

from .utils import WORKSPACE_ROOT


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

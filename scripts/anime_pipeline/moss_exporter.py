"""MOSS-Image Layout Draft Exporter & Compiler for KohakuGBC.

Strictly adheres to OpenMOSS MOSS-Image layout specification:
- Draft schema: width, height, bbox_coord_space ("abs"), bbox_format ("xyxy"), summary, items
- Compiler schema: abs_centralized, yxyx, 16-pixel multiples
- Multi-granularity anti-contention filtering (macro, fine, all, dropout)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fields(value: dict, allowed: set[str], required: set[str], location: str) -> None:
    require(isinstance(value, dict), f"{location} must be an object")
    unknown = set(value.keys()) - allowed
    missing = required - set(value.keys())
    require(not unknown, f"{location}: unknown fields {sorted(unknown)}")
    require(not missing, f"{location}: missing fields {sorted(missing)}")


def text(value: str, location: str) -> None:
    require(isinstance(value, str) and bool(value.strip()), f"{location} must be nonempty text")


def strings(value: list, location: str) -> None:
    require(isinstance(value, list), f"{location} must be an array")
    for index, item in enumerate(value):
        text(item, f"{location}[{index}]")


def centered_box(box: list[int | float], width: int, height: int, location: str) -> list[int]:
    require(isinstance(box, list) and len(box) == 4, f"{location} must be [x1,y1,x2,y2]")
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in box),
        f"{location} must contain four finite numbers",
    )
    x1, y1, x2, y2 = box
    require(x1 < x2 and y1 < y2, f"{location} must have positive width and height")
    require(
        0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
        f"{location} exceeds the {width}x{height} canvas (got {box})",
    )
    result = [
        math.floor(round(y1, 9)) - height // 2,
        math.floor(round(x1, 9)) - width // 2,
        math.ceil(round(y2, 9)) - height // 2,
        math.ceil(round(x2, 9)) - width // 2,
    ]
    require(
        result[0] < result[2] and result[1] < result[3],
        f"{location} is too small at export precision",
    )
    return result


def compile_moss_prompt(draft: dict) -> str:
    """Compile a caption or absolute-pixel Layout draft into a MOSS-Image prompt.

    Conforms 100% to OpenMOSS compile_prompt.py reference implementation.
    """
    require(isinstance(draft, dict), "Draft must be a JSON object")
    if "caption" in draft:
        fields(draft, {"caption"}, {"caption"}, "Caption draft")
        text(draft["caption"], "caption")
        return draft["caption"]

    root_fields = {"width", "height", "bbox_coord_space", "bbox_format", "summary", "items"}
    fields(draft, root_fields, root_fields, "Layout draft")
    width, height = draft["width"], draft["height"]
    for name, value in (("width", width), ("height", height)):
        require(
            type(value) is int and value > 0 and value % 16 == 0,
            f"{name} must be a positive integer multiple of 16 (got {value})",
        )
    require(draft["bbox_coord_space"] == "abs", "Draft bbox_coord_space must be abs")
    require(draft["bbox_format"] == "xyxy", "Draft bbox_format must be xyxy")

    summary = draft["summary"]
    fields(summary, {"caption", "meta", "canvas", "palette"}, {"caption"}, "summary")
    text(summary["caption"], "summary.caption")
    if "canvas" in summary:
        require(
            summary["canvas"] in ("single", "multi_panel", "collage"),
            "summary.canvas must be single, multi_panel or collage",
        )
    for key in ("meta", "palette"):
        if key in summary:
            strings(summary[key], f"summary.{key}")

    items = draft["items"]
    require(isinstance(items, list) and bool(items), "items must be a nonempty flat array")
    compiled_items = []
    for index, item in enumerate(items):
        location = f"items[{index}]"
        fields(
            item,
            {"label", "caption", "bbox", "text", "tags", "palette"},
            {"label", "caption", "bbox"},
            location,
        )
        require(
            item["label"] in ("object", "character", "text"),
            f"{location}.label must be object, character or text (got {item['label']})",
        )
        text(item["caption"], f"{location}.caption")
        if item["label"] == "text":
            require("text" in item, f"{location}: text items require literal text")
        if "text" in item:
            text(item["text"], f"{location}.text")
        for key in ("tags", "palette"):
            if key in item:
                strings(item[key], f"{location}.{key}")
        compiled_items.append({
            key: centered_box(value, width, height, f"{location}.bbox") if key == "bbox" else value
            for key, value in item.items()
        })

    result = {
        "summary": summary,
        "items": compiled_items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def get_image_size(img_path: str | Path | None) -> tuple[int, int]:
    """Retrieve (width, height) of image, returning (1024, 1024) fallback on failure."""
    if not img_path:
        return 1024, 1024
    p = Path(img_path)
    if not p.is_file():
        return 1024, 1024
    try:
        from PIL import Image

        with Image.open(p) as img:
            return img.size
    except Exception:
        return 1024, 1024


def align_dimension_16(dim: int, min_val: int = 256) -> int:
    """Align canvas dimension to nearest multiple of 16."""
    val = max(min_val, round(dim / 16.0) * 16)
    return int(val)


def _compute_box_overlap(box_a: list[int], box_b: list[int]) -> float:
    """Compute ratio of Box B covered by Box A (Intersection / Area B)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return intersection / b_area


def _clean_caption_text(desc_item: dict | str, cls_fallback: str = "item", language: str = "zh") -> str:
    """Extract clean, descriptive text removing raw detector logs."""
    raw = desc_item.get("text", "") if isinstance(desc_item, dict) else str(desc_item)
    if not raw or not raw.strip():
        readable = cls_fallback.replace("_", " ").strip()
        if language == "zh":
            return f"清晰生动的二次元【{readable}】表现。"
        return f"A distinct anime {readable}."

    # If already a high-quality VLM description
    if isinstance(desc_item, dict) and desc_item.get("type") == "vlm_yolo_desc":
        clean = raw.strip()
        if len(clean) > 5 and not clean.startswith("Character"):
            return clean

    # Strip detector template strings (Chinese / English)
    m_zh = re.search(r"角色【?([\w\-]+)】?通过模型.*检测识别", raw)
    if m_zh:
        part_name = m_zh.group(1).replace("_", " ")
        return f"二次元角色【{part_name}】特写细节。"

    m = re.search(r"Character\s+([\w\-]+)\s+detected\s+via", raw, re.IGNORECASE)
    if m:
        part_name = m.group(1).replace("_", " ")
        if language == "zh":
            return f"二次元角色【{part_name}】特写细节。"
        return f"Close-up depiction of character {part_name}."

    cleaned = re_detector_source.sub("", raw).strip()
    if not cleaned or len(cleaned) < 3:
        readable = cls_fallback.replace("_", " ").strip()
        if language == "zh":
            cleaned = f"清晰可见的动漫【{readable}】细节。"
        else:
            cleaned = f"A visible anime {readable}."
    return cleaned


import re

re_detector_source = re.compile(
    r"(?:Character\s+)?[\w\-.]+\s+detected\s+via\s+[\w\-.]+\s*(?:\(confidence:\s*[0-9.]+\))?[.]?",
    re.IGNORECASE,
)


def graph_to_moss_draft(
    graph_data: dict,
    granularity: str = "macro",
    target_canvas: tuple[int, int] | None = None,
    meta_tags: list[str] | None = None,
    language: str = "zh",
) -> dict:
    """Convert GBC scene graph into a strictly compliant OpenMOSS Layout draft.

    Args:
        graph_data: Graph dictionary with vertices, edges, captions.
        granularity:
            - 'macro': (Default) Coarse entities (characters and distinct props), dropping heavily nested sub-parts.
            - 'fine': Micro visual parts (eyes, mouth, accessories), suppressing enclosing macro boxes to prevent attention conflict.
            - 'all': All valid nodes (no pruning).
            - 'dropout': Caption-only draft for text-only baseline retention.
        target_canvas: Optional explicit (width, height) multiples of 16.
        meta_tags: Optional medium/genre strings for summary.meta.
        language: 'zh' (Chinese) or 'en' (English).

    Returns:
        A draft dict compliant with OpenMOSS layout.md.
    """
    # 1. Summary Caption & Meta
    default_caption = "精美的二次元动漫数码插画。" if language == "zh" else "An anime digital illustration."
    summary_caption = (
        graph_data.get("detail_caption")
        or graph_data.get("short_caption")
        or default_caption
    ).strip()

    # Absorb semantic relations (excluding structural 'contains' or 'part of')
    rel_clauses = []
    for edge in graph_data.get("edges", []):
        rel = edge.get("relation", "").strip().lower()
        src = edge.get("source", "").strip()
        tgt = edge.get("target", "").strip()
        if rel and rel not in ("contains", "part of", "inside", "has part", "submask"):
            if language == "zh":
                clean_rel = f"{src} 与 {tgt} 关联：{rel}".replace("_", " ").strip()
            else:
                clean_rel = f"{src} {rel} {tgt}".replace("_", " ").strip()
            if len(clean_rel) > 4:
                rel_clauses.append(clean_rel)

    if rel_clauses:
        sep = "；" if language == "zh" else "; "
        prefix = "包含关联特征：" if language == "zh" else "With "
        summary_caption = f"{summary_caption} {prefix}{sep.join(rel_clauses[:3])}。"

    if granularity == "dropout":
        return {"caption": summary_caption}

    # 2. Canvas Geometry
    if target_canvas:
        canvas_w = align_dimension_16(target_canvas[0])
        canvas_h = align_dimension_16(target_canvas[1])
    else:
        orig_w, orig_h = get_image_size(graph_data.get("img_path"))
        canvas_w = align_dimension_16(orig_w)
        canvas_h = align_dimension_16(orig_h)

    # 3. Extract Candidates from Vertices
    raw_candidates = []
    sub_parts_keywords = {
        "eyes", "eye", "mouth", "nose", "hand", "hands", "ear", "ears",
        "breast", "breasts", "penis", "pussy", "jewelry", "eyebrow", "hair",
        "face", "head", "accessory", "collar", "choker",
    }

    # Find parent character vertices from edges and sub_masks
    parent_vids = set()
    for e in graph_data.get("edges", []):
        if e.get("relation") in ("part of", "contains") and e.get("source"):
            parent_vids.add(e.get("source"))
    for v in graph_data.get("vertices", []):
        if v.get("sub_masks"):
            parent_vids.add(v.get("vertex_id"))

    for v in graph_data.get("vertices", []):
        vid = v.get("vertex_id", "")
        vlabel = v.get("label", "")
        # Skip root scene vertex
        if vid == "" or vlabel == "image":
            continue

        bbox_dict = v.get("bbox")
        if not bbox_dict:
            continue

        l = float(bbox_dict.get("left", 0.0))
        t = float(bbox_dict.get("top", 0.0))
        r = float(bbox_dict.get("right", 1.0))
        b = float(bbox_dict.get("bottom", 1.0))
        if r <= l or b <= t:
            continue

        # Scale and clamp to canvas
        x1 = max(0, min(canvas_w - 4, int(math.floor(l * canvas_w))))
        y1 = max(0, min(canvas_h - 4, int(math.floor(t * canvas_h))))
        x2 = max(x1 + 4, min(canvas_w, int(math.ceil(r * canvas_w))))
        y2 = max(y1 + 4, min(canvas_h, int(math.ceil(b * canvas_h))))

        # Determine label ('character', 'object', 'text')
        vid_lower = vid.lower()
        is_sub_part = any(part in vid_lower for part in sub_parts_keywords)
        descs = v.get("descs", [])
        desc_all_text = " ".join(
            (d.get("text", "") if isinstance(d, dict) else str(d)) for d in descs
        ).lower()

        if "text" in vid_lower or "caption" in vid_lower or "watermark" in vid_lower:
            item_label = "text"
        elif is_sub_part:
            # Sub-features are physical details categorized as 'object' in MOSS specification
            item_label = "object"
        elif (
            vid in parent_vids
            or any(k in vid_lower for k in ["girl", "boy", "person", "character", "woman", "man"])
            or "character " in desc_all_text
        ):
            item_label = "character"
        else:
            item_label = "object"

        # Extract best caption
        caption_text = ""
        # Prefer VLM detailed or short description
        for d in descs:
            if isinstance(d, dict) and d.get("type") == "vlm_yolo_desc":
                caption_text = d.get("text", "")
                break
        if not caption_text and descs:
            caption_text = _clean_caption_text(descs[0], cls_fallback=vid, language=language)
        if not caption_text:
            if language == "zh":
                caption_text = f"二次元动漫【{vid.replace('_', ' ')}】表现。"
            else:
                caption_text = f"An anime {item_label} representing {vid.replace('_', ' ')}."

        # Compute area
        area = (x2 - x1) * (y2 - y1)
        raw_candidates.append({
            "vid": vid,
            "label": item_label,
            "caption": caption_text,
            "bbox": [x1, y1, x2, y2],
            "area": area,
            "is_sub_part": is_sub_part,
            "is_macro": (item_label == "character") or (not is_sub_part and area > 0.01 * canvas_w * canvas_h),
        })

    # 4. Anti-Contention / Granularity Filtering
    final_items = []
    if granularity == "macro":
        # Keep macro entities (characters and distinct props), drop heavily contained sub-parts
        macros = [c for c in raw_candidates if not c["is_sub_part"]]
        if not macros:
            # Fallback to largest box
            macros = sorted(raw_candidates, key=lambda c: -c["area"])[:1]
        final_items = macros
    elif granularity == "fine":
        # Keep detailed sub-parts, prune enclosing macro boxes that swallow them
        sub_items = [c for c in raw_candidates if c["is_sub_part"]]
        if sub_items:
            # Drop parent boxes that swallow these sub-items to prevent attention contention
            retained = list(sub_items)
            for cand in raw_candidates:
                if not cand["is_sub_part"]:
                    # Check if cand swallows any sub_item (>50% of sub_item is inside cand)
                    swallows = any(_compute_box_overlap(cand["bbox"], s["bbox"]) > 0.5 for s in sub_items)
                    if not swallows:
                        retained.append(cand)
            final_items = retained
        else:
            final_items = raw_candidates
    else:  # 'all'
        final_items = raw_candidates

    # If still empty, create default canvas character item
    if not final_items:
        pad_x, pad_y = int(canvas_w * 0.1), int(canvas_h * 0.1)
        final_items = [
            {
                "label": "character",
                "caption": summary_caption[:180],
                "bbox": [pad_x, pad_y, canvas_w - pad_x, canvas_h - pad_y],
            }
        ]

    # 5. Format items according strictly to MOSS schema
    moss_items = []
    for item in final_items:
        clean_item = {
            "label": item["label"],
            "caption": item["caption"],
            "bbox": item["bbox"],
        }
        if item["label"] == "text":
            clean_item["text"] = item.get("text") or item["caption"]
        moss_items.append(clean_item)

    # 6. Construct summary
    summary_dict: dict = {
        "caption": summary_caption,
        "canvas": "single",
    }
    if meta_tags:
        summary_dict["meta"] = [str(m).strip() for m in meta_tags if str(m).strip()]

    return {
        "width": canvas_w,
        "height": canvas_h,
        "bbox_coord_space": "abs",
        "bbox_format": "xyxy",
        "summary": summary_dict,
        "items": moss_items,
    }


def export_moss_dataset(
    graphs: list[dict],
    output_dir: Path,
    granularity: str = "macro",
    target_canvas: tuple[int, int] | None = None,
    language: str = "zh",
) -> tuple[Path, Path]:
    """Export scene graphs to validated MOSS draft JSONL and compiled prompt JSONL.

    Returns:
        (drafts_path, compiled_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    drafts_file = output_dir / f"moss_draft_{granularity}.jsonl"
    compiled_file = output_dir / f"moss_compiled_{granularity}.jsonl"

    draft_lines = []
    compiled_lines = []

    for idx, g in enumerate(graphs):
        # Extract meta genres from tags if present
        meta_tags = None
        if "detail_caption" in g:
            if language == "zh":
                meta_tags = ["二次元插画", "数码立绘", "动漫画风"]
            else:
                meta_tags = ["anime illustration", "digital media"]

        draft = graph_to_moss_draft(
            graph_data=g,
            granularity=granularity,
            target_canvas=target_canvas,
            meta_tags=meta_tags,
            language=language,
        )

        # Validate with official compiler
        compiled_str = compile_moss_prompt(draft)

        draft_lines.append(json.dumps(draft, ensure_ascii=False) + "\n")
        compiled_lines.append(compiled_str + "\n")

    drafts_file.write_text("".join(draft_lines), encoding="utf-8")
    compiled_file.write_text("".join(compiled_lines), encoding="utf-8")

    return drafts_file, compiled_file

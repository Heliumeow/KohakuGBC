from __future__ import annotations

import re
from pathlib import Path


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

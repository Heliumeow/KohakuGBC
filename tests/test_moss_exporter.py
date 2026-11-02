import json
from pathlib import Path
import sys
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.anime_pipeline.moss_exporter import (
    align_dimension_16,
    compile_moss_prompt,
    graph_to_moss_draft,
)


def test_align_dimension_16():
    assert align_dimension_16(1080) == 1088
    assert align_dimension_16(1920) == 1920
    assert align_dimension_16(1024) == 1024
    assert align_dimension_16(1080) % 16 == 0


def test_moss_draft_and_compile():
    sample_graph = {
        "img_path": None,
        "detail_caption": "Anime hero standing in a field.",
        "vertices": [
            {
                "vertex_id": "",
                "label": "image",
                "bbox": {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0},
            },
            {
                "vertex_id": "hero_character",
                "label": "entity",
                "bbox": {"left": 0.3, "top": 0.2, "right": 0.7, "bottom": 0.8},
                "descs": [{"text": "Character hero in anime art style"}],
                "sub_masks": ["hero_eyes_0"],
            },
            {
                "vertex_id": "hero_eyes_0",
                "label": "entity",
                "bbox": {"left": 0.45, "top": 0.3, "right": 0.55, "bottom": 0.35},
                "descs": [{"text": "Detailed anime cyan eyes"}],
            },
        ],
        "edges": [
            {"source": "", "target": "hero_character", "relation": "contains"},
            {"source": "hero_character", "target": "hero_eyes_0", "relation": "part of"},
        ],
    }

    # Macro mode: should keep character and prune eyes
    d_macro = graph_to_moss_draft(sample_graph, granularity="macro", target_canvas=(1024, 1024))
    assert d_macro["width"] == 1024 and d_macro["height"] == 1024
    assert len(d_macro["items"]) == 1
    assert d_macro["items"][0]["label"] == "character"
    compiled_m = compile_moss_prompt(d_macro)
    assert "abs_centralized" in compiled_m
    assert "yxyx" in compiled_m

    # Fine mode: should keep eyes and prune enclosing character
    d_fine = graph_to_moss_draft(sample_graph, granularity="fine", target_canvas=(1024, 1024))
    assert len(d_fine["items"]) == 1
    assert d_fine["items"][0]["label"] == "object"
    assert "eyes" in d_fine["items"][0]["caption"].lower()
    compiled_f = compile_moss_prompt(d_fine)
    assert "abs_centralized" in compiled_f

    # Dropout mode
    d_drop = graph_to_moss_draft(sample_graph, granularity="dropout")
    assert "caption" in d_drop and "items" not in d_drop
    compiled_d = compile_moss_prompt(d_drop)
    assert compiled_d == d_drop["caption"]

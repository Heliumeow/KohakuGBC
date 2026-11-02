"""Anime image annotation pipeline subpackage for KohakuGBC.

Modular architecture components:
- cltagger: StandaloneCLTagger and tag extraction
- yolo: UltralyticsEnsemble multi-model anime prior detector with preset support
- vlm_descriptor: describe_yolo_detections and JSON/reasoning parsers
- graph_builder: merge_yolo_to_graph and build_fast_yolo_graph
- gbc_runner: run_gbc, check_vlm_reachability, print_summary
- utils: normalize_path, collect_images, WORKSPACE_ROOT
"""

from __future__ import annotations

from .cltagger import StandaloneCLTagger, extract_copyright_from_character, resolve_cltagger_model, run_cl_tagger
from .gbc_runner import check_vlm_reachability, print_summary, run_gbc
from .graph_builder import build_fast_yolo_graph, merge_yolo_to_graph
from .utils import WORKSPACE_ROOT, collect_images, normalize_path
from .vlm_descriptor import describe_yolo_detections, extract_caption_from_reasoning, parse_vlm_json
from .yolo import DEFAULT_PRESETS, UltralyticsEnsemble, load_presets, resolve_auto_preset

__all__ = [
    "WORKSPACE_ROOT",
    "normalize_path",
    "collect_images",
    "StandaloneCLTagger",
    "extract_copyright_from_character",
    "resolve_cltagger_model",
    "run_cl_tagger",
    "DEFAULT_PRESETS",
    "load_presets",
    "resolve_auto_preset",
    "UltralyticsEnsemble",
    "parse_vlm_json",
    "extract_caption_from_reasoning",
    "describe_yolo_detections",
    "merge_yolo_to_graph",
    "build_fast_yolo_graph",
    "check_vlm_reachability",
    "run_gbc",
    "print_summary",
]

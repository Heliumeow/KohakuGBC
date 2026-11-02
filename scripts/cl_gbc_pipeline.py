#!/usr/bin/env python3
"""CLTagger + Ultralytics YOLO + KohakuGBC Automated Image Annotation Pipeline.

High-level workflow coordinator:
1. CLTagger (Anime ONNX Classifier) extracts preliminary prior tags.
2. Ultralytics YOLO models detect fine-grained anime features (eyes, mouth, breasts, jewelry).
3. VLM Region Descriptor generates detailed & short captions for detected crops.
4. KohakuGBC / Fast Graph builds structured scene graphs & relationships.
5. Interactive HTML Visualizer (viewer.html) is generated.

Usage:
    python scripts/cl_gbc_pipeline.py <IMAGE_PATH_OR_DIR> [OPTIONS]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Ensure workspace root and scripts are in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
for p in [str(WORKSPACE_ROOT / "src"), str(WORKSPACE_ROOT), str(WORKSPACE_ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from scripts.anime_pipeline import (
        UltralyticsEnsemble,
        build_fast_yolo_graph,
        check_vlm_reachability,
        collect_images,
        describe_yolo_detections,
        export_moss_dataset,
        merge_yolo_to_graph,
        print_summary,
        run_cl_tagger,
        run_gbc,
    )
except ImportError:
    from anime_pipeline import (
        UltralyticsEnsemble,
        build_fast_yolo_graph,
        check_vlm_reachability,
        collect_images,
        describe_yolo_detections,
        export_moss_dataset,
        merge_yolo_to_graph,
        print_summary,
        run_cl_tagger,
        run_gbc,
    )


def _maybe_export_moss(args: argparse.Namespace, graphs_file: Path, output_dir: Path) -> None:
    """Conditionally export scene graphs to OpenMOSS Layout format."""
    if not args.export_moss:
        return
    print("\n[*] Exporting OpenMOSS Layout dataset...")
    all_graphs = []
    with open(graphs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_graphs.append(json.loads(line))
    if not all_graphs:
        print("[!] No scene graphs found to export for MOSS.", file=sys.stderr)
        return

    granularities = ["macro", "fine"] if args.moss_granularity == "pyramid" else [args.moss_granularity]
    for gran in granularities:
        d_file, c_file = export_moss_dataset(all_graphs, output_dir, granularity=gran, language=args.language)
        print(f"[+] MOSS Draft ({gran}):    {d_file}")
        print(f"[+] MOSS Compiled ({gran}): {c_file}")


def _generate_html_viewer(graphs_file: Path, output_dir: Path, latest_symlink: Path) -> None:
    """Generate interactive viewer.html from graphs.jsonl."""
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
    parser.add_argument(
        "--yolo-preset",
        default="anime-sfw",
        help="YOLO model preset: 'anime-sfw' (default), 'anime-nsfw', 'real', 'all', or 'auto'",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.20, help="Confidence threshold for YOLO detections (default: 0.20)")
    parser.add_argument("--yolo-imgsz", type=int, default=1280, help="Inference resolution for YOLO models (default: 1280)")
    parser.add_argument("--yolo-device", default="auto", help="Device for YOLO models: 'auto', 'cuda:0', or 'cpu' (default: auto)")
    parser.add_argument("--describe-yolo", action=argparse.BooleanOptionalAction, default=True, help="Use VLM to describe YOLO-detected regions in detail (default: True)")
    parser.add_argument("--yolo-vlm-workers", type=int, default=4, help="Concurrent workers for VLM crop descriptions (default: 4)")
    parser.add_argument("--export-moss", action="store_true", help="Export dataset in OpenMOSS Layout draft & compiled prompt format")
    parser.add_argument(
        "--moss-granularity",
        choices=["macro", "fine", "all", "pyramid", "dropout"],
        default="macro",
        help="Granularity for MOSS export: 'macro' (default), 'fine', 'all', 'pyramid' (macro+fine), or 'dropout'",
    )
    parser.add_argument(
        "--language",
        "--lang",
        dest="language",
        choices=["zh", "en"],
        default="zh",
        help="Language for prompts, VLM descriptions, and MOSS captions: 'zh' (default, Chinese) or 'en' (English)",
    )
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
            preset=args.yolo_preset,
            model_names=args.yolo_models,
            tag_hint=tag_results,
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
                                language=args.language,
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
            fast_g = build_fast_yolo_graph(img_p, item, item.get("spatial_priors", []), language=args.language)
            fast_graphs.append(fast_g)

        with open(graphs_file, "w", encoding="utf-8") as f:
            for g in fast_graphs:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")

        print_summary(graphs_file)
        print(f"[+] Output directory: {output_dir}")
        print(f"[+] Fast Graphs:      {graphs_file}")
        print(f"[+] Latest symlink:   {latest_symlink / 'graphs.jsonl'}")

        _maybe_export_moss(args, graphs_file, output_dir)
        _generate_html_viewer(graphs_file, output_dir, latest_symlink)
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

    # Stage 4: OpenMOSS Export (if requested)
    _maybe_export_moss(args, graphs_file, output_dir)

    # Stage 5: Auto-generate interactive HTML viewer
    _generate_html_viewer(graphs_file, output_dir, latest_symlink)


if __name__ == "__main__":
    main()

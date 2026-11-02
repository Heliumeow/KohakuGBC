from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .utils import WORKSPACE_ROOT
from .vlm_descriptor import check_vlm_reachability


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


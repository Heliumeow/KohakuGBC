# Anime Tagging Pipeline & Interactive Visualizer

This guide covers the two-stage anime image annotation pipeline in KohakuGBC, integrating
**CLTagger** (Anime ONNX classifier) as a visual prior with **KohakuGBC** (VLM + OWLv2/SAM 3 scene
graph generation) and the interactive HTML visualizer.

---

## 1. Overview & Architecture

Standard Vision-Language Models (VLMs) often struggle with fine-grained anime franchise lore,
frequently hallucinating character identities or mistaking lore-specific accessories (e.g. Arknights
infection monitors or headgear) for generic objects like headphones.

To resolve this, KohakuGBC provides an automated two-stage pipeline:

```
[ Input Image ]
       │
       ▼
[ Stage 1: CLTagger (ONNX) ] ──▶ Character, Copyright & Visual Tags with Confidences
       │
       ▼
[ Stage 2: KohakuGBC (VLM + Detector) ]
       │  • VLM receives visual tags as reference hints
       │  • Detail caption identifies specific character & franchise lore
       │  • Elements list retains standard COCO nouns for open-vocabulary detectors
       ▼
[ Output: Scene Graph (graphs.jsonl) ]
       │
       ▼
[ Stage 3: Interactive Visualizer (viewer.html) ]
       • SVG BBox overlays with color-coded categories
       • Hover/Click tooltips with localized descriptions
```

---

## 2. Quick Start

### 2.1 One-Click Pipeline Execution

Run the complete pipeline on a single image (supports Windows paths `C:\...` or Linux paths `/mnt/c/...`):

```bash
# Single image run
.venv/bin/python scripts/cl_gbc_pipeline.py "C:\Users\YourUsername\Pictures\your_image.png"

# Batch run on an entire folder
.venv/bin/python scripts/cl_gbc_pipeline.py "/mnt/c/Users/YourUsername/Pictures"

# Preliminary tagging only (generates input.jsonl without launching VLM/detector)
.venv/bin/python scripts/cl_gbc_pipeline.py "C:\path\to\image.png" --no-gbc
```

### 2.2 Output Structure

Each run automatically creates a dedicated, isolated directory under `runs/`:
- **Single image:** `runs/<image_name>_<YYYYMMDD_HHMMSS>/`
- **Batch:** `runs/run_<YYYYMMDD_HHMMSS>/`

Inside the run directory:
- `input.jsonl`: Filtered character, copyright, and visual general tags.
- `graphs.jsonl`: The resolved hierarchical scene graph.
- `viewer.html`: The standalone interactive visualizer.
- `report/`: Performance metrics and execution event logs.

> [!TIP]
> A symlink `runs/latest` is automatically created and updated on every run. You can always inspect the newest result at `runs/latest/graphs.jsonl` or `runs/latest/viewer.html`.

---

## 3. Interactive Scene Graph Visualizer

To visualize and inspect bounding boxes, relationships, and localized descriptions:

```bash
# 1. Generate or inspect the latest run
.venv/bin/python scripts/visualize_graph.py

# 2. Automatically open in your default browser (supports Windows host through WSL)
.venv/bin/python scripts/visualize_graph.py --open

# 3. Start a local HTTP server
.venv/bin/python scripts/visualize_graph.py --serve --port 8899
```

### Key UI Features:
- **Responsive SVG BBoxes**: Normalized coordinates mapped accurately to the image.
  - 🔹 **Entity (Cyan)**: Standalone physical objects (character, hair, weapons, clothes).
  - 🟣 **Composition (Purple)**: Part-whole groupings (twintails, horn structures).
  - 🔸 **Relation (Amber)**: Spatial interactions (holding weapons, wearing accessories).
- **Hover Tooltips**: Shows node ID, confidence score, bounding coordinates, and localized caption.
- **Bi-directional Focus**: Clicking a box highlights its node card in the sidebar; clicking a node card pulses its bounding box on the image.
- **Filter Toolbar**: Toggle visibility of entities, compositions, or relations, and filter by minimum confidence.

---

## 4. Prior Tag Hints in KohakuGBC Core

Prior knowledge injection is implemented across three core layers:

1. **`src/kohakugbc/program/gbc.kir`**:
   The `by_query` branch forwards prior tags to the root query request:
   ```kir
   (image, tags, patch, controls)image_query_request(request)
   ```

2. **`src/kohakugbc/nodes/root.py`**:
   `hint_tags(tags, threshold)` prioritizes tags in category order (`character`, `copyright`, `general`) and applies threshold filtering.

3. **`src/kohakugbc/nodes/specs/chat.py`**:
   The prompt injects prior tags as reference hints while instructing the VLM to:
   - Identify specific character names and franchises in the **detailed caption**.
   - Use standard detectable object nouns (e.g. `warrior`, `sword`, `horns`) in the `elements` list and concise caption brackets to ensure open-vocabulary detectors (OWLv2, SAM 3) can reliably locate them.

---

## 5. Engine & VRAM Allocation Recommendations

When running locally on consumer GPUs (e.g. RTX 5070 12GB):

| Setup | VLM (`vlm`) | Detector (`detect`) | Embedder (`embed`) | Single Image Speed |
|---|---|---|---|---|
| **Balanced (Recommended)** | `llama-server` (Qwen3.5-9B Q8_0, ~10GB VRAM) | `google/owlv2-base-patch16-ensemble` (CUDA, ~600MB) | `google/embeddinggemma-300m` (CUDA, ~600MB) | **~35 - 45 seconds** |
| **Fine-Grained Segmentation** | `llama-server` (Qwen3.5-9B, CPU or offloaded) | `sam3` (CUDA, ~3.5GB) | `google/embeddinggemma-300m` (CUDA) | **~15 - 25 seconds** |
| **CPU Fallback** | `llama-server` (CUDA) | `sam3` (CPU) | `google/embeddinggemma-300m` (CPU) | **~4 - 5 minutes** |

To configure engines, edit `configs/machine/llama_test.py`:

```python
engines = {
    "vlm": {
        "entry": "api",
        "protocol": "openai",
        "flavour": "llama.cpp",
        "model": "/usr/local/gguf/qwen3.5-9b/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf",
        "endpoints": [{"base_url": "http://127.0.0.1:8080/v1", "max_inflight": 4}],
    },
    "detect": {
        "entry": "torch",
        "devices": ["cuda:0"],
        "model": "google/owlv2-base-patch16-ensemble",
        "max_batch": 4,
    },
    "embed": {
        "entry": "torch",
        "devices": ["cuda:0"],
        "model": "google/embeddinggemma-300m",
        "max_batch": 64,
    },
    "cpu": {"entry": "python", "workers": 4},
}
```


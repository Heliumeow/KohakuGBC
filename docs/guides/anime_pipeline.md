# Anime Tagging Pipeline & Interactive Visualizer

This guide covers the two-stage anime image annotation pipeline in KohakuGBC, integrating
**CLTagger** (Anime ONNX classifier) and **Ultralytics YOLO Ensemble** (anime face/eye/body/accessory spatial priors) with **KohakuGBC** (VLM + OWLv2/SAM 3 scene graph generation) and the interactive HTML visualizer.

---

## 1. Overview & Architecture

Standard Vision-Language Models (VLMs) and generic open-vocabulary detectors often struggle with fine-grained anime franchise lore and stylized features (such as anime eyes, facial details, or specific accessories), frequently missing localized features or hallucinating identities.

To resolve this, KohakuGBC provides an automated two-stage pipeline:

```
[ Input Image ]
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[ Stage 1a: CLTagger (ONNX) ]         [ Stage 1b: Ultralytics YOLO Ensemble ]
  • Character, Copyright & Tags         • High-precision spatial anchor boxes
  • Confidence score ranking            • Eyes, mouth, breasts, jewelry, etc.
       │                                         │
       └────────────────────┬────────────────────┘
                            │
                            ▼
           [ Stage 2: KohakuGBC (VLM + Detector) ]
              • VLM receives visual tags as reference hints
              • Detail caption identifies specific character & franchise lore
              • Elements list retains standard COCO nouns for open detectors
              • Merges high-precision YOLO spatial priors into the graph
                            │
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

### 2.1 Prerequisites & Dependencies

In addition to standard KohakuGBC dependencies (`torch` extra for detectors and embedders), the anime pipeline requires `onnxruntime`, `ultralytics`, and optional `huggingface_hub`:

```bash
# Install with anime extra dependencies
pip install -e ".[anime,torch]"
```

### 2.2 One-Click Pipeline Execution

Run the complete pipeline on a single image (supports Windows paths `C:\...` or Linux paths `/mnt/c/...`):

```bash
# Single image run (CLTagger + YOLO priors + KohakuGBC + viewer.html)
.venv/bin/python scripts/cl_gbc_pipeline.py "C:\Users\YourUsername\Pictures\your_image.png"

# Batch run on an entire folder
.venv/bin/python scripts/cl_gbc_pipeline.py "/mnt/c/Users/YourUsername/Pictures"

# Ultra-fast prior tagging & bbox detection only (<1s, generates viewer.html without VLM/GBC)
.venv/bin/python scripts/cl_gbc_pipeline.py "C:\path\to\image.png" --no-gbc
```

### 2.3 Ultralytics YOLO Spatial Priors (`models/ultralytics/`)

Place any specialized YOLO models (e.g. `Eyes.pt`, `adetailer2dMouth_v10.pt`, `breasts_seg.pt`, `jewelryDetector_v021.pt`) into `models/ultralytics/`.

- **Default behavior**: The pipeline automatically discovers and runs **all** `.pt` and `.onnx` models in `models/ultralytics/` on every image.
- **Custom selection**: Pass `--yolo-models model1.pt model2.pt` to activate only specific models.
- **Disable YOLO**: Pass `--no-yolo` to skip YOLO spatial priors.
- **Tuning knobs**: Adjust confidence with `--yolo-conf 0.20` and resolution with `--yolo-imgsz 1280`.

### 2.4 Output Structure

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

### 2.5 CLTagger Model Resolution

`scripts/cl_gbc_pipeline.py` includes a standalone ONNX inference implementation (requiring only `onnxruntime`, `numpy`, and `Pillow`, with zero external project dependencies).

The script locates the model files (`model.onnx`, `model.onnx.data`, `model_vocabulary.json`) in the following order:
1. User-specified path via `--cltagger-dir /path/to/model_dir`
2. Environment variable `CLTAGGER_DIR`
3. Local workspace directory `models/cltagger/v2_01a`
4. Auto-download from HuggingFace Hub: [`cella110n/cl_tagger_v2`](https://huggingface.co/cella110n/cl_tagger_v2) (via `huggingface_hub`)

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
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
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


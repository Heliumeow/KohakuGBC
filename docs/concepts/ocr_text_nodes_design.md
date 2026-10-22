# Design Proposal: Dual-Channel OCR & Text Nodes Integration in KohakuGBC

**Author:** Antigravity & User  
**Status:** Proposal / Draft  
**Target Component:** `kohakugbc` (`gbc.kir`, `nodes`, `engines`)

---

## 1. Background & Motivation

In the current Graph-Based Captioning (GBC) architecture:
- The VLM generates a **Detailed Caption** that accurately recognizes in-image text (such as character names, UI stats, game indicators, and signboards).
- However, when constructing the **Scene Graph (VSG)**, text strings are deliberately excluded from graph vertices.
- The reason is structural: GBC delegates element localization to open-vocabulary object detectors (**OWLv2** or **SAM 3**). These detectors are trained on concrete physical objects (people, clothing, swords, vehicles) and perform poorly on fine-grained textual bounding boxes, causing text candidate nodes to fail confidence thresholds or crop verification.

For downstream tasks such as **game UI analysis, poster layout understanding, and typography-aware generation (Flux / SD3 / Anima)**, having text as first-class scene graph vertices with exact bounding boxes and semantic relations is highly valuable.

---

## 2. Proposed Architecture: Dual-Channel Grounding

Instead of forcing general object detectors to ground text, we propose a **Dual-Channel Grounding Pipeline**:

```
                       [ Input Image ]
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       [ Channel A: VLM ]           [ Channel B: OCR ]
   (Root Query + Prior Tags)     (PaddleOCR / DBNet / ONNX)
               │                             │
       ┌───────┴───────┐                     │
       ▼               ▼                     │
  [Captions]   [Physical Elements]           │
                       │                     │
                       ▼                     ▼
               [SAM 3 / OWLv2]        [Text BBoxes]
                       │              & Transcriptions
                       ▼                     │
             [Physical Vertices]      [Text Vertices]
                       │                     │
                       └──────────────┬──────┘
                                      ▼
                           [ Scene Graph Fusion ]
                       (Spatial & Semantic Linking)
                                      │
                                      ▼
                        [ Comprehensive GBC Graph ]
```

### 2.1 Channel A: Physical Entity Lane (Existing GBC)
- VLM extracts concrete physical objects (`warrior`, `sword`, `horns`, `beast`).
- OWLv2 or SAM 3 performs bounding-box detection and instance mask segmentation.
- Attributes (color, material, pose) and part compositions are queried as usual.

### 2.2 Channel B: Dedicated OCR Lane
- Add an `ocr` engine entry into Togeari (`entry("ocr", "onnx")` or `entry("ocr", "torch")`).
- Input: image patch.
- Output: list of text bounding boxes, detected transcriptions, and OCR confidence scores.
- Lightweight options:
  - ONNX runtime with mobile DBNet + CRNN (ultra-low latency, CPU/GPU friendly).
  - RapidOCR / PaddleOCR.

### 2.3 Graph Fusion & Semantic Linking
- **Vertex Types**:
  - `entity` (label="entity"): Physical visual objects.
  - `composition` (label="composition"): Groupings of entities.
  - `text` (label="text"): Detected text elements with properties:
    - `bbox`: Exact quadrilateral / rectangular bounding box.
    - `text`: Transcribed string content.
    - `descs`: Font color, style, language, orientation.
- **Relational Edges**:
  - Spatial containment: `[text|button]` ("labeled on button"), `[text|shirt]` ("printed on shirt").
  - Semantic association: `[text|character]` ("title or nameplate for character"), `[text|stat]` ("numerical level indicator").

---

## 3. Workflow Implementation in KIR

In `src/kohakugbc/program/gbc.kir`:

```kir
def gbc(image, tags, patch, controls):
    # Branch 1: Existing Physical Entity Pipeline
    (image, tags, patch, controls)image_query_request(request)
    (request)ask(answer)
    (image, answer)image_root(root)
    # ... detect, crop, attributes, relations ...

    # Branch 2: OCR Lane
    (patch)detect_text(raw_texts)
    (raw_texts, root)build_text_vertices(text_vertices)

    # Branch 3: Multi-Modal Graph Fusion
    (vertices, text_vertices)fuse_graph(unified_graph)
    (unified_graph)sink(result)
```

---

## 4. Key Advantages

1. **Zero Hallucination / Drift**: Specialized OCR models detect arbitrary characters, numbers, and multilingual typography with pixel accuracy.
2. **Detector Isolation**: General object detectors remain focused on standard COCO/LVIS visual nouns without text interference.
3. **Downstream Compatibility**: Downstream diffusion models or multimodal agents can query both "where is the character" and "where is the text banner" independently.


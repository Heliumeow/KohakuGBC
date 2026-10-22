#!/usr/bin/env python3
"""Interactive Scene Graph & BBox Visualizer for KohakuGBC.

Reads a KohakuGBC `graphs.jsonl` output and generates a standalone,
interactive HTML file with responsive SVG bounding boxes, hover tooltips,
detail captions, and node hierarchy exploration.

Usage:
    python scripts/visualize_graph.py [GRAPHS_JSONL_OR_RUN_DIR] [OPTIONS]
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import re
import subprocess
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KohakuGBC Scene Graph Visualizer</title>
<style>
  :root {
    --bg-main: #0b0f19;
    --bg-card: #111827;
    --bg-hover: #1f2937;
    --text-primary: #f3f4f6;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    --border-color: #374151;
    --accent-entity: #06b6d4;
    --accent-composition: #a855f7;
    --accent-relation: #f59e0b;
    --accent-highlight: #ec4899;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-main);
    color: var(--text-primary);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  header {
    background-color: var(--bg-card);
    border-bottom: 1px solid var(--border-color);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    z-index: 10;
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .header-title h1 {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #fff;
  }
  .path-badge {
    font-family: monospace;
    font-size: 0.8rem;
    background: #1e293b;
    color: #94a3b8;
    padding: 3px 8px;
    border-radius: 6px;
    max-width: 480px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .stats-bar {
    display: flex;
    gap: 10px;
    align-items: center;
  }
  .badge {
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .badge-entity { background: rgba(6, 182, 212, 0.15); color: var(--accent-entity); border: 1px solid rgba(6, 182, 212, 0.3); }
  .badge-composition { background: rgba(168, 85, 247, 0.15); color: var(--accent-composition); border: 1px solid rgba(168, 85, 247, 0.3); }
  .badge-relation { background: rgba(245, 158, 11, 0.15); color: var(--accent-relation); border: 1px solid rgba(245, 158, 11, 0.3); }

  main {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  /* Viewport */
  .viewport-container {
    flex: 1;
    position: relative;
    background: #030712;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .toolbar {
    position: absolute;
    top: 16px;
    left: 16px;
    z-index: 20;
    background: rgba(17, 24, 39, 0.85);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 8px 14px;
    display: flex;
    gap: 16px;
    align-items: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  }

  .tool-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    cursor: pointer;
  }
  .tool-item input[type="checkbox"] {
    accent-color: var(--accent-entity);
    cursor: pointer;
  }
  .slider-group {
    display: flex;
    align-items: center;
    gap: 8px;
    border-left: 1px solid var(--border-color);
    padding-left: 12px;
  }
  .slider-group input[type="range"] {
    width: 80px;
    accent-color: var(--accent-entity);
  }

  .stage {
    flex: 1;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    overflow: auto;
  }

  .image-wrapper {
    position: relative;
    box-shadow: 0 10px 40px rgba(0,0,0,0.7);
    border-radius: 6px;
    overflow: hidden;
    line-height: 0;
  }
  .image-wrapper img {
    display: block;
    max-width: 100%;
    max-height: calc(100vh - 120px);
    object-fit: contain;
    user-select: none;
  }

  svg.overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }

  .bbox {
    cursor: pointer;
    pointer-events: all;
    transition: all 0.15s ease-out;
  }
  .bbox.entity {
    stroke: var(--accent-entity);
    stroke-width: 2;
    fill: rgba(6, 182, 212, 0.05);
  }
  .bbox.composition {
    stroke: var(--accent-composition);
    stroke-width: 2.5;
    stroke-dasharray: 4 2;
    fill: rgba(168, 85, 247, 0.05);
  }
  .bbox.relation {
    stroke: var(--accent-relation);
    stroke-width: 2;
    stroke-dasharray: 6 3;
    fill: rgba(245, 158, 11, 0.05);
  }

  .bbox:hover, .bbox.active {
    stroke: var(--accent-highlight) !important;
    stroke-width: 3.5 !important;
    fill: rgba(236, 72, 153, 0.2) !important;
    stroke-dasharray: none !important;
    filter: drop-shadow(0 0 6px var(--accent-highlight));
  }

  /* Tooltip */
  .tooltip {
    position: absolute;
    display: none;
    z-index: 100;
    background: rgba(17, 24, 39, 0.95);
    backdrop-filter: blur(10px);
    border: 1px solid #4b5563;
    border-radius: 8px;
    padding: 12px 16px;
    max-width: 360px;
    pointer-events: none;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    font-size: 0.82rem;
    line-height: 1.4;
  }
  .tooltip-title {
    font-weight: 700;
    font-size: 0.92rem;
    color: #fff;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
  }
  .tooltip-desc {
    color: #d1d5db;
    margin-top: 6px;
    font-size: 0.8rem;
  }

  /* Sidebar */
  .sidebar {
    width: 440px;
    background: var(--bg-card);
    border-left: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .tabs {
    display: flex;
    border-bottom: 1px solid var(--border-color);
    background: #0f172a;
  }
  .tab-btn {
    flex: 1;
    padding: 12px 8px;
    text-align: center;
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    border-bottom: 2px solid transparent;
  }
  .tab-btn.active {
    color: #fff;
    border-bottom: 2px solid var(--accent-entity);
    background: rgba(6, 182, 212, 0.05);
  }

  .tab-content {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: none;
  }
  .tab-content.active {
    display: block;
  }

  /* Caption Card */
  .section-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    color: var(--text-muted);
    margin-bottom: 8px;
  }
  .caption-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 16px;
    font-size: 0.85rem;
    line-height: 1.5;
    color: #e2e8f0;
  }

  /* Node List */
  .search-input {
    width: 100%;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 12px;
    color: #fff;
    font-size: 0.82rem;
    margin-bottom: 12px;
    outline: none;
  }
  .search-input:focus {
    border-color: var(--accent-entity);
  }

  .node-item {
    background: #182234;
    border: 1px solid #283548;
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .node-item:hover, .node-item.selected {
    border-color: var(--accent-highlight);
    background: #1f2d44;
  }
  .node-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }
  .node-name {
    font-weight: 600;
    font-size: 0.85rem;
    color: #fff;
  }
  .node-desc {
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.35;
  }
</style>
</head>
<body>

<header>
  <div class="header-title">
    <h1>KohakuGBC Scene Graph</h1>
    <span class="path-badge" title="__IMG_PATH__">__IMAGE_BASENAME__</span>
  </div>
  <div class="stats-bar">
    <span class="badge badge-entity">Entities: __COUNT_ENTITIES__</span>
    <span class="badge badge-composition">Compositions: __COUNT_COMPOSITIONS__</span>
    <span class="badge badge-relation">Relations: __COUNT_RELATIONS__</span>
  </div>
</header>

<main>
  <div class="viewport-container">
    <div class="toolbar">
      <label class="tool-item">
        <input type="checkbox" id="toggle-entity" checked>
        <span style="color:var(--accent-entity)">●</span> Entities
      </label>
      <label class="tool-item">
        <input type="checkbox" id="toggle-composition" checked>
        <span style="color:var(--accent-composition)">●</span> Compositions
      </label>
      <label class="tool-item">
        <input type="checkbox" id="toggle-relation" checked>
        <span style="color:var(--accent-relation)">●</span> Relations
      </label>
      <div class="slider-group">
        <span style="font-size:0.75rem; color:var(--text-muted)">Min Conf:</span>
        <input type="range" id="conf-slider" min="0" max="1" step="0.05" value="0.0">
        <span id="conf-val" style="font-size:0.75rem; font-family:monospace">0.00</span>
      </div>
    </div>

    <div class="stage">
      <div class="image-wrapper" id="img-wrapper">
        <img src="__IMG_B64__" id="target-img" alt="Target Image">
        <svg class="overlay" id="svg-overlay" viewBox="0 0 __ORIG_W__ __ORIG_H__" preserveAspectRatio="none">
        </svg>
      </div>
    </div>

    <div class="tooltip" id="tooltip"></div>
  </div>

  <div class="sidebar">
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('captions')">Captions</button>
      <button class="tab-btn" onclick="switchTab('nodes')">Nodes (__COUNT_NODES__)</button>
      <button class="tab-btn" onclick="switchTab('relations')">Relations (__COUNT_RELATIONS__)</button>
    </div>

    <div id="tab-captions" class="tab-content active">
      <div class="section-title">Detail Caption</div>
      <div class="caption-box">__DETAIL_CAPTION__</div>

      <div class="section-title">Short Caption</div>
      <div class="caption-box">__SHORT_CAPTION__</div>

      <div class="section-title">Image Metadata</div>
      <div class="caption-box" style="font-family:monospace; font-size:0.75rem;">
        Resolution: __ORIG_W__ × __ORIG_H__ px<br>
        File: __IMG_PATH__
      </div>
    </div>

    <div id="tab-nodes" class="tab-content">
      <input type="text" class="search-input" id="node-search" placeholder="Filter nodes by name or description...">
      <div id="nodes-container"></div>
    </div>

    <div id="tab-relations" class="tab-content">
      <div id="relations-container"></div>
    </div>
  </div>
</main>

<script id="graph-data" type="application/json">
__GRAPH_JSON__
</script>

<script>
  const graph = JSON.parse(document.getElementById('graph-data').textContent);
  const svgOverlay = document.getElementById('svg-overlay');
  const tooltip = document.getElementById('tooltip');
  const imgWrapper = document.getElementById('img-wrapper');
  const targetImg = document.getElementById('target-img');

  const origW = __ORIG_W__;
  const origH = __ORIG_H__;

  let activeVertexId = null;
  const vertices = (graph.vertices || []).filter(v => v.label !== 'image');

  // Render SVG Bounding Boxes
  function renderSvgBoxes() {
    svgOverlay.innerHTML = '';
    const showEntity = document.getElementById('toggle-entity').checked;
    const showComp = document.getElementById('toggle-composition').checked;
    const showRel = document.getElementById('toggle-relation').checked;
    const minConf = parseFloat(document.getElementById('conf-slider').value);

    vertices.forEach(v => {
      const type = v.label;
      if (type === 'entity' && !showEntity) return;
      if (type === 'composition' && !showComp) return;
      if (type === 'relation' && !showRel) return;

      const conf = v.bbox && v.bbox.confidence != null ? v.bbox.confidence : 1.0;
      if (conf < minConf && type === 'entity') return;

      const bbox = v.bbox;
      if (!bbox) return;

      const x = bbox.left * origW;
      const y = bbox.top * origH;
      const w = (bbox.right - bbox.left) * origW;
      const h = (bbox.bottom - bbox.top) * origH;

      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', x);
      rect.setAttribute('y', y);
      rect.setAttribute('width', Math.max(w, 4));
      rect.setAttribute('height', Math.max(h, 4));
      rect.setAttribute('class', `bbox ${type} ${v.vertex_id === activeVertexId ? 'active' : ''}`);
      rect.setAttribute('data-id', v.vertex_id);

      rect.addEventListener('mouseenter', (e) => showTooltip(e, v));
      rect.addEventListener('mousemove', (e) => updateTooltipPos(e));
      rect.addEventListener('mouseleave', hideTooltip);
      rect.addEventListener('click', () => selectNode(v.vertex_id));

      svgOverlay.appendChild(rect);
    });
  }

  function showTooltip(e, v) {
    const desc = (v.descs && v.descs[0] && v.descs[0].text) ? v.descs[0].text : 'No description';
    const conf = v.bbox && v.bbox.confidence != null ? v.bbox.confidence.toFixed(2) : 'N/A';

    tooltip.innerHTML = `
      <div class="tooltip-title">
        <span>${v.vertex_id}</span>
        <span style="font-size:0.75rem; color:var(--accent-${v.label})">${v.label} (${conf})</span>
      </div>
      <div class="tooltip-desc">${desc}</div>
    `;
    tooltip.style.display = 'block';
    updateTooltipPos(e);
  }

  function updateTooltipPos(e) {
    const margin = 15;
    let left = e.clientX + margin;
    let top = e.clientY + margin;
    if (left + 360 > window.innerWidth) left = e.clientX - 370;
    if (top + 160 > window.innerHeight) top = e.clientY - 170;
    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
  }

  function hideTooltip() {
    tooltip.style.display = 'none';
  }

  function selectNode(vertexId) {
    activeVertexId = vertexId;
    document.querySelectorAll('.bbox').forEach(r => {
      r.classList.toggle('active', r.getAttribute('data-id') === vertexId);
    });
    document.querySelectorAll('.node-item').forEach(n => {
      n.classList.toggle('selected', n.getAttribute('data-id') === vertexId);
      if (n.getAttribute('data-id') === vertexId) {
        n.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
  }

  // Populate Nodes List
  function populateNodesList(filterText = '') {
    const container = document.getElementById('nodes-container');
    container.innerHTML = '';
    const q = filterText.toLowerCase();

    vertices.forEach(v => {
      const desc = (v.descs && v.descs[0] && v.descs[0].text) ? v.descs[0].text : '';
      if (q && !v.vertex_id.toLowerCase().includes(q) && !desc.toLowerCase().includes(q)) return;

      const conf = v.bbox && v.bbox.confidence != null ? `conf: ${v.bbox.confidence.toFixed(2)}` : '';
      const div = document.createElement('div');
      div.className = `node-item ${v.vertex_id === activeVertexId ? 'selected' : ''}`;
      div.setAttribute('data-id', v.vertex_id);

      div.innerHTML = `
        <div class="node-header">
          <span class="node-name" style="color:var(--accent-${v.label})">${v.vertex_id}</span>
          <span style="font-size:0.75rem; color:var(--text-muted)">${v.label} ${conf}</span>
        </div>
        <div class="node-desc">${desc}</div>
      `;

      div.addEventListener('mouseenter', () => selectNode(v.vertex_id));
      div.addEventListener('click', () => selectNode(v.vertex_id));
      container.appendChild(div);
    });
  }

  // Populate Relations List
  function populateRelations() {
    const container = document.getElementById('relations-container');
    container.innerHTML = '';
    const relations = vertices.filter(v => v.label === 'relation');

    if (relations.length === 0) {
      container.innerHTML = '<div style="color:var(--text-muted); font-size:0.85rem;">No relations found in this graph.</div>';
      return;
    }

    relations.forEach(r => {
      const desc = (r.descs && r.descs[0] && r.descs[0].text) ? r.descs[0].text : '';
      const div = document.createElement('div');
      div.className = `node-item ${r.vertex_id === activeVertexId ? 'selected' : ''}`;
      div.setAttribute('data-id', r.vertex_id);

      div.innerHTML = `
        <div class="node-header">
          <span class="node-name" style="color:var(--accent-relation)">${r.vertex_id}</span>
        </div>
        <div class="node-desc">${desc}</div>
      `;
      div.addEventListener('mouseenter', () => selectNode(r.vertex_id));
      container.appendChild(div);
    });
  }

  function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach((b, i) => {
      b.classList.toggle('active', b.getAttribute('onclick').includes(tab));
    });
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('tab-' + tab).classList.add('active');
  }

  // Event Listeners
  document.getElementById('toggle-entity').addEventListener('change', renderSvgBoxes);
  document.getElementById('toggle-composition').addEventListener('change', renderSvgBoxes);
  document.getElementById('toggle-relation').addEventListener('change', renderSvgBoxes);
  document.getElementById('conf-slider').addEventListener('input', (e) => {
    document.getElementById('conf-val').textContent = parseFloat(e.target.value).toFixed(2);
    renderSvgBoxes();
  });
  document.getElementById('node-search').addEventListener('input', (e) => {
    populateNodesList(e.target.value);
  });

  // Initial render
  renderSvgBoxes();
  populateNodesList();
  populateRelations();
</script>
</body>
</html>
"""


def normalize_path(path_str: str) -> Path:
    """Normalize Windows or Linux path to local Path."""
    p = path_str.strip().strip('"').strip("'")
    match = re.match(r"^([a-zA-Z]):[/\\](.*)", p)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(p).resolve()


def resolve_graphs_file(target: str | None) -> tuple[Path, Path]:
    """Find the graphs.jsonl file and corresponding directory."""
    if not target:
        run_dir = WORKSPACE_ROOT / "runs" / "latest"
        graphs_file = run_dir / "graphs.jsonl"
    else:
        p = normalize_path(target)
        if p.is_file():
            graphs_file = p
            run_dir = p.parent
        elif p.is_dir():
            graphs_file = p / "graphs.jsonl"
            run_dir = p
        else:
            raise FileNotFoundError(f"Target path not found: {target}")

    if not graphs_file.exists():
        raise FileNotFoundError(f"graphs.jsonl not found at: {graphs_file}")

    return graphs_file, run_dir


def load_image_base64(img_path_str: str, max_dimension: int = 1920) -> tuple[str, int, int]:
    """Load image, optionally resize to save HTML weight, and return base64 data URL and dimensions."""
    p = normalize_path(img_path_str)
    if not p.exists():
        print(f"[Warning] Image file not found: {p}", file=sys.stderr)
        return "", 1920, 1080

    with Image.open(p) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        # Resize if overly large
        if max(orig_w, orig_h) > max_dimension:
            scale = max_dimension / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88, optimize=True)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}", orig_w, orig_h


def generate_html(graph_data: dict, img_b64: str, orig_w: int, orig_h: int) -> str:
    """Generate self-contained interactive viewer HTML using template replacement."""
    detail_caption = graph_data.get("detail_caption", "")
    short_caption = graph_data.get("short_caption", "")
    img_path = graph_data.get("img_path", "")
    vertices = graph_data.get("vertices", [])

    entities = [v for v in vertices if v.get("label") == "entity"]
    compositions = [v for v in vertices if v.get("label") == "composition"]
    relations = [v for v in vertices if v.get("label") == "relation"]

    graph_json_str = json.dumps(graph_data, ensure_ascii=False)

    replacements = {
        "__IMG_PATH__": html.escape(str(img_path)),
        "__IMAGE_BASENAME__": html.escape(Path(img_path).name),
        "__COUNT_ENTITIES__": str(len(entities)),
        "__COUNT_COMPOSITIONS__": str(len(compositions)),
        "__COUNT_RELATIONS__": str(len(relations)),
        "__COUNT_NODES__": str(max(0, len(vertices) - 1)),
        "__IMG_B64__": img_b64,
        "__ORIG_W__": str(orig_w),
        "__ORIG_H__": str(orig_h),
        "__DETAIL_CAPTION__": html.escape(detail_caption),
        "__SHORT_CAPTION__": html.escape(short_caption),
        "__GRAPH_JSON__": graph_json_str,
    }

    res = HTML_TEMPLATE
    for placeholder, val in replacements.items():
        res = res.replace(placeholder, val)
    return res


def open_in_browser(file_path: Path) -> None:
    """Open file in default browser (supports Windows host through WSL)."""
    if Path("/proc/version").exists() and "microsoft" in Path("/proc/version").read_text().lower():
        try:
            res = subprocess.run(["wslpath", "-w", str(file_path)], capture_output=True, text=True, check=True)
            win_path = res.stdout.strip()
            subprocess.run(["cmd.exe", "/c", "start", "", win_path], check=False)
            print(f"[+] Opened in Windows browser: {win_path}")
            return
        except Exception:
            pass
    webbrowser.open(f"file://{file_path.resolve()}")


def serve_dir(directory: Path, port: int = 8899) -> None:
    """Serve directory via simple HTTP server."""
    os.chdir(directory)
    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer(("0.0.0.0", port), handler)
    print(f"\n[+] Local Visualizer Server running at: http://localhost:{port}/viewer.html")
    print("    Press Ctrl+C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="KohakuGBC Interactive Scene Graph Visualizer")
    parser.add_argument("target", nargs="?", default=None, help="Path to graphs.jsonl or run directory (default: runs/latest)")
    parser.add_argument("--output", default=None, help="Output HTML file path (default: <run_dir>/viewer.html)")
    parser.add_argument("--open", action="store_true", help="Automatically open generated HTML in browser")
    parser.add_argument("--serve", action="store_true", help="Launch local HTTP server to view in browser")
    parser.add_argument("--port", type=int, default=8899, help="Port for HTTP server (default: 8899)")
    args = parser.parse_args()

    graphs_file, run_dir = resolve_graphs_file(args.target)
    print(f"[*] Reading graph from: {graphs_file}")

    with open(graphs_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            print("[Error] graphs.jsonl is empty.", file=sys.stderr)
            sys.exit(1)
        graph_data = json.loads(first_line)

    img_path = graph_data.get("img_path", "")
    print(f"[*] Encoding image: {img_path}...")
    img_b64, orig_w, orig_h = load_image_base64(img_path)

    print(f"[*] Generating interactive viewer HTML...")
    html_content = generate_html(graph_data, img_b64, orig_w, orig_h)

    output_path = Path(args.output).resolve() if args.output else run_dir / "viewer.html"
    output_path.write_text(html_content, encoding="utf-8")
    print(f"[+] Successfully generated: {output_path}")

    latest_viewer = WORKSPACE_ROOT / "runs" / "latest" / "viewer.html"
    if latest_viewer.parent.exists() and output_path != latest_viewer:
        try:
            latest_viewer.write_text(html_content, encoding="utf-8")
        except Exception:
            pass

    if args.open:
        open_in_browser(output_path)

    if args.serve:
        serve_dir(run_dir, port=args.port)


if __name__ == "__main__":
    main()


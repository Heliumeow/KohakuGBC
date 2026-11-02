from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import os
import re
import sys
import time
from pathlib import Path


def parse_vlm_json(content: str) -> dict:
    """Robustly parse JSON object from VLM string response."""
    if not content:
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def extract_caption_from_reasoning(reasoning: str, cls_name: str) -> tuple[str, str]:
    """Fallback extraction of visual features from VLM reasoning tokens when JSON format was interrupted."""
    if not reasoning:
        return f"Anime character {cls_name}.", f"anime {cls_name}"
    clean = re.sub(r"[\*\#\`]", "", reasoning)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if len(s.strip()) > 20]
    good_s = [
        s for s in sentences
        if not s.lower().startswith("the user")
        and not s.lower().startswith("let's")
        and not s.lower().startswith("we need")
    ]
    if good_s:
        detailed = " ".join(good_s[-2:])
    elif sentences:
        detailed = sentences[-1]
    else:
        detailed = f"Anime character {cls_name} with distinct styling."
    short = f"anime {cls_name}"
    return detailed[:300], short


def check_vlm_reachability(endpoint: str = "http://127.0.0.1:8080/v1") -> bool:
    """Check if the VLM server endpoint is responding."""
    import httpx

    url = endpoint.rstrip("/")
    for test_path in ["/models", "", "/health"]:
        try:
            r = httpx.get(f"{url}{test_path}", timeout=1.5)
            if r.status_code < 500:
                return True
        except Exception:
            pass
    return False


def describe_yolo_detections(
    image_path: Path,
    detections: list[dict],
    endpoint: str = "http://127.0.0.1:8080/v1",
    model_name: str | None = None,
    max_workers: int = 4,
    char_context: str | None = None,
) -> list[dict]:
    """Use the active VLM server to generate detailed and short captions for YOLO-detected bounding boxes."""
    if not detections:
        return detections

    try:
        from PIL import Image
    except ImportError:
        return detections

    import httpx

    if not check_vlm_reachability(endpoint):
        return detections

    model = model_name or os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[Warning] Could not load image for VLM crop description: {e}", file=sys.stderr)
        return detections

    w, h = img.size

    def _describe_one(det: dict) -> dict:
        cls_name = det.get("class", "object")
        box = det.get("bbox", {})
        l, t, r, b = box.get("left", 0.0), box.get("top", 0.0), box.get("right", 1.0), box.get("bottom", 1.0)
        bw, bh = max(0.001, r - l), max(0.001, b - t)

        # 15% contextual padding
        pad_x, pad_y = bw * 0.15, bh * 0.15
        x1 = max(0, int((l - pad_x) * w))
        y1 = max(0, int((t - pad_y) * h))
        x2 = min(w, int((r + pad_x) * w))
        y2 = min(h, int((b + pad_y) * h))

        crop = img.crop((x1, y1, x2, y2))
        if crop.width < 224 or crop.height < 224:
            scale = max(224 / max(1, crop.width), 224 / max(1, crop.height))
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt_focus = f"{cls_name}"
        if char_context:
            prompt_focus += f" of character '{char_context}'"

        system_msg = (
            "You are an expert anime character and visual art analyzer. "
            "When given a cropped image patch, describe the specific visual feature in concise, precise detail. "
            "Output JSON only."
        )
        user_msg = (
            f"Focus on the {prompt_focus} shown in this cropped image patch.\n"
            "Describe its detailed visual appearance (color, shape, styling, texture, highlights, expression or decorations).\n"
            "Return a valid JSON object with exactly two keys:\n"
            "- \"detailed_caption\": 1-2 detailed sentences describing the visual features.\n"
            "- \"short_caption\": A concise phrase (3-8 words) summarizing the appearance."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_msg},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 1800,
            "temperature": 0.2,
        }

        det_copy = dict(det)
        try:
            resp = httpx.post(f"{endpoint.rstrip('/')}/chat/completions", json=payload, timeout=90.0)
            if resp.status_code == 200:
                c = resp.json()["choices"][0]["message"]
                raw_content = c.get("content", "")
                parsed = parse_vlm_json(raw_content)

                detailed = parsed.get("detailed_caption")
                short = parsed.get("short_caption")

                # Fallback to reasoning extraction if JSON was incomplete
                if not detailed and c.get("reasoning_content"):
                    detailed, short = extract_caption_from_reasoning(c["reasoning_content"], cls_name)

                if not detailed:
                    detailed = f"Anime character {cls_name} detected with confidence {det.get('conf', 0):.2f}."
                if not short:
                    short = f"anime {cls_name}"

                det_copy["descs"] = [
                    {
                        "label": "detail",
                        "text": detailed,
                        "type": "vlm_yolo_desc",
                    },
                    {
                        "label": "short",
                        "text": short,
                        "type": "vlm_yolo_desc",
                    },
                    {
                        "label": "source",
                        "text": f"Character {cls_name} detected via {det.get('model', 'yolo')} (confidence: {det.get('conf', 0):.2f}).",
                        "type": "yolo_prior",
                    },
                ]
            else:
                det_copy["descs"] = [
                    {
                        "label": "detail",
                        "text": f"Character {cls_name} detected via {det.get('model')} (confidence: {det.get('conf', 0):.2f}).",
                        "type": "yolo_prior",
                    }
                ]
        except Exception as e:
            det_copy["descs"] = [
                {
                    "label": "detail",
                    "text": f"Character {cls_name} detected via {det.get('model')} (confidence: {det.get('conf', 0):.2f}).",
                    "type": "yolo_prior",
                }
            ]
        return det_copy

    print(f"[*] Generating VLM visual descriptions for {len(detections)} YOLO detection(s) on {image_path.name}...")
    start_t = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(detections))) as executor:
        results = list(executor.map(_describe_one, detections))
    print(f"[+] Described {len(results)} YOLO detection(s) in {time.time() - start_t:.1f}s")
    return results

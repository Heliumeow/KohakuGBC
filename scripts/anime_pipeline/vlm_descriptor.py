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


CLS_ZH_MAP: dict[str, str] = {
    "eyes": "眼睛/瞳孔",
    "eye": "眼睛",
    "mouth": "嘴唇/表情",
    "face": "面部五官",
    "hair": "发型/发丝",
    "breasts": "胸部服饰",
    "breast": "胸部服饰",
    "jewelry": "首饰饰品",
    "gem_with_border_jewelry": "镶边宝石饰品",
    "loop_jewelry": "环状耳饰或颈圈",
    "bracelet": "手镯/腕饰",
    "long_jewelry": "长款吊坠挂饰",
    "head_jewelry": "头饰/发饰",
    "collar": "衣领/项圈",
    "choker": "颈饰/项圈",
}


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


def extract_caption_from_reasoning(reasoning: str, cls_name: str, language: str = "zh") -> tuple[str, str]:
    """Fallback extraction of visual features from VLM reasoning tokens when JSON format was interrupted."""
    cls_zh = CLS_ZH_MAP.get(cls_name.lower(), cls_name)
    if not reasoning:
        if language == "zh":
            return f"二次元角色的{cls_zh}细节。", f"{cls_zh}细节"
        return f"Anime character {cls_name}.", f"anime {cls_name}"

    # Try explicit JSON key extraction from reasoning text first
    m_det = re.search(r'"detailed_caption"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', reasoning)
    m_short = re.search(r'"short_caption"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', reasoning)
    if m_det:
        det_val = m_det.group(1).encode().decode("unicode_escape", errors="ignore").strip()
        short_val = (
            m_short.group(1).encode().decode("unicode_escape", errors="ignore").strip()
            if m_short
            else (f"{cls_zh}细节" if language == "zh" else f"anime {cls_name}")
        )
        if len(det_val) > 8:
            return det_val[:300], short_val[:60]

    clean = re.sub(r"[\*\#\`]", "", reasoning)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？\n])\s+", clean) if len(s.strip()) > 15]
    good_s = [
        s for s in sentences
        if not s.lower().startswith("the user")
        and not s.lower().startswith("let's")
        and not s.lower().startswith("we need")
        and not s.startswith("用户")
        and not s.startswith("我们需要")
        and not s.startswith("分析")
        and "json" not in s.lower()
        and "caption" not in s.lower()
        and "formatting" not in s.lower()
    ]
    if good_s:
        detailed = " ".join(good_s[-2:])
    elif sentences:
        detailed = sentences[-1]
    else:
        if language == "zh":
            detailed = f"二次元角色【{cls_zh}】具有鲜明的视觉表现。"
        else:
            detailed = f"Anime character {cls_name} with distinct styling."

    if language == "zh":
        short = f"{cls_zh}细节"
    else:
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
    language: str = "zh",
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
        cls_zh = CLS_ZH_MAP.get(cls_name.lower(), cls_name)
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

        if language == "zh":
            prompt_focus = f"角色【{cls_zh}】部位"
            if char_context:
                prompt_focus = f"角色【{char_context}】的【{cls_zh}】部位"

            system_msg = (
                "你是一位精通二次元动漫插画与角色美术特征的资深视觉分析专家。"
                "针对提供的局部裁剪图像，请以精准、优美、地道的动漫术语描述该细节特征。"
                "请直接输出合法的 JSON 格式结果，严禁输出多余的思考内容。"
            )
            user_msg = (
                f"请仔细观察裁剪图像中展示的{prompt_focus}区域。\n"
                "详细描述其视觉外观特征（包括色彩明暗、反光高光、线条唇形/瞳孔细节、神态表情或材质纹理）。\n"
                "直接返回一个合法的 JSON 对象，且仅包含两个键：\n"
                "- \"detailed_caption\": 1-2 句细腻生动、符合二次元美术风格的中文视觉特征描写。\n"
                "- \"short_caption\": 4-8 个字的精炼中文短语。"
            )
        else:
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
            "max_tokens": 2500,
            "temperature": 0.1,
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
                    detailed, short = extract_caption_from_reasoning(
                        c["reasoning_content"], cls_name, language=language
                    )

                if not detailed:
                    if language == "zh":
                        detailed = f"二次元角色【{cls_zh}】细节，线条与色彩表现自然。"
                        short = f"{cls_zh}细节"
                    else:
                        detailed = f"Anime character {cls_name} detected with confidence {det.get('conf', 0):.2f}."
                        short = f"anime {cls_name}"

                source_text = (
                    f"角色【{cls_zh}】通过模型 {det.get('model', 'yolo')} 检测识别（置信度: {det.get('conf', 0):.2f}）。"
                    if language == "zh"
                    else f"Character {cls_name} detected via {det.get('model', 'yolo')} (confidence: {det.get('conf', 0):.2f})."
                )

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
                        "text": source_text,
                        "type": "yolo_prior",
                    },
                ]
            else:
                fallback_text = (
                    f"角色【{cls_zh}】通过模型 {det.get('model')} 检测识别（置信度: {det.get('conf', 0):.2f}）。"
                    if language == "zh"
                    else f"Character {cls_name} detected via {det.get('model')} (confidence: {det.get('conf', 0):.2f})."
                )
                det_copy["descs"] = [
                    {
                        "label": "detail",
                        "text": fallback_text,
                        "type": "yolo_prior",
                    }
                ]
        except Exception as e:
            fallback_text = (
                f"角色【{cls_zh}】通过模型 {det.get('model')} 检测识别（置信度: {det.get('conf', 0):.2f}）。"
                if language == "zh"
                else f"Character {cls_name} detected via {det.get('model')} (confidence: {det.get('conf', 0):.2f})."
            )
            det_copy["descs"] = [
                {
                    "label": "detail",
                    "text": fallback_text,
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


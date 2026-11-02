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
    "person": "角色全身/人物主体",
    "head": "角色头部/发型",
    "face": "面部五官",
    "eyes": "眼睛/瞳孔",
    "eye": "眼睛",
    "mouth": "嘴唇/表情",
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
    "hand": "手部姿态",
    "hands": "手部姿态",
}


def clean_vlm_text(text: str) -> str:
    """Clean leaked reasoning or prompt evaluation fragments from caption text."""
    if not text:
        return ""
    t = text.strip()
    # If the model wrapped the actual caption in quotes after self-evaluation thoughts
    m_quote = re.search(r'["“]([^"”]{10,})["”]', t)
    if m_quote and any(eval_word in t for eval_word in ["字范围", "符合", "Good", "fits", "Check", "稍微"]):
        t = m_quote.group(1)
    t = re.sub(r"^.*?->\s*", "", t)
    t = re.sub(r"^\s*\(.*?\)\s*", "", t)
    t = re.sub(r"^Check\s+.*?[.:]\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^Good\.\s*\"?.*?[.:\"]?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^Fits well.*?[.:]\s*", "", t, flags=re.IGNORECASE)
    t = t.strip("\"'“” ")
    return t.strip()


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
        try:
            det_val = json.loads(f'"{m_det.group(1)}"').strip()
        except Exception:
            det_val = m_det.group(1).strip()

        if m_short:
            try:
                short_val = json.loads(f'"{m_short.group(1)}"').strip()
            except Exception:
                short_val = m_short.group(1).strip()
        else:
            short_val = f"{cls_zh}细节" if language == "zh" else f"anime {cls_name}"

        if len(det_val) > 8:
            return det_val[:300], short_val[:60]

    clean = re.sub(r"[\*\#\`]", "", reasoning)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？\n])\s+", clean) if len(s.strip()) > 15]
    good_s = [
        s for s in sentences
        if not s.lower().startswith("the user")
        and not s.lower().startswith("let's")
        and not s.lower().startswith("we need")
        and not s.lower().startswith("good.")
        and not s.lower().startswith("check")
        and not s.startswith("用户")
        and not s.startswith("我们需要")
        and not s.startswith("分析")
        and "json" not in s.lower()
        and "caption" not in s.lower()
        and "formatting" not in s.lower()
        and "fits well" not in s.lower()
        and (language != "zh" or bool(re.search(r"[\u4e00-\u9fff]", s)))
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


def _fallback_global_desc(tag_item: dict | None, language: str = "zh") -> dict[str, str]:
    """Generate high-quality rule-based fallback global captions from tags."""
    char_str = "二次元动漫角色" if language == "zh" else "Anime character"
    copy_str = ""
    top_tags = ""
    if tag_item:
        tags_info = tag_item.get("tags", {})
        char_dict = tags_info.get("character", {})
        copy_dict = tags_info.get("copyright", {})
        gen_dict = tags_info.get("general", {})
        sorted_chars = [t.replace("_", " ") for t in sorted(char_dict.keys(), key=lambda k: -char_dict[k])]
        sorted_copys = [t.replace("_", " ") for t in sorted(copy_dict.keys(), key=lambda k: -copy_dict[k])]
        sorted_gens = [t.replace("_", " ") for t in sorted(gen_dict.keys(), key=lambda k: -gen_dict[k])[:15]]
        if sorted_chars:
            char_str = ", ".join(sorted_chars[:2])
        if sorted_copys:
            copy_str = f"（出自《{', '.join(sorted_copys[:2])}》）" if language == "zh" else f" from {', '.join(sorted_copys[:2])}"
        top_tags = ("、" if language == "zh" else ", ").join(sorted_gens)

    if language == "zh":
        detailed = f"{char_str}{copy_str}的二次元动漫插画。画面主要特征包括：{top_tags or '角色主体与背景立绘'}。"
        short = f"{char_str}{copy_str}数码立绘插画"
    else:
        detailed = f"An anime digital illustration of {char_str}{copy_str}. Visual features include: {top_tags or 'character and background'}."
        short = f"Illustration of {char_str}"
    return {"detailed_caption": detailed, "short_caption": short}


def describe_full_image(
    image_path: Path,
    tag_item: dict | None = None,
    endpoint: str = "http://127.0.0.1:8080/v1",
    model_name: str | None = None,
    language: str = "zh",
) -> dict[str, str]:
    """Generate comprehensive global visual descriptions (detailed and short captions) for the entire image.

    Args:
        image_path: Path to the image file.
        tag_item: Optional CLTagger preliminary tags (character, copyright, general).
        endpoint: VLM server base URL.
        model_name: Optional explicit VLM model name.
        language: 'zh' (Chinese) or 'en' (English).

    Returns:
        dict: {"detailed_caption": str, "short_caption": str}
    """
    if not check_vlm_reachability(endpoint):
        return _fallback_global_desc(tag_item, language)

    try:
        from PIL import Image
        import httpx

        with Image.open(image_path) as im:
            im = im.convert("RGB")
            max_dim = max(im.size)
            if max_dim > 1920:
                scale = 1920 / max_dim
                im = im.resize((int(im.width * scale), int(im.height * scale)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=92)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[Warning] Failed to read/encode full image for VLM: {e}", file=sys.stderr)
        return _fallback_global_desc(tag_item, language)

    model = model_name or os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    tags_hint = ""
    char_name = None
    if tag_item:
        tags_info = tag_item.get("tags", {})
        char_dict = tags_info.get("character", {})
        copy_dict = tags_info.get("copyright", {})
        gen_dict = tags_info.get("general", {})
        chars = [t.replace("_", " ") for t in sorted(char_dict.keys(), key=lambda k: -char_dict[k])[:3]]
        copys = [t.replace("_", " ") for t in sorted(copy_dict.keys(), key=lambda k: -copy_dict[k])[:2]]
        gens = [t.replace("_", " ") for t in sorted(gen_dict.keys(), key=lambda k: -gen_dict[k])[:15]]

        parts = []
        if chars:
            char_name = chars[0]
            parts.append(f"识别角色：{', '.join(chars)}")
        if copys:
            parts.append(f"所属作品：{', '.join(copys)}")
        if gens:
            parts.append(f"画面特征标签：{', '.join(gens)}")
        if parts:
            tags_hint = "参考视觉先验（仅供识别角色与元素参考）：\n" + "；".join(parts) + "\n\n"

    if language == "zh":
        system_msg = (
            "你是一位精通二次元动漫插画、角色美术设计与场景空间构图的资深视觉分析专家。"
            "请遵循系统性的视觉分析流程（主体角色造型/服饰/神态 -> 场景建筑与天体环境 -> 整体色彩与光影氛围）进行专业拆解。"
            "请直接输出合法的 JSON 格式结果，严禁输出多余的思考内容。"
        )
        user_msg = (
            f"请对整幅二次元动漫插画进行系统、专业、画面感强的视觉分析：\n"
            f"{tags_hint}"
            "【画面分析维度与规则】：\n"
            "1. 详细长描述 (\"detailed_caption\")：\n"
            "   - 核心角色主体：准确辨析人物身份、发型发色、瞳色眼妆、神情面貌，服饰款式（剪裁、层叠与材质纹理）及肢体动作手势。\n"
            "   - 场景与环境空间：建筑透视、自然地貌、天体天象（满月/星空/云雾）与空间纵深感。\n"
            "   - 色彩光影与特效：主色调、冷暖对比光斑、动态粒子或光束特效、整体视觉张力与艺术风格。\n"
            "   - 篇幅：2-3 句语言流畅凝练、色彩层次分明的中文综合描写（约100-150字）。\n"
            "2. 精炼概括短句 (\"short_caption\")：\n"
            "   - 1 句精炼紧凑概括“主体角色在何种核心场景下的关键姿态与神态”（15-25字）。\n\n"
            "请直接返回合法的 JSON 对象，包含两个键：\n"
            "- \"detailed_caption\": 2-3 句中文画面综合描述。\n"
            "- \"short_caption\": 1 句精炼中文概括短句。"
        )
    else:
        system_msg = (
            "You are an expert anime character and visual art analyzer. "
            "Conduct a structured visual analysis covering character design, setting, and lighting atmosphere. "
            "Output valid JSON only."
        )
        user_msg = (
            f"Please provide a comprehensive visual analysis of the full anime illustration.\n"
            f"{tags_hint}"
            "Dimensions to cover:\n"
            "- Character: hairstyle, eyes, facial expression, attire styling and pose.\n"
            "- Environment & Lighting: architectural structures, celestial background, color harmony, and atmospheric effects.\n"
            "Return a valid JSON object with exactly two keys:\n"
            "- \"detailed_caption\": 2-3 detailed, descriptive sentences (100-150 words).\n"
            "- \"short_caption\": 1 concise sentence summarizing the main subject and setting (15-25 words)."
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_msg},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 2500,
        "temperature": 0.1,
    }

    try:
        resp = httpx.post(f"{endpoint.rstrip('/')}/chat/completions", json=payload, timeout=90.0)
        if resp.status_code == 200:
            c = resp.json()["choices"][0]["message"]
            raw_content = c.get("content", "")
            parsed = parse_vlm_json(raw_content)
            detailed = clean_vlm_text(parsed.get("detailed_caption", ""))
            short = clean_vlm_text(parsed.get("short_caption", ""))
            if not detailed and c.get("reasoning_content"):
                detailed, short = extract_caption_from_reasoning(
                    c["reasoning_content"], char_name or "character", language=language
                )
                detailed = clean_vlm_text(detailed)
                short = clean_vlm_text(short)
            if detailed:
                return {
                    "detailed_caption": detailed,
                    "short_caption": short or detailed[:30],
                }
    except Exception as e:
        print(f"[Warning] Full image VLM request failed: {e}", file=sys.stderr)

    return _fallback_global_desc(tag_item, language)


def describe_yolo_detections(
    image_path: Path,
    detections: list[dict],
    endpoint: str = "http://127.0.0.1:8080/v1",
    model_name: str | None = None,
    max_workers: int = 4,
    char_context: str | None = None,
    global_context: str | None = None,
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

        # Adaptive contextual padding: generous margin for tiny features so VLM sees crisp structure
        pad_x = max(bw * 0.25, 0.02)
        pad_y = max(bh * 0.25, 0.02)
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

            context_header = ""
            if global_context:
                context_header = f"【画面全局背景参考】：\n{global_context.strip()}\n\n"

            system_msg = (
                "你是一位精通二次元动漫插画与角色美术特征的资深视觉分析专家。"
                "针对提供的局部特写切片，请进行高纯度、高聚焦的细节特征分析。"
                "请直接输出合法的 JSON 格式结果，严禁输出多余的思考内容。"
            )
            user_msg = (
                f"{context_header}【局部特征分析任务（纯度聚焦）】：\n"
                f"请仔细观察从整幅画面中裁剪出的{prompt_focus}区域。\n"
                "【观察重点与隔离约束】：\n"
                "1. 纯粹聚焦该局部切片本身：描述切片内清晰可见的微观美术特征（如眼睑弧度/睫毛/瞳孔反光高光、唇峰曲线/光泽、发丝分簇质感、面部微晕或饰品雕纹）。\n"
                "2. 局部纯度隔离：严禁在描述中复述切片外部的无关大图背景（如远景大楼、天空、天体等）。\n"
                "3. 篇幅约束：输出 1-2 句符合二次元美术术语的紧凑生动描述（40-70字）。\n\n"
                "直接返回合法的 JSON 对象，仅包含两个键：\n"
                "- \"detailed_caption\": 1-2 句高纯度局部特征描写。\n"
                "- \"short_caption\": 4-8 个字的精炼中文短语。"
            )
        else:
            prompt_focus = f"{cls_name}"
            if char_context:
                prompt_focus += f" of character '{char_context}'"

            context_header = ""
            if global_context:
                context_header = f"[Global Context Reference]:\n{global_context.strip()}\n\n"

            system_msg = (
                "You are an expert anime character and visual art analyzer. "
                "Analyze cropped image patches focusing purely on the local visual features with strict isolation. "
                "Output JSON only."
            )
            user_msg = (
                f"{context_header}[Local Feature Inspection Task (Feature Isolation)]:\n"
                f"Focus strictly on the {prompt_focus} shown in this cropped image patch.\n"
                "Rules:\n"
                "1. Describe solely the features visible within this crop (shape, color gradients, specular highlights, textures, styling).\n"
                "2. Do not mention or repeat background or elements outside this crop.\n"
                "3. Keep description to 1-2 concise, vivid sentences (30-60 words).\n\n"
                "Return a valid JSON object with exactly two keys:\n"
                "- \"detailed_caption\": 1-2 detailed sentences describing only this feature.\n"
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

                detailed = clean_vlm_text(parsed.get("detailed_caption", ""))
                short = clean_vlm_text(parsed.get("short_caption", ""))

                # Fallback to reasoning extraction if JSON was incomplete
                if not detailed and c.get("reasoning_content"):
                    detailed, short = extract_caption_from_reasoning(
                        c["reasoning_content"], cls_name, language=language
                    )
                    detailed = clean_vlm_text(detailed)
                    short = clean_vlm_text(short)

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


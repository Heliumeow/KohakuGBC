# 二次元图像打标流水线与交互式场景图可视化指南

本指南详细介绍 KohakuGBC 中的二次元图像全自动打标与结构化场景图生成流水线。该系统深度整合了 **CLTagger**（二次元 ONNX 分类器）、**Ultralytics YOLO 模型群先验与预设系统**（二次元眼部、面部、口部、身体、饰品等先验框）、**VLM 局部细粒度微观描述器**、**KohakuGBC 核心**（全局 VLM + OWLv2/SAM 3 场景图推理）以及**独立的交互式 HTML 可视化器**。

[English Version](anime_pipeline.md)

---

## 1. 架构概览

通用的视觉-语言大模型（VLM）和开集目标检测器在面对动漫二次元插画时，往往缺乏特定角色（如明日方舟、碧蓝档案、原神、女神异闻录等）的原作背景知识，且对高度风格化的五官（如二次元眼睛、高光、发型、配饰）感知迟钝，容易漏检微观特征或捏造角色身份。

为了解决这一问题，KohakuGBC 实现了高内聚的模块化流水线：

```
[ 输入图像 ]
     │
     ├─────────────────────────────────────────┐
     ▼                                         ▼
[ 阶段 1a: CLTagger (ONNX 标签先验) ]    [ 阶段 1b: Ultralytics YOLO 空间先验 ]
  • 角色名、作品版权、特征标签             • 毫秒级多模型扫描 (眼睛/口部/饰品等)
  • 置信度过滤与智能排序                   • 支持领域预设 (anime-sfw, nsfw, real)
     │                                         │
     │                                         ▼
     │                               [ 阶段 1c: VLM 局部微观视觉描述 ]
     │                                 • 15% 上下文边缘扩展与自适应缩放
     │                                 • 并发生成详尽瞳孔/嘴形/材质描写
     │                                         │
     └────────────────────┬────────────────────┘
                          │
                          ▼
         [ 阶段 2: KohakuGBC 场景图递归推理 ]
            • VLM 结合动漫标签先验，精准识别角色与世界观背景
            • 保持 COCO 标准名词以确保开集检测器稳定定位
            • 将 YOLO 精准局部节点层级化挂载至人物下 (part of)
                          │
                          ▼
            [ 输出: 结构化场景图 (graphs.jsonl) ]
                          │
                          ▼
            [ 阶段 3: 交互式 HTML 可视化 (viewer.html) ]
            • SVG 边界框渲染与颜色分类
            • 点击/悬停查看细粒度微观描述与拓扑关系
```

---

## 2. 快速上手

### 2.1 安装与环境依赖

除了 KohakuGBC 的标准依赖外，二次元流水线还集成了 `onnxruntime` 与 `ultralytics`：

```bash
# 安装完整依赖环境
pip install -e ".[anime,torch]"
```

### 2.2 一键运行流水线

主入口脚本为 `scripts/cl_gbc_pipeline.py`，支持 Windows 路径（`C:\...`）与 WSL Linux 路径（`/mnt/c/...`）：

```bash
# 1. 单张图片运行完整流水线 (CLTagger + YOLO + VLM局部描述 + GBC + viewer.html)
.venv/bin/python scripts/cl_gbc_pipeline.py "C:\Users\YourUsername\Pictures\your_image.png"

# 2. 对整个目录进行批量处理
.venv/bin/python scripts/cl_gbc_pipeline.py "/mnt/c/Users/YourUsername/Pictures"

# 3. 极速先验打标模式 (<1 秒，不调用 GBC，直接生成轻量图与 viewer.html)
.venv/bin/python scripts/cl_gbc_pipeline.py "C:\path\to\image.png" --no-gbc
```

---

## 3. YOLO 空间先验与预设系统 (Presets)

你可以将任意针对特定部位训练的 YOLO 模型（`.pt` 或 `.onnx`）放入 `models/ultralytics/` 目录中。

为了避免模型间领域冲突（例如在全年龄插画中因运行 R18 模型导致大衣褶皱误检），系统内置了领域预设（`--yolo-preset`）：

### 3.1 预设分类

| 预设名称 (`--yolo-preset`) | 目标领域 | 启用的模型与过滤规则 |
|---|---|---|
| **`anime-sfw`** *(默认)* | 二次元全年龄插画/立绘 | 眼睛、嘴唇、饰品、面部、发型、手部（`Eyes.pt`、`adetailer2dMouth_v10.pt`、`jewelryDetector_v021.pt` 等）。**严格排除生殖器官/敏感部位模型**。 |
| **`anime-nsfw`** (或 `r18`) | 二次元成人向/敏感概念打标 | 在 SFW 基础上额外加载胸部、乳头、隐私部位模型（`breasts_seg.pt`、`nipples_v2_yolov11s-seg.pt`、`cockAndBallDetection2D_v20.pt` 等）。 |
| **`real`** | 三次元真人摄影 | 加载真人人脸/人体检测模型（如 `yolov8n-face.pt`），彻底隔离二次元手办卡通风格模型。 |
| **`all`** | 全量探索模式 | 加载 `models/ultralytics/` 目录内所有能扫描到的可用模型。 |
| **`auto`** | 混合数据集智能判定 | 自动分析第一阶段 CLTagger 标签：若检出强成人向标签（`nude`、`naked`、`penis`、`nipples` 等）$\ge 0.35$，自动激活 `anime-nsfw`；否则自动选用 `anime-sfw`。 |

### 3.2 使用示例

```bash
# 使用全年龄安全预设（默认，杜绝误检）
.venv/bin/python scripts/cl_gbc_pipeline.py "C:\path\to\image.png" --yolo-preset anime-sfw

# 启用 R18 解剖/敏感部位打标
.venv/bin/python scripts/cl_gbc_pipeline.py "C:\path\to\image.png" --yolo-preset anime-nsfw

# 自动按图像标签智能分流
.venv/bin/python scripts/cl_gbc_pipeline.py "C:\path\to\image.png" --yolo-preset auto

# 手动指定运行某几个特定模型
.venv/bin/python scripts/cl_gbc_pipeline.py "C:\path\to\image.png" --yolo-models Eyes.pt adetailer2dMouth_v10.pt

# 调整检测置信度与输入分辨率
.venv/bin/python scripts/cl_gbc_pipeline.py "C:\path\to\image.png" --yolo-conf 0.25 --yolo-imgsz 1280
```

### 3.3 自定义预设配置 (`models/ultralytics/presets.json`)

系统会优先加载 `models/ultralytics/presets.json`（或 `configs/yolo_presets.json`）。你可以添加专属预设或通配符规则：

```json
{
  "presets": {
    "anime-sfw": {
      "description": "Anime all-ages illustrations",
      "models": ["Eyes.pt", "adetailer2dMouth_v10.pt", "jewelryDetector_v021.pt"],
      "include_patterns": ["*eye*", "*mouth*", "*face*", "*jewelry*", "*hair*", "*hand*"],
      "exclude_patterns": ["*cock*", "*ball*", "*penis*", "*pussy*", "*vagina*", "*nipple*", "*cum*"]
    },
    "my-custom-preset": {
      "models": ["Eyes.pt", "jewelryDetector_v021.pt"]
    }
  }
}
```

---

## 4. VLM 局部细粒度看图描述器 (`--describe-yolo`)

当 YOLO 圈选出高精度的局部微观框后，系统默认会自动对其进行多线程局部看图：

1. **上下文保护性裁剪**：
   在检测框四周增加 15% 边缘留白，保留五官在面部或饰品在身体上的空间上下文，并自动将过小切片（如数十像素的眼睛）超分辨率缩放至 224px 以上以供 VLM 充分提取特征。
2. **结构化细粒度输出**：
   并发请求本地 VLM 服务（如 `llama-server`），针对瞳孔色调、虹膜花纹、唇形弧度、珠宝宝石材质生成结构化 `detailed_caption` 与简短短语 `short_caption`。
3. **层级拓扑挂载**：
   在场景图中自动寻找母体人物实体（如 `angelina` 或 `male character`），并将 YOLO 节点规范命名为 `angelina_eyes_0`、`angelina_mouth_0` 等，自动建立 `part of` 关系边。
4. **命令行控制**：
   - `--describe-yolo`（默认开启）
   - `--no-describe-yolo`（关闭局部看图，仅保留带来源置信度的快速占位节点）
   - `--yolo-vlm-workers 4`（并发线程数）

---

## 5. 模块化子包架构 (`scripts/anime_pipeline/`)

流水线代码已彻底解耦为高内聚子模块：

```text
scripts/
├── anime_pipeline/
│   ├── __init__.py           # 统一接口导出
│   ├── cltagger.py           # CLTagger ONNX 独立推理与标签/版权映射
│   ├── yolo.py               # Ultralytics 多模型扫描、预设过滤与 NMS 去重
│   ├── vlm_descriptor.py     # 局部图像裁剪、Prompt 构造与双层容错解析器
│   ├── graph_builder.py      # 拓扑层级挂载与快速图生成
│   ├── gbc_runner.py         # KohakuGBC 进程管控、配置覆盖与健康探针
│   └── utils.py              # 跨平台路径归一化与图片搜集
└── cl_gbc_pipeline.py        # 主入口编排脚本 (~200 行)
```

你可以在外部任意 Python 脚本中单独复用其中的任意模块：

```python
from scripts.anime_pipeline import (
    StandaloneCLTagger,
    UltralyticsEnsemble,
    describe_yolo_detections,
    run_cl_tagger,
)

# 仅执行 YOLO 部位检测
ensemble = UltralyticsEnsemble(preset="anime-sfw")
detections = ensemble.detect("path/to/image.png")
```

---

## 6. 产物目录与格式

每次运行均会在 `runs/` 下自动创建隔离的时间戳目录：
- 单图运行：`runs/<图片名>_<年月日_时分秒>/`
- 批处理：`runs/run_<年月日_时分秒>/`

### 目录产物说明：
- `input.jsonl`：CLTagger 筛选后的角色名、作品版权、特征标签与 YOLO 空间先验。
- `yolo_priors.json`：提取出的所有 YOLO 局部检测框及对应的 VLM 局部描述缓存。
- `graphs.jsonl`：最终生成的层级化场景图（包含节点详情、微观部件、构图与关系边）。
- `viewer.html`：单文件交互式 HTML 可视化页面。
- `report/`：性能耗时分析与执行事件日志。

> [!TIP]
> 根目录下会自动维护 `runs/latest` 软链接。每次运行完毕后，只需在浏览器中打开 `runs/latest/viewer.html` 即可直接检视最新标注结果。

---

## 7. 交互式可视化查看器

除了流水线自动生成的 `viewer.html` 外，你还可以使用可视化工具单独启动服务：

```bash
# 1. 重新生成或检视最新一次运行结果
.venv/bin/python scripts/visualize_graph.py

# 2. 自动唤起系统默认浏览器（支持在 WSL2 中直接唤起 Windows 浏览器）
.venv/bin/python scripts/visualize_graph.py --open

# 3. 启动本地轻量 HTTP 服务
.venv/bin/python scripts/visualize_graph.py --serve --port 8899
```

### 查看器特性：
- 🔹 **实体节点 (Entity, 青色)**：独立物理对象（人物主体、武器、服装、眼睛、嘴唇等）。
- 🟣 **复合结构 (Composition, 紫色)**：部分与整体的关系聚合（如双马尾、建筑群）。
- 🔸 **空间关系 (Relation, 琥珀色)**：实体间的空间交互（手持武器、身着大衣、挂饰佩戴等）。
- **双向联动聚焦**：点击图像上的 BBox 即可在侧边栏精确定位节点并展示其微观视觉描述；在侧边栏选择节点，图像上的框将产生高亮呼吸效果。

---

## 8. 显卡显存分配与部署建议 (RTX 5070 12GB 等消费级单卡)

在 12GB 显存单卡环境下本地运行的最佳实践组合：

| 引擎类型 | 模型选择 | 显存占用 | 说明 |
|---|---|---|---|
| **VLM 语言模型** | `llama-server` (Qwen3.5-9B Q8_0 GGUF + BF16 mmproj) | ~9.5 GB | 负责整图构图理解、关系推理及 YOLO 局部看图 |
| **开集目标检测** | `google/owlv2-base-patch16-ensemble` (CUDA) | ~600 MB | 负责 GBC 内部常规大实体圈框 |
| **文本嵌入模型** | `google/embeddinggemma-300m` (CUDA) | ~600 MB | 负责实体去重与近义合并 |
| **YOLO 局部模型群** | `Eyes.pt`, `adetailer2dMouth_v10.pt` 等 | ~180 MB | 毫秒级轻量推理，按需载入显存 |

通过这种显存精细切分，单张 12GB 显卡即可在显存不溢出的前提下，同时流畅运行全套二次元多阶段打标流水线。

# Local GPU machine configuration using llama-server (OpenAI API flavour) for VLM
# with OWLv2 detector and EmbeddingGemma on GPU for fast anime scene graph generation.

import os

_VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
_VLM_ENDPOINT = os.environ.get("VLM_ENDPOINT", "http://127.0.0.1:8080/v1")

engines = {
    "vlm": {
        "entry": "api",
        "protocol": "openai",
        "flavour": "llama.cpp",
        "model": _VLM_MODEL,
        "endpoints": [{"base_url": _VLM_ENDPOINT, "max_inflight": 4}],
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

machine = {
    "vlm_waiting": 16,
    "report": "runs/test_run/report",
}

source = {
    "kind": "jsonl",
    "path": "runs/test_run/input.jsonl",
}

sink = {
    "kind": "graphs",
    "path": "runs/test_run/graphs.jsonl",
}

controls = {
    "tag_hint": True,
}



# Test machine configuration using llama-server (OpenAI API flavour) for VLM
# and local CPU models for detect & embed to prevent VRAM contention with llama-server.

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



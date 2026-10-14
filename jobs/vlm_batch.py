"""Batch LLM / multimodal (OCR) inference with vLLM. Runs ON the Colab VM.

    python job.py IN_DIR OUT_DIR
or, on a warm worker (colab-link serve), handle(IN_DIR, OUT_DIR) once per job with
the model loaded only on the first call.

IN_DIR/tasks.jsonl, one task per line:
    {"id": "a1", "prompt": "Summarise ...."}
    {"id": "o1", "prompt": "Transcribe all text.", "image": "imgs/receipt.png"}
Optional per-task keys: "system", "max_tokens", "temperature".

Writes OUT_DIR/results.jsonl ({"id", "output", "prompt_tokens", "output_tokens"})
and OUT_DIR/metrics.json. Model/limits come from env: MODEL, MAX_MODEL_LEN,
GPU_MEM_UTIL, MAX_TOKENS, MAX_NUM_SEQS, ENFORCE_EAGER=1 (skip torch.compile/CUDA graphs),
ENABLE_THINKING=1 (Qwen3.5+ reasoning; off by default).
"""
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-VL-4B-Instruct")
out_dir: Path
_llm = None  # kept across handle() calls on a warm worker


def progress(stage, **kw):
    (out_dir / "progress.json").write_text(json.dumps({"stage": stage, "t": time.time(), **kw}))
    print(f"[job] {stage} {kw}", flush=True)


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def load_llm():
    global _llm
    if _llm is None:
        from vllm import LLM  # noqa: E402  (import is slow; time it)

        _llm = LLM(
            model=MODEL,
            max_model_len=int(os.environ.get("MAX_MODEL_LEN", "8192")),
            gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.90")),
            limit_mm_per_prompt={"image": 1},
            enforce_eager=os.environ.get("ENFORCE_EAGER", "0") == "1",
            quantization=os.environ.get("QUANTIZATION") or None,  # e.g. "fp8" = on-the-fly on L4/G4
            max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "256")),
        )
    return _llm


def handle(in_dir: Path, out_dir_: Path):
    # vLLM may start its engine with multiprocessing 'spawn' (seen on G4/Blackwell),
    # which re-imports the main module, so all work must sit behind the __main__ guard.
    global out_dir
    out_dir = out_dir_
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in (in_dir / "tasks.jsonl").read_text().splitlines() if l.strip()]
    progress("loading", model=MODEL, tasks=len(tasks), warm=_llm is not None)

    t0 = time.time()
    from vllm import SamplingParams  # noqa: E402

    llm = load_llm()
    # Qwen3.5+ think by default; OCR/extraction wants the plain answer.
    THINKING = os.environ.get("ENABLE_THINKING", "0") == "1"
    load_s = time.time() - t0
    progress("generating", load_s=round(load_s, 1))

    conversations, params = [], []
    for t in tasks:
        msgs = [{"role": "system", "content": t["system"]}] if t.get("system") else []
        content = [{"type": "text", "text": t["prompt"]}]
        if t.get("image"):
            content.insert(0, {"type": "image_url", "image_url": {"url": data_url(in_dir / t["image"])}})
        msgs.append({"role": "user", "content": content})
        conversations.append(msgs)
        params.append(SamplingParams(
            max_tokens=int(t.get("max_tokens", os.environ.get("MAX_TOKENS", "1024"))),
            temperature=float(t.get("temperature", 0.0)),
        ))

    t1 = time.time()
    outs = llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": THINKING})
    gen_s = time.time() - t1

    out_tokens = 0
    with open(out_dir / "results.jsonl", "w") as f:
        for t, o in zip(tasks, outs):
            n = len(o.outputs[0].token_ids)
            out_tokens += n
            f.write(json.dumps({"id": t["id"], "output": o.outputs[0].text,
                                "prompt_tokens": len(o.prompt_token_ids), "output_tokens": n},
                               ensure_ascii=False) + "\n")

    import torch  # noqa: E402

    metrics = {"model": MODEL, "tasks": len(tasks), "gpu": torch.cuda.get_device_name(0),
               "load_s": round(load_s, 1), "gen_s": round(gen_s, 1),
               "output_tokens": out_tokens, "output_tok_per_s": round(out_tokens / gen_s, 1)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    progress("done", **metrics)


if __name__ == "__main__":
    handle(Path(sys.argv[1]), Path(sys.argv[2]))

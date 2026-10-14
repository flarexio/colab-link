"""Build examples/ocr_demo: a few synthetic images with known text + text tasks.

    uv run --with pillow examples/make_ocr_demo.py
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).parent / "ocr_demo"
(root / "imgs").mkdir(parents=True, exist_ok=True)

docs = {
    "receipt": ["ACME CAFE", "Latte        x2   $9.00", "Bagel        x1   $3.50",
                "TOTAL             $12.50", "Paid VISA **** 4821"],
    "memo": ["MEMO 2026-10-14", "To: Ops team", "Rotate the API keys before Friday.",
             "Owner: Lin  Ticket: OPS-3317"],
    "table": ["Item | Qty | Price", "Widget | 4 | 2.25", "Gadget | 1 | 19.99", "Gizmo | 12 | 0.40"],
}
font = ImageFont.load_default(size=28)
tasks = []
for name, lines in docs.items():
    img = Image.new("RGB", (720, 60 + 44 * len(lines)), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((30, 30 + 44 * i), line, fill="black", font=font)
    img.rotate(1.5, expand=True, fillcolor="white").save(root / "imgs" / f"{name}.png")
    tasks.append({"id": f"ocr-{name}", "image": f"imgs/{name}.png",
                  "prompt": "Transcribe all text in this image exactly, one line per line.",
                  "expected": "\n".join(lines)})

tasks += [
    {"id": "llm-summary", "prompt": "用一句繁體中文總結：Colab CLI lets you rent GPUs from a terminal, "
     "run scripts remotely, and release the VM when finished, billed in compute units."},
    {"id": "llm-json", "prompt": 'Extract {"name","date","amount"} as JSON only: '
     '"Invoice from Contoso dated 3 Oct 2026 for 1,240.50 USD."'},
]
(root / "tasks.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n")
print(f"wrote {len(tasks)} tasks to {root}")

"""Build examples/bench: N synthetic OCR images + N text tasks, with expected
answers, for throughput / accuracy comparisons across GPUs and models.

    env -u PYTHONPATH uv run -q --no-project --with pillow examples/make_bench.py --n 100
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()
rng = random.Random(42)
root = Path(__file__).parent / "bench"
(root / "imgs").mkdir(parents=True, exist_ok=True)

items = ["Latte", "Bagel", "Mocha", "Scone", "Tea", "Muffin", "Toast", "Juice", "Salad", "Wrap"]
shops = ["ACME CAFE", "BLUE DOOR", "NORTH DELI", "SUNRISE BAKERY", "CORNER BISTRO"]
font = ImageFont.load_default(size=26)
tasks = []
for i in range(a.n):
    rows = rng.sample(items, rng.randint(3, 6))
    lines = [rng.choice(shops), f"Order #{rng.randint(1000, 9999)}  2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"]
    total = 0.0
    for it in rows:
        q, p = rng.randint(1, 4), rng.randint(150, 990) / 100
        total += q * p
        lines.append(f"{it} x{q} {q * p:.2f}")
    lines.append(f"TOTAL {total:.2f}")
    img = Image.new("RGB", (640, 50 + 40 * len(lines)), "white")
    d = ImageDraw.Draw(img)
    for j, line in enumerate(lines):
        d.text((24, 24 + 40 * j), line, fill="black", font=font)
    img.rotate(rng.uniform(-2, 2), expand=True, fillcolor="white").save(root / "imgs" / f"r{i:04d}.png")
    tasks.append({"id": f"ocr-{i:04d}", "image": f"imgs/r{i:04d}.png", "max_tokens": 256,
                  "prompt": "Transcribe all text in this receipt exactly, one line per line.",
                  "expected": "\n".join(lines)})

for i in range(a.n):
    name, amt = rng.choice(["Contoso", "Fabrikam", "Northwind", "Tailspin"]), rng.randint(100, 99999) / 100
    day, mon = rng.randint(1, 28), rng.choice(["Jan", "Mar", "Jun", "Sep", "Nov"])
    tasks.append({"id": f"llm-{i:04d}", "max_tokens": 128,
                  "prompt": f'Extract {{"vendor","date","amount"}} as compact JSON only: '
                            f'"Invoice from {name} dated {day} {mon} 2026 for {amt:,.2f} USD."',
                  "expected": json.dumps({"vendor": name, "date": f"{day} {mon} 2026", "amount": amt})})

(root / "tasks.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n")
print(f"wrote {len(tasks)} tasks to {root}")

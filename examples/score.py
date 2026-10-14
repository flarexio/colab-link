"""Score a results dir against examples/<set>/tasks.jsonl expectations.

    python3 examples/score.py examples/bench results/bench-a100
OCR: exact match after collapsing whitespace, plus character accuracy.
LLM: JSON parses and vendor/amount match.
"""
import difflib
import json
import re
import sys
from pathlib import Path

tasks = {t["id"]: t for t in map(json.loads, Path(sys.argv[1], "tasks.jsonl").read_text().splitlines())}
res = {r["id"]: r for r in map(json.loads, Path(sys.argv[2], "results.jsonl").read_text().splitlines())}
norm = lambda s: re.sub(r"[ \t]+", " ", s.strip())  # noqa: E731

ocr_exact = ocr_n = llm_ok = llm_n = 0
char_acc = []
for tid, t in tasks.items():
    if "expected" not in t:
        continue
    out = res.get(tid, {}).get("output", "")
    if tid.startswith("ocr"):
        ocr_n += 1
        ocr_exact += norm(out) == norm(t["expected"])
        char_acc.append(difflib.SequenceMatcher(None, norm(out), norm(t["expected"])).ratio())
    else:
        llm_n += 1
        exp = json.loads(t["expected"])
        try:
            got = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
            # The prompt doesn't demand a numeric amount, so "474.03 USD" counts.
            amt = float(re.search(r"[\d.]+", str(got.get("amount")).replace(",", "")).group(0))
            llm_ok += got.get("vendor") == exp["vendor"] and abs(amt - exp["amount"]) < 0.01
        except Exception:
            pass

print(json.dumps({"ocr_exact": f"{ocr_exact}/{ocr_n}",
                  "ocr_char_acc": round(sum(char_acc) / max(len(char_acc), 1), 4),
                  "llm_json_ok": f"{llm_ok}/{llm_n}"}))

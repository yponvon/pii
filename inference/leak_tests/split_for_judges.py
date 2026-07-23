"""Split the redacted transcripts into N chunks, one per leak judge.

Second step of the residual-PII leak test. Divides redacted_all.jsonl into N
roughly equal chunks so each LLM judge can review one chunk independently.

Usage: python split_for_judges.py [N]   (default 8)
Output: judge_chunk_*.jsonl
"""
import sys, math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "leak_tests"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
rows = open(OUT / "redacted_all.jsonl").readlines()
size = math.ceil(len(rows) / N)
for k in range(N):
    chunk = rows[k * size:(k + 1) * size]
    if chunk:
        (OUT / f"judge_chunk_{k}.jsonl").write_text("".join(chunk))
        print(f"judge_chunk_{k}.jsonl: {len(chunk)} transcripts")
print("Each judge reads one chunk (prompt: leak_judge_prompt.md) and writes judge_result_<k>.json")

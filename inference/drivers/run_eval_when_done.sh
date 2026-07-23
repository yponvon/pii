#!/bin/zsh
while pgrep -f train_lora_synthetic.py >/dev/null; do sleep 60; done
P=/Users/yvonne/INTERN/echolens/venv/bin/python3
A=/Users/yvonne/INTERN/echolens/data_25_june/finetune/lora_fulltext_mixed_output/best
echo "=== TRAINING DONE ==="; tail -c 400 train_mixed.log | tr '\r' '\n' | tail -3
echo; echo "=== SYNTHETIC-STYLE 9-LABEL EVAL ON HELD-OUT AUTHENTIC TEST ==="
$P benchmark_all_labels.py --synthetic "$A" 2>&1 | tail -30
echo; echo "=== LEGACY BENCHMARK (comparable to 309 checkpoint) ==="
$P benchmark_all_labels.py "$A" 2>&1 | tail -25

#!/bin/zsh
sleep 30
while pgrep -f train_lora_mixed2.py >/dev/null; do sleep 60; done
P=/Users/yvonne/INTERN/echolens/venv/bin/python3
A=/Users/yvonne/INTERN/echolens/data_25_june/finetune/lora_fulltext_mixed2_output/best
echo "=== eval_loss history ==="; grep -o "'eval_loss': [0-9.]*" train_mixed2.log | tail -20
echo "=== TEST (419 authentic, 9 labels) ==="
$P benchmark_all_labels.py --synthetic "$A" 2>&1 | grep -E "TP=|OVERALL"

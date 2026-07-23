#!/bin/zsh
cd data_25_june/finetune
while pgrep -f "train_sweep.py|run_sweep.sh" >/dev/null; do sleep 120; done
P=venv/bin/python3
echo "===== SWEEP val_loss ranking ====="
for t in D A B C; do
  b=$(grep -o "'eval_loss': [0-9.]*" sweep_$t.log 2>/dev/null | grep -oE "[0-9.]+" | sort -n | head -1)
  echo "$t best_val_loss=$b"
done
WIN=$(for t in D A B C; do b=$(grep -o "'eval_loss': [0-9.]*" sweep_$t.log 2>/dev/null|grep -oE "[0-9.]+"|sort -n|head -1); echo "$b $t"; done | sort -n | head -1 | awk '{print $2}')
echo "WINNER=$WIN"
echo; echo "########## FINAL EVAL: 419 authentic, 9 labels, same protocol ##########"
for name adap in baseline None run1 lora_fulltext_mixed_output/best winner lora_sweep_${WIN}_output/best; do
  echo "===== $name ($adap) ====="
  $P benchmark_all_labels.py --synthetic $adap 2>&1 | grep -E "TP=|OVERALL|ALL "
  echo
done

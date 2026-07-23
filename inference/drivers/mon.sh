#!/bin/zsh
cd data_25_june/finetune
while pgrep -f train_lora_mixed2.py >/dev/null; do
  step=$(grep -oE "[0-9]+/2865|[0-9]+/[0-9]+ \[" train_mixed2.log | tail -1)
  ev=$(grep -o "'eval_loss': [0-9.]*" train_mixed2.log | tail -1)
  echo "$(date +%H:%M) step=$step $ev"
  sleep 300
done
echo "=== TRAINING EXITED ==="
grep -o "'eval_loss': [0-9.]*" train_mixed2.log
tail -3 train_mixed2.log | tr '\r' '\n' | tail -3

#!/bin/zsh
cd data_25_june/finetune
P=venv/bin/python3
# tag | enc_lr | task_lr | max_len | train | val
CFG=(
 "D 1e-5 5e-4 512 train_mixed2.jsonl val_mixed2.jsonl"
 "A 5e-6 2.5e-4 512 train_mixed2.jsonl val_mixed2.jsonl"
 "B 1e-5 5e-4 512 train_B.jsonl val_B.jsonl"
 "C 5e-6 2.5e-4 1024 train_mixed2.jsonl val_mixed2.jsonl"
)
for row in "${CFG[@]}"; do
  set -- ${(z)row}; tag=$1
  echo "########## RUN $tag  enc=$2 task=$3 maxlen=$4 train=$5 ##########  $(date +%H:%M)"
  RUN_TAG=$tag ENC_LR=$2 TASK_LR=$3 MAX_LEN=$4 EPOCHS=3 TRAIN=$5 VAL=$6 \
    $P train_sweep.py > sweep_$tag.log 2>&1
  echo "RUN $tag eval_loss history:"; grep -o "'eval_loss': [0-9.]*" sweep_$tag.log
done
echo "########## SWEEP COMPLETE ##########  $(date +%H:%M)"
for tag in D A B C; do
  best=$(grep -o "'eval_loss': [0-9.]*" sweep_$tag.log | grep -oE "[0-9.]+" | sort -n | head -1)
  echo "$tag best_val_loss=$best"
done

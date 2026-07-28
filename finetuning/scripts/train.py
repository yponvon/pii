"""Fine-tune the GLiNER2 PII model on the mixed training split.

Trains with LoRA on a mixed dataset of synthetic and authentic examples. The
configuration uses a low encoder learning rate (1e-5) with a higher task-head
learning rate (5e-4) and a 512-token maximum sequence length. Training runs for
up to 15 epochs; early stopping (patience 3, monitoring eval_loss) selects the
stopping point, and the best checkpoint is saved. On completion the script writes
the loss history and renders a train and validation loss PNG so plateau versus
oscillation is visible.

Usage:
  cd pii/inference/harness
  python3 benchmark_all_labels.py --synthetic --windowed <output_dir>/best
  # --windowed uses overlapping-window inference so PII beyond the 512-token
  # limit is still scored.
"""
import json
import subprocess
import sys
from pathlib import Path

from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

MODEL_PATH = "fastino/gliner2-privacy-filter-PII-multi"
_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA = _ROOT / "training_data" / "train_mixed2.jsonl"
VAL_DATA = _ROOT / "val_data" / "val_mixed2.jsonl"
OUT = _ROOT / "models" / "finetuned_pii_9label"
LOGS = _ROOT / "finetuning" / "logs"


def main() -> None:
    cfg = TrainingConfig(
        output_dir=str(OUT),
        experiment_name="finetuned_pii_9label",
        num_epochs=15,
        batch_size=4, eval_batch_size=4,
        encoder_lr=1e-5, task_lr=5e-4,
        eval_strategy="epoch",
        early_stopping=True, early_stopping_patience=3,
        save_best=True, metric_for_best="eval_loss",
        greater_is_better=False,
        use_lora=True, lora_r=16, lora_alpha=32.0, lora_dropout=0.0,
        save_adapter_only=True, max_len=512, seed=42,
    )
    print(f"WORKABLE RUN (D config) -> {OUT}")
    model = GLiNER2.from_pretrained(MODEL_PATH)
    results = GLiNER2Trainer(model=model, config=cfg).train(
        train_data=str(TRAIN_DATA),
        eval_data=str(VAL_DATA),
    )
    # Save the loss history and render the loss graph.
    try:
        hist_path = OUT / "history.json"
        hist_path.write_text(json.dumps(results, default=str, indent=1))
        print(f"saved {hist_path}")
    except Exception as exc:
        print("could not serialize results:", exc)
    # Plot from the log file, which stays robust even if the results shape varies.
    log = LOGS / "train.log"
    if log.exists():
        subprocess.run(
            [sys.executable,
             str(Path(__file__).parent / "plot_loss.py"), str(log), str(OUT / "loss.png")],
            check=False,
        )
        print(f"loss graph: {OUT / 'loss.png'}")


if __name__ == "__main__":
    main()

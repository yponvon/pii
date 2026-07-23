"""
train_windowed.py -- retrain the 9-label model on the WINDOWED training split.

Unlike the earlier full-transcript run (which truncated each ~1900-token call to
its first 512 tokens and so never learned the ~23% of PII that appears later),
this trains on overlapping 256-token windows built by
build_windowed_training_data.py, so the model sees the WHOLE call.

Config: same LoRA / learning rates as the workable run, but tuned for the
smaller windows -- max_len 320 (windows are 256 tokens; 320 leaves headroom) and
a larger batch, since short sequences fit comfortably. Early stopping (patience
3) decides when to stop. A train+val loss PNG is rendered at the end.

Batch size can be overridden with the BATCH env var (used by the orchestrator to
fall back to a smaller batch if memory is tight):
  BATCH=4 ./venv/bin/python pii/finetuning/scripts/train_windowed.py
"""
import json
import os
import subprocess
from pathlib import Path

from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

BASE = Path(__file__).resolve().parent.parent.parent   # .../pii
MODEL_PATH = "fastino/gliner2-privacy-filter-PII-multi"
SPLITS = BASE / "finetuning" / "splits"
OUT = BASE / "models" / "finetuned_windowed_9label"
LOGS = BASE / "finetuning" / "logs"

BATCH = int(os.environ.get("BATCH", "8"))


def main() -> None:
    cfg = TrainingConfig(
        output_dir=str(OUT),
        experiment_name="windowed_9label",
        num_epochs=15,
        batch_size=BATCH, eval_batch_size=BATCH,
        encoder_lr=1e-5, task_lr=5e-4,
        eval_strategy="epoch",
        early_stopping=True, early_stopping_patience=3,
        save_best=True, metric_for_best="eval_loss",
        greater_is_better=False,
        use_lora=True, lora_r=16, lora_alpha=32.0, lora_dropout=0.0,
        save_adapter_only=True, max_len=320, seed=42,
    )
    print(f"WINDOWED RETRAIN (batch={BATCH}, max_len=320) -> {OUT}")
    model = GLiNER2.from_pretrained(MODEL_PATH)
    results = GLiNER2Trainer(model=model, config=cfg).train(
        train_data=str(SPLITS / "train_windowed.jsonl"),
        eval_data=str(SPLITS / "val_windowed.jsonl"),
    )
    try:
        (OUT / "history.json").write_text(json.dumps(results, default=str, indent=1))
    except Exception as exc:
        print("could not serialize results:", exc)

    log = LOGS / "train_windowed.log"
    if log.exists():
        subprocess.run(
            [str(BASE.parent / "venv" / "bin" / "python3"),
             str(Path(__file__).parent / "plot_loss.py"), str(log), str(OUT / "loss.png")],
            check=False,
        )
        print(f"loss graph: {OUT / 'loss.png'}")


if __name__ == "__main__":
    main()

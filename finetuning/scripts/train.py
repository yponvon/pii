"""Fine-tune the GLiNER2 PII model on the training split.

Trains with LoRA on the mixed synthetic + authentic training split
(data/train.jsonl, built by build_splits.py). The configuration uses a low
encoder learning rate (1e-5) with a higher task-head learning rate (5e-4) and a
512-token maximum sequence length. Training runs for up to 15 epochs; early
stopping (patience 3, monitoring eval_loss) selects the stopping point, and the
best checkpoint is saved as <output_dir>/best. On completion the script writes
the loss history and renders a train/validation loss PNG so plateau versus
oscillation is visible.

Each run writes to its OWN folder so retraining never overwrites the shipped
keeper adapter (models/finetuned_pii_9label/, loaded by inference by default):

    python train.py                 -> models/runs/run_<timestamp>/best
    python train.py my_experiment    -> models/runs/my_experiment/best   (refuses
                                        to overwrite an existing run)

A run is a *candidate*. Benchmark its adapter, and only if it beats the keeper
promote it -- e.g. copy models/runs/<name>/best over
models/finetuned_pii_9label/best. Training itself never touches the keeper.

Usage:
  python finetuning/data_prep/build_splits.py     # build data/train.jsonl + data/val.jsonl
  python finetuning/scripts/train.py [run_name]    # train a candidate
  python evaluation/run_benchmark.py               # score it (point the adapter at the new run)
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

MODEL_PATH = "fastino/gliner2-privacy-filter-PII-multi"
_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA = _ROOT / "data" / "train.jsonl"
VAL_DATA = _ROOT / "data" / "val.jsonl"
RUNS_DIR = _ROOT / "models" / "runs"     # one subfolder per training run (gitignored)
KEEPER = _ROOT / "models" / "finetuned_pii_9label"   # the shipped adapter; never written here
LOGS = _ROOT / "finetuning" / "logs"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a PII adapter candidate.")
    ap.add_argument("run_name", nargs="?", default=None,
                    help="name for this run's output folder under models/runs/; "
                         "defaults to run_<timestamp>.")
    args = ap.parse_args()

    run_name = args.run_name or "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RUNS_DIR / run_name
    if out.exists() and any(out.iterdir()):
        print(f"ERROR: {out} already exists and is not empty. Pick another run "
              f"name so an earlier run is not overwritten.")
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)

    cfg = TrainingConfig(
        output_dir=str(out),
        experiment_name=run_name,
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
    print(f"Training run '{run_name}' -> {out}")
    print(f"(the shipped keeper at {KEEPER} is left untouched; promote this run "
          f"only if it beats the keeper)")
    model = GLiNER2.from_pretrained(MODEL_PATH)
    results = GLiNER2Trainer(model=model, config=cfg).train(
        train_data=str(TRAIN_DATA),
        eval_data=str(VAL_DATA),
    )
    # Save the loss history and render the loss graph.
    try:
        hist_path = out / "history.json"
        hist_path.write_text(json.dumps(results, default=str, indent=1))
        print(f"saved {hist_path}")
    except Exception as exc:
        print("could not serialize results:", exc)
    # Plot from the log file, which stays robust even if the results shape varies.
    log = LOGS / "train.log"
    if log.exists():
        subprocess.run(
            [sys.executable,
             str(Path(__file__).parent / "plot_loss.py"), str(log), str(out / "loss.png")],
            check=False,
        )
        print(f"loss graph: {out / 'loss.png'}")
    print(f"\nBest adapter: {out / 'best'}")
    print("Benchmark it, then promote over models/finetuned_pii_9label/best if it wins.")


if __name__ == "__main__":
    main()

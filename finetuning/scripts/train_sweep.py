"""
train_sweep.py  --  one parametrised LoRA run, driven by env vars, for the
hyperparameter queue (run_sweep.sh runs these sequentially; MPS is single-GPU).

Env: RUN_TAG, TRAIN, VAL, ENC_LR, TASK_LR, MAX_LEN, EPOCHS.
Output dir = lora_sweep_<RUN_TAG>_output.

NOTE: the whole body MUST live under `if __name__ == "__main__"`. macOS spawns
DataLoader workers, which re-import this module; top-level training code would
recurse and deadlock (this bit the first attempt).
"""
import os
from pathlib import Path

MODEL_PATH = "fastino/gliner2-privacy-filter-PII-multi"
D = Path(__file__).resolve().parents[2] / "finetuning" / "splits"


def main() -> None:
    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
    tag = os.environ["RUN_TAG"]
    cfg = TrainingConfig(
        output_dir=str(D / f"lora_sweep_{tag}_output"),
        experiment_name=f"sweep_{tag}",
        num_epochs=int(os.environ.get("EPOCHS", "5")),
        batch_size=4, eval_batch_size=4,
        encoder_lr=float(os.environ.get("ENC_LR", "1e-5")),
        task_lr=float(os.environ.get("TASK_LR", "5e-4")),
        eval_strategy="epoch",
        early_stopping=True, early_stopping_patience=3,
        save_best=True, metric_for_best="eval_loss",
        greater_is_better=False,
        use_lora=True, lora_r=16, lora_alpha=32.0, lora_dropout=0.0,
        save_adapter_only=True,
        max_len=int(os.environ.get("MAX_LEN", "512")),
        seed=42,
    )
    print(f"[{tag}] enc_lr={cfg.encoder_lr} task_lr={cfg.task_lr} "
          f"max_len={cfg.max_len} epochs={cfg.num_epochs}", flush=True)
    model = GLiNER2.from_pretrained(MODEL_PATH)
    GLiNER2Trainer(model=model, config=cfg).train(
        train_data=str(D / os.environ["TRAIN"]),
        eval_data=str(D / os.environ["VAL"]),
    )


if __name__ == "__main__":
    main()

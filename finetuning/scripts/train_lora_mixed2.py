"""
train_lora_synthetic.py

LoRA fine-tuning on the synthetic corpus split (generated_data_14jul/,
738 train / 129 val / 153 held-out test, 9 labels -- the base 7 plus
account_number and full_name). See build_synthetic_training_data.py for how
the split was built.

Hyperparameters are IDENTICAL to train_lora_majority.py (the run that produced
the current best results); only the data paths, experiment_name and output_dir
differ, so the comparison is apples-to-apples.

Usage:
  python3 train_lora_mixed2.py
"""

from pathlib import Path

from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

MODEL_PATH = "fastino/gliner2-privacy-filter-PII-multi"
DATA_DIR = Path(__file__).resolve().parents[2] / "finetuning" / "splits"


def main() -> None:
    train_path = DATA_DIR / "train_mixed2.jsonl"
    val_path = DATA_DIR / "val_mixed2.jsonl"
    output_dir = Path(__file__).resolve().parents[2] / "models" / "lora_fulltext_mixed2_output"

    config = TrainingConfig(
        output_dir=str(output_dir),
        experiment_name="gliner2_pii_fulltext_mixed2",
        num_epochs=15,
        batch_size=4,
        eval_batch_size=4,
        encoder_lr=1e-5,
        task_lr=5e-4,
        eval_strategy="epoch",
        early_stopping=True,
        early_stopping_patience=3,
        save_best=True,
        metric_for_best="eval_loss",
        greater_is_better=False,
        use_lora=True,
        lora_r=16,
        lora_alpha=32.0,
        lora_dropout=0.0,
        save_adapter_only=True,
        max_len=1024,
        seed=42,
    )

    print(f"Loading base model: {MODEL_PATH}")
    model = GLiNER2.from_pretrained(MODEL_PATH)

    trainer = GLiNER2Trainer(model=model, config=config)

    print(f"Training on synthetic split  train={train_path}  val={val_path}")
    results = trainer.train(train_data=str(train_path), eval_data=str(val_path))

    print("\n=== Training complete ===")
    print(results)
    print(f"\nAdapter saved under: {output_dir}")


if __name__ == "__main__":
    main()

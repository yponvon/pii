"""Parse a training log and plot train and validation loss.

Renders the per-step training loss and per-epoch validation loss so plateau
versus oscillation is visible. Can be run at any time, whether training is still
in progress or already finished.

Usage:
  python3 plot_loss.py <path/to/train.log> [out.png]
  # The output defaults to the log name with a .png suffix.

Reads:
  - per-step training loss from "loss=<float> ... epoch=<float>"
  - per-epoch validation loss from "eval_loss: <float>" (or "'eval_loss': <float>")
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(log_text):
    # Training: per-step loss paired with its epoch fraction.
    train = []  # (epoch_float, loss)
    for m in re.finditer(r"loss=([0-9.]+),\s*lr=[0-9.eE+-]+,[^\n]*?epoch=([0-9.]+)", log_text):
        train.append((float(m.group(2)), float(m.group(1))))
    # Validation: one value per epoch, in order.
    val = [float(x) for x in re.findall(r"'?eval_loss'?:\s*([0-9.]+)", log_text)]
    return train, val


def main():
    if len(sys.argv) < 2:
        print("usage: plot_loss.py <train.log> [out.png]"); sys.exit(1)
    log = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else log.with_suffix(".png")
    train, val = parse(log.read_text(errors="ignore"))
    if not train and not val:
        print("no loss data found yet in", log); return

    fig, ax = plt.subplots(figsize=(10, 5))
    if train:
        xs = [e for e, _ in train]
        ys = [l for _, l in train]
        ax.plot(xs, ys, color="#9ec5fe", lw=0.8, alpha=0.7, label="train loss (per step)")
        # Smoothed training curve.
        if len(ys) >= 20:
            k = max(5, len(ys) // 50)
            sm = [sum(ys[max(0, i - k):i + 1]) / len(ys[max(0, i - k):i + 1]) for i in range(len(ys))]
            ax.plot(xs, sm, color="#1c6dd0", lw=1.8, label=f"train loss (smoothed, k={k})")
    if val:
        vx = list(range(len(val)))  # Epoch index.
        ax.plot(vx, val, "o-", color="#d6336c", lw=2, ms=7, label="val loss (per epoch)")
        best = min(range(len(val)), key=lambda i: val[i])
        ax.axvline(best, color="#d6336c", ls="--", alpha=0.4)
        ax.annotate(f"best ep{best}={val[best]:.1f}", (best, val[best]),
                    textcoords="offset points", xytext=(6, 8), color="#d6336c")

    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(f"Training progression — {log.name}")
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"saved {out}")
    # Also print the validation trajectory as text for a quick plateau or oscillation read.
    if val:
        print("val loss by epoch:", " -> ".join(f"{v:.1f}" for v in val))
        trend = "descending (good)" if val == sorted(val, reverse=True) else \
                "oscillating/rising (watch)" if len(val) > 1 else "one point"
        print("trend:", trend)


if __name__ == "__main__":
    main()

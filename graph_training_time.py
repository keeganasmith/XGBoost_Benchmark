#!/usr/bin/env python3
import os
import re
import sys
from statistics import mean
import matplotlib.pyplot as plt

TIME_RE = re.compile(r"Training Time \(seconds\):\s*([0-9]*\.?[0-9]+)")

def extract_training_time(path: str) -> float:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    m = TIME_RE.search(text)
    if not m:
        raise ValueError(f"Could not find training time line in {path}")
    return float(m.group(1))

def get_group_times(paths):
    times = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")
        times.append(extract_training_time(p))
    return times

def main():
    regular_paths = [f"regular_outputs/output_{i}" for i in range(3)]
    composable_paths = [f"composable_outputs/composable_output_{i}" for i in range(3)]

    regular_times = get_group_times(regular_paths)
    composable_times = get_group_times(composable_paths)

    regular_avg = mean(regular_times)
    composable_avg = mean(composable_times)

    print("Regular training times (s):", regular_times)
    print("Composable training times (s):", composable_times)
    print(f"Regular average (s): {regular_avg:.4f}")
    print(f"Composable average (s): {composable_avg:.4f}")

    # Bar plot
    labels = ["Regular", "Composable"]
    avgs = [regular_avg, composable_avg]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, avgs)
    plt.ylabel("Average Training Time (seconds)")
    plt.title("Average Training Time: Regular vs Composable")

    # Value labels on bars
    for i, v in enumerate(avgs):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("avg_training_time_comparison.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

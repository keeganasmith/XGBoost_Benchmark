#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
COLS_TO_EXCLUDE=["available_GB", "swap_total_GB", "mem_total_GB",
        "swap_free_GB", "mem_available_GB"]

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(f"{path}: expected a 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in COLS_TO_EXCLUDE:
        if(col in list(df.columns)):
            df = df.drop(columns=[col])

    # Convert numeric columns
    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Create relative time (seconds from start)
    df["t_seconds"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem", required=True, help="Memory CSV (mem_* columns)")
    ap.add_argument("--disk", required=True, help="Disk CSV (total_GB, used_GB, etc.)")
    ap.add_argument("--out", default=None, help="Optional output PNG")
    args = ap.parse_args()

    mem_df = load_csv(args.mem)
    disk_df = load_csv(args.disk)

    fig, (ax_mem, ax_disk) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # -------- Memory Plot --------
    mem_cols = [c for c in mem_df.columns if c.startswith("mem_")]
    if not mem_cols:
        # fallback: plot all numeric except timestamp/t_seconds
        mem_cols = [
            c for c in mem_df.columns
            if c not in ["timestamp", "t_seconds"]
        ]

    for c in mem_cols:
        ax_mem.plot(mem_df["t_seconds"], mem_df[c], label=c)

    ax_mem.set_title("Regular Node")
    ax_mem.set_ylabel("GB")
    ax_mem.set_xlabel("Time (seconds)")
    ax_mem.grid(True)
    ax_mem.legend()

    # -------- Disk Plot --------
    disk_cols = ["total_GB", "used_GB", "free_GB"]
    disk_cols = [c for c in disk_cols if c in disk_df.columns]

    if not disk_cols:
        disk_cols = [
            c for c in disk_df.columns
            if c not in ["timestamp", "t_seconds"]
        ]

    for c in disk_cols:
        ax_disk.plot(disk_df["t_seconds"], disk_df[c], label=c)

    ax_disk.set_title("Composable Memory Node")
    ax_disk.set_ylabel("GB")
    ax_disk.set_xlabel("Time (seconds)")
    ax_disk.grid(True)
    ax_disk.legend()

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()

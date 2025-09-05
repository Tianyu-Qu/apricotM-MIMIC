import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd


def pad_truncate(events: pd.DataFrame, max_len: int) -> np.ndarray:
    # events columns: sample_id, hours, var_id, value
    # We will output per sample: [max_len, 4] = [time, var_id, value, other(zeros)]
    X_list = []
    for sample_id, df in events.groupby("sample_id", sort=False):
        arr = np.zeros((max_len, 4), dtype=np.float32)
        seq = df[["hours", "var_id", "value"]].values
        if len(seq) <= max_len:
            arr[: len(seq), 0] = seq[:, 0]  # time
            arr[: len(seq), 1] = seq[:, 1]  # var id
            arr[: len(seq), 2] = seq[:, 2]  # value
        else:
            seq = seq[-max_len:, :]
            arr[:, 0] = seq[:, 0]
            arr[:, 1] = seq[:, 1]
            arr[:, 2] = seq[:, 2]
        X_list.append((sample_id, arr))
    return X_list


def process_split(in_split_dir: str, out_split_dir: str, max_len: int) -> None:
    os.makedirs(out_split_dir, exist_ok=True)
    events = pd.read_parquet(os.path.join(in_split_dir, "events.parquet"))
    events = events.sort_values(["sample_id", "hours"])  # ensure sorted
    packed = pad_truncate(events, max_len)

    # Save as npz with aligned order
    sample_ids = [sid for sid, _ in packed]
    X = np.stack([arr for _, arr in packed], axis=0)
    np.savez_compressed(os.path.join(out_split_dir, "seq.npz"), sample_ids=np.array(sample_ids), X=X)


def main():
    parser = argparse.ArgumentParser(description="Build fixed-length sequences for Mamba inputs.")
    parser.add_argument("--in_root", type=str, required=True, help="Input root from step 2")
    parser.add_argument("--out_root", type=str, required=True, help="Output root for step 3")
    parser.add_argument("--max_len", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    for split in os.listdir(args.in_root):
        split_in = os.path.join(args.in_root, split)
        if not os.path.isdir(split_in):
            continue
        process_split(split_in, os.path.join(args.out_root, split), args.max_len)


if __name__ == "__main__":
    main()



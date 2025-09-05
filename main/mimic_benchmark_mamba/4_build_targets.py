import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def build_targets(task: str, labels: pd.DataFrame) -> np.ndarray:
    # Generic mapping per common benchmark tasks
    task = task.lower()
    if task in ("in_hospital_mortality", "mortality"):
        # Expect a binary column, e.g., in_hospital_mortality or mortality
        for col in ["in_hospital_mortality", "mortality", "label"]:
            if col in labels.columns:
                y = labels[col].astype(int).values.reshape(-1, 1)
                return y
        raise ValueError("Mortality task requires a binary label column.")
    elif task in ("decompensation",):
        # Decomp can be per-timestep; here we build a per-sample label if present
        for col in ["decompensation", "label"]:
            if col in labels.columns:
                y = labels[col].astype(int).values.reshape(-1, 1)
                return y
        raise ValueError("Decompensation task requires a label column.")
    elif task in ("length_of_stay", "los"):
        for col in ["length_of_stay", "los", "remain_los"]:
            if col in labels.columns:
                y = labels[col].astype(float).values.reshape(-1, 1)
                return y
        raise ValueError("LOS task requires a numeric LOS column.")
    elif task in ("phenotyping",):
        # Multi-label columns; use all non-identifier columns as labels
        exclude = {"sample_id", "filename", "stay", "stay_id", "subject_id", "hadm_id", "icustay_id"}
        label_cols = [c for c in labels.columns if c not in exclude]
        if not label_cols:
            raise ValueError("Phenotyping requires multiple label columns.")
        y = labels[label_cols].astype(int).values
        return y
    else:
        raise ValueError(f"Unknown task: {task}")


def align_samples(seq_ids: np.ndarray, labels: pd.DataFrame) -> pd.DataFrame:
    # Align order of labels to seq sample_ids
    labels = labels.copy()
    labels = labels.set_index("sample_id").loc[seq_ids].reset_index()
    return labels


def process_split(task: str, in_split_dir: str, out_split_dir: str) -> None:
    os.makedirs(out_split_dir, exist_ok=True)
    labels = pd.read_parquet(os.path.join(in_split_dir, "labels.parquet"))
    seq = np.load(os.path.join(in_split_dir, "seq.npz"))
    sample_ids = seq["sample_ids"]
    labels = align_samples(sample_ids, labels)
    y = build_targets(task, labels)
    np.save(os.path.join(out_split_dir, "targets.npy"), y)


def main():
    parser = argparse.ArgumentParser(description="Build targets for each split.")
    parser.add_argument("--task", type=str, required=True, help="Task name: mortality|decompensation|los|phenotyping")
    parser.add_argument("--in_root", type=str, required=True, help="Input root from step 3")
    parser.add_argument("--out_root", type=str, required=True, help="Output root for step 4")
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    for split in os.listdir(args.in_root):
        split_in = os.path.join(args.in_root, split)
        if not os.path.isdir(split_in):
            continue
        process_split(args.task, split_in, os.path.join(args.out_root, split))


if __name__ == "__main__":
    main()



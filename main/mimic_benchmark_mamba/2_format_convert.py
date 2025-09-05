import argparse
import json
import os
from typing import Dict, List

import pandas as pd


def load_variable_vocab(clean_root: str) -> List[str]:
    variables = set()
    for split in os.listdir(os.path.join(clean_root)):
        split_dir = os.path.join(clean_root, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if not fname.endswith(".parquet") or fname == "labels.parquet":
                continue
            df = pd.read_parquet(os.path.join(split_dir, fname), columns=["variable"])
            variables.update(df["variable"].unique().tolist())
    vocab = sorted(list(variables))
    return vocab


def build_var_to_id(vocab: List[str]) -> Dict[str, int]:
    # Reserve 0 for padding as required by the Mamba-ready pipeline
    return {v: i + 1 for i, v in enumerate(vocab)}


def convert_split(clean_split_dir: str, out_split_dir: str, var_to_id: Dict[str, int]) -> None:
    os.makedirs(out_split_dir, exist_ok=True)
    labels = pd.read_parquet(os.path.join(clean_split_dir, "labels.parquet"))
    labels.to_parquet(os.path.join(out_split_dir, "labels.parquet"), index=False)

    records = []
    sample_ids = []
    for fname in os.listdir(clean_split_dir):
        if not fname.endswith(".parquet") or fname == "labels.parquet":
            continue
        df = pd.read_parquet(os.path.join(clean_split_dir, fname))
        # Expect columns: sample_id, hours, variable, value
        if not {"sample_id", "hours", "variable", "value"}.issubset(df.columns):
            raise ValueError("Expected columns sample_id, hours, variable, value in long parquet.")
        df = df.sort_values(["sample_id", "hours"])  # ensure sorted by time
        df["var_id"] = df["variable"].map(var_to_id).astype("Int64")
        df = df.dropna(subset=["var_id"])  # drop variables not in vocab (shouldn't happen)
        df["var_id"] = df["var_id"].astype(int)
        df = df[["sample_id", "hours", "var_id", "value"]]
        records.append(df)
        sample_ids.extend(df["sample_id"].unique().tolist())

    if not records:
        raise ValueError(f"No timeseries parquet files found in {clean_split_dir}")

    all_events = pd.concat(records, axis=0, ignore_index=True)
    all_events.to_parquet(os.path.join(out_split_dir, "events.parquet"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Map variables to ids and normalize events for Mamba.")
    parser.add_argument("--in_root", type=str, required=True, help="Input root from step 1 (clean)")
    parser.add_argument("--out_root", type=str, required=True, help="Output root for step 2")
    args = parser.parse_args()

    clean_root = os.path.join(args.in_root, "clean")
    if not os.path.isdir(clean_root):
        raise ValueError("Input root must contain a 'clean' directory from step 1.")

    vocab = load_variable_vocab(clean_root)
    var_to_id = build_var_to_id(vocab)

    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "var_to_id.json"), "w") as f:
        json.dump(var_to_id, f)

    for split in os.listdir(clean_root):
        split_in = os.path.join(clean_root, split)
        if not os.path.isdir(split_in):
            continue
        split_out = os.path.join(args.out_root, split)
        convert_split(split_in, split_out, var_to_id)


if __name__ == "__main__":
    main()



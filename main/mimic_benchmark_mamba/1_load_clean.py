import argparse
import os
from typing import List, Dict, Tuple

import pandas as pd


def _detect_time_column(df: pd.DataFrame) -> str:
    for cand in ["Hours", "hours", "Time", "time", "Minutes", "minutes"]:
        if cand in df.columns:
            return cand
    raise ValueError("No time column found. Expected one of Hours/Time/Minutes.")


def _to_hours(df: pd.DataFrame, time_col: str) -> pd.Series:
    s = df[time_col]
    if "min" in time_col.lower():
        return s.astype(float) / 60.0
    return s.astype(float)


def _melt_long(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    value_cols = [c for c in df.columns if c != time_col]
    long_df = df.melt(id_vars=[time_col], value_vars=value_cols, var_name="variable", value_name="value")
    # Drop rows with all-missing/empty values
    long_df = long_df.dropna(subset=["value"]).copy()
    # Coerce to numeric when possible
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])  # keep only numeric values
    long_df = long_df.rename(columns={time_col: "hours"})
    return long_df


def _read_listfile(path: str) -> pd.DataFrame:
    lf = pd.read_csv(path)
    # Normalize column names
    lf.columns = [c.strip().lower() for c in lf.columns]
    return lf


def _find_splits(task_root: str) -> List[str]:
    # Common splits in mimic3-benchmarks
    candidates = ["train", "train", "train", "val", "validation", "test"]
    found = []
    for name in ["train", "val", "validation", "test"]:
        if os.path.exists(os.path.join(task_root, name, "listfile.csv")):
            found.append(name)
    # Ensure unique and stable order: train, val/validation, test
    order = []
    for name in ["train", "val", "validation", "test"]:
        if name in found and name not in order:
            order.append(name)
    return order


def _extract_labels_columns(lf: pd.DataFrame) -> List[str]:
    # Exclude filename-like columns
    exclude = {"stay", "stay_id", "subject_id", "hadm_id", "icustay_id", "filename", "time", "hours"}
    label_cols = [c for c in lf.columns if c not in exclude]
    return label_cols


def process_split(task_root: str, split: str, out_root: str) -> None:
    split_dir = os.path.join(task_root, split)
    listfile = os.path.join(split_dir, "listfile.csv")
    lf = _read_listfile(listfile)

    # Identify columns
    if "filename" not in lf.columns:
        # Some variants use 'ts_filename' or similar
        cand = [c for c in lf.columns if "file" in c]
        if not cand:
            raise ValueError("listfile.csv must contain a filename column.")
        lf = lf.rename(columns={cand[0]: "filename"})

    label_cols = _extract_labels_columns(lf)
    if not label_cols:
        raise ValueError("No label columns detected in listfile.csv.")

    # Prepare out dirs
    out_split_dir = os.path.join(out_root, "clean", split)
    os.makedirs(out_split_dir, exist_ok=True)

    labels_records = []

    for idx, row in lf.iterrows():
        ts_path = os.path.join(split_dir, row["filename"])  # timeseries file path
        if not os.path.exists(ts_path):
            # Some datasets store timeseries in a subdir (e.g., train/ or test/)
            alt = os.path.join(split_dir, os.path.basename(row["filename"]))
            if os.path.exists(alt):
                ts_path = alt
            else:
                raise FileNotFoundError(f"Timeseries file not found: {ts_path}")

        ts = pd.read_csv(ts_path)
        time_col = _detect_time_column(ts)
        ts = ts.sort_values(by=time_col)
        ts["hours"] = _to_hours(ts, time_col)

        # Melt only non-time, non-hours columns against the time column
        cols_to_melt = [c for c in ts.columns if c not in {time_col, "hours"}]
        wide = pd.concat([ts[[time_col]], ts[cols_to_melt]], axis=1)
        long_df = _melt_long(wide)
        # Ensure hours is present after melt
        if "hours" not in long_df.columns:
            long_df["hours"] = _to_hours(ts, time_col)

        # Normalize variable names as strings
        long_df["variable"] = long_df["variable"].astype(str)
        long_df = long_df[["hours", "variable", "value"]]

        # Assign a sample_id = row index within split
        sample_id = f"{split}_{idx}"
        long_df.insert(0, "sample_id", sample_id)

        # Save long table for this sample
        out_file = os.path.join(out_split_dir, f"{sample_id}.parquet")
        long_df.to_parquet(out_file, index=False)

        # Capture labels for this sample
        labels_records.append({"sample_id": sample_id, **{c: row[c] for c in label_cols}})

    # Save labels table for the split
    labels_df = pd.DataFrame(labels_records)
    labels_df.to_parquet(os.path.join(out_split_dir, "labels.parquet"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Load and clean MIMIC benchmark data into long format.")
    parser.add_argument("--task_root", type=str, required=True, help="Path to task root (e.g., mimic3-benchmarks/decompensation/)")
    parser.add_argument("--out_root", type=str, required=True, help="Output root for intermediate artifacts")
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    splits = _find_splits(args.task_root)
    if not splits:
        raise ValueError("No splits found. Expected e.g., train/, val/ or validation/, test/ with listfile.csv.")

    for split in splits:
        process_split(args.task_root, split, args.out_root)


if __name__ == "__main__":
    main()



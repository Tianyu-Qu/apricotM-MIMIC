import argparse
import os

import h5py
import numpy as np


def write_group(h5: h5py.File, group: str, X: np.ndarray, y: np.ndarray, static: np.ndarray = None) -> None:
    g = h5.require_group(group)
    g.create_dataset("X", data=X, compression="gzip")
    g.create_dataset("y", data=y, compression="gzip")
    if static is not None:
        g.create_dataset("static", data=static, compression="gzip")


def process_split(in_split_dir: str) -> tuple:
    seq = np.load(os.path.join(in_split_dir, "seq.npz"))
    X = seq["X"]
    y = np.load(os.path.join(in_split_dir, "targets.npy"))
    # Optional static features: if present, include
    static_path = os.path.join(in_split_dir, "static.npy")
    static = np.load(static_path) if os.path.exists(static_path) else None
    return X, y, static


def main():
    parser = argparse.ArgumentParser(description="Export splits to Mamba-ready HDF5.")
    parser.add_argument("--in_root", type=str, required=True, help="Input root from step 4")
    parser.add_argument("--out_file", type=str, required=True, help="Output HDF5 filepath")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with h5py.File(args.out_file, "w") as h5:
        for split in os.listdir(args.in_root):
            split_in = os.path.join(args.in_root, split)
            if not os.path.isdir(split_in):
                continue
            X, y, static = process_split(split_in)
            write_group(h5, split, X, y, static)


if __name__ == "__main__":
    main()



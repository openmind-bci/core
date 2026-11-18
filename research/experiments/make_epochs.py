"""Build preprocessed epochs for a few subjects and cache them.

    uv run python research/experiments/make_epochs.py --dataset physionetmi --subjects 1 2 3

Epochs are saved to $OPENMIND_DATA_DIR/cache/epochs/<dataset>_s<subjects>.npz.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from openmind_core.datasets import DATASETS, load_epochs
from openmind_core.paths import data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="physionetmi")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    start = time.perf_counter()
    epochs = load_epochs(args.dataset, args.subjects)
    elapsed = time.perf_counter() - start

    print(f"{args.dataset}: {len(epochs)} epochs, shape {epochs.X.shape}, {elapsed:.1f} s")
    print(f"channels: {', '.join(epochs.channels)} @ {epochs.srate:g} Hz")
    print(f"rejected (> 150 uV peak to peak): {epochs.info['rejected']}")
    for subject in epochs.subjects:
        subset = epochs.select(epochs.subject == subject)
        ptp = float(np.median(np.ptp(subset.X, axis=2)))
        print(f"  subject {subject}: {subset.counts()}  median ptp {ptp:.0f} uV")

    name = f"{args.dataset}_s{'-'.join(map(str, args.subjects))}.npz"
    path = epochs.save(data_dir() / "cache" / "epochs" / name)
    print(f"saved to {path}")


if __name__ == "__main__":
    main()

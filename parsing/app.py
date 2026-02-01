import sys
from pathlib import Path

import numpy as np
import pandas as pd

from handler import (
    CleanDataHandler,
    DropUselessDataHandler, DataFrameHandler, DropEmptySalaryHandler,
)

from feature import build_xy


def save_as_npy(csv_path: str, x: np.ndarray, y: np.ndarray) -> None:
    base_dir = Path(csv_path).parent
    np.save(str(base_dir / "x_data.npy"), x)
    np.save(str(base_dir / "y_data.npy"), y)


def build_chain_handler() -> DataFrameHandler:
    return (
        CleanDataHandler()
        .set_next_handler(DropUselessDataHandler())
        .set_next_handler(DropEmptySalaryHandler())
    )


def run_pipeline(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    handler_chain = build_chain_handler()

    df = handler_chain.handle(df)

    x, y = build_xy(df)
    save_as_npy(csv_path, x, y)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python app.py path/to/hh.csv")

    csv_path = sys.argv[1]
    run_pipeline(csv_path)

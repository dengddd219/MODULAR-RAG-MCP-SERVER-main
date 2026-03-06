"""Offline training script for Layer 1 spam gateway classifier.

This script trains a lightweight binary classifier to distinguish between
valid queries and "spam"/low-value traffic, and serializes the trained
pipeline to ``data/models/spam_gateway.pkl``.

Design goals (from DEV_SPEC D0-a):
- Local, offline training (no external APIs)
- Very fast inference at runtime
- Simple, reproducible training procedure

Default training data format
----------------------------
The script expects a CSV file with at least two columns:

- ``text``: the raw user query (string)
- ``label``: binary label, one of {0, 1} where:
  - 0 → non-spam / keep
  - 1 → spam / to be short‑circuited

You can change the column names via CLI arguments.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_OUTPUT_PATH = Path("data/models/spam_gateway.pkl")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight spam gateway classifier and save to data/models. "
            "Supports CSV and Excel (.xlsx/.xls) input files."
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data with 'text' and 'label' columns (CSV or Excel).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="File encoding for the CSV (default: utf-8, e.g. use 'gb18030' on Windows if needed).",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the text column in the CSV file (default: text).",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Name of the label column in the CSV file (default: label).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use as validation set (default: 0.2).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to save the trained model pipeline (default: data/models/spam_gateway.pkl).",
    )
    return parser.parse_args()


def load_dataset(
    path: Path,
    text_column: str,
    label_column: str,
    encoding: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load training dataset from a CSV or Excel file.

    Args:
        path: Path to the data file.
        text_column: Name of the text column.
        label_column: Name of the label column.
        encoding: Encoding for CSV files (ignored for Excel).

    Returns:
        Tuple of (texts, labels).
    """
    if not path.exists():
        raise FileNotFoundError(f"Training data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(path, encoding=encoding)
    elif suffix in {".xlsx", ".xls"}:
        # Excel files: let pandas choose an appropriate engine (e.g. openpyxl)
        try:
            df = pd.read_excel(path)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Reading Excel files requires an extra dependency (e.g. 'openpyxl'). "
                "Install it with 'pip install openpyxl' or convert the file to CSV."
            ) from exc
    else:
        raise ValueError(
            f"Unsupported file type for training data: {suffix!r}. "
            "Supported: .csv, .txt, .xlsx, .xls"
        )

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV (available: {list(df.columns)})")
    if label_column not in df.columns:
        raise ValueError(
            f"Column '{label_column}' not found in CSV (available: {list(df.columns)})"
        )

    texts = df[text_column].astype(str).values
    labels = df[label_column].astype(int).values

    return texts, labels


def build_pipeline() -> Pipeline:
    """Build the spam classification pipeline (Optimized for small data)."""
    # 改为字符级提取，并降低词频门槛，榨干小样本的特征
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',   # 按字符片段拆分，专治各种乱码和变体垃圾词
        ngram_range=(2, 5),   # 提取2到5个字符的组合
        min_df=1,             # 保留哪怕只出现过1次的特征！
        max_df=0.95,
        sublinear_tf=True,
    )
    classifier = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        C=2.0  # 稍微增加正则化强度，防止在小数据上死记硬背
    )
    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("clf", classifier),
        ]
    )


def main() -> int:
    """Entry point for training the spam gateway model."""
    args = parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)

    print(f"[spam-train] Loading data from: {data_path}")
    X, y = load_dataset(
        data_path,
        args.text_column,
        args.label_column,
        args.encoding,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    print(f"[spam-train] Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("[spam-train] Evaluating on validation set...")
    y_pred = pipeline.predict(X_val)
    report = classification_report(y_val, y_pred, digits=4)
    print(report)

    # Ensure output directory exists
    output_dir = output_path.parent
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(pipeline, output_path)
    print(f"[spam-train] Model saved to: {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



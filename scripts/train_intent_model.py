"""Layer 2 细粒度意图分类器离线训练脚本。intend mode

此脚本训练一个 TF-IDF + XGBoost（或备选 RandomForest）的多分类意图分类器，
并将训练好的流水线序列化保存到 ``data/models/intent_classifier.pkl``。

设计目标 (来自 DEV_SPEC D0-b):
- 本地离线训练
- 利用查询处理器生成的干净特征（短文本查询）
- 支持多种业务意图（如：退货、面料护理、搭配建议、投诉升级、闲聊）

默认训练数据格式
----------------------------
脚本期望一个至少包含两列的 CSV 文件：
- ``text``: 原始用户查询或特征文本（字符串）
- ``intent``: 类别标签（字符串），例如 "returns", "fabric_care" ...

标签列将使用 scikit-learn 的 LabelEncoder 进行编码，
映射关系会存储在流水线中，供运行时路由解析使用。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


DEFAULT_OUTPUT_PATH = Path("data/models/intent_classifier.pkl")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "训练一个 TF-IDF + XGBoost 意图分类器并保存到 data/models。 "
            "支持 CSV 和 Excel (.xlsx/.xls) 输入文件。"
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="包含 'text' 和 'intent' 列的训练数据路径 (CSV 或 Excel)。",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="CSV 文件中正文列的名称 (默认: text)。",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="intent",
        help="CSV 文件中意图标签列的名称 (默认: intent)。",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="CSV 文件的编码格式 (默认: utf-8; Excel 文件会自动忽略此参数)。",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="用作验证集的数据比例 (默认: 0.2)。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="训练好的模型流水线保存位置 (默认: data/models/intent_classifier.pkl)。",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["xgboost", "logistic"],
        default="xgboost",
        help="分类器类型：xgboost（默认）或 logistic。",
    )
    return parser.parse_args()


def load_dataset(
    path: Path,
    text_column: str,
    label_column: str,
    encoding: str,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """从 CSV 或 Excel 加载数据集并对标签进行编码。

    在加载阶段会自动丢弃以下无效样本：
    - 文本为空或为 NaN
    - 标签为 NaN / 空字符串 / 字面量 "nan" / "None"

    Returns:
        包含 (文本数组, 编码后的标签数组, 标签编码器实例) 的元组。
    """
    if not path.exists():
        raise FileNotFoundError(f"未找到训练数据文件: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(path, encoding=encoding)
    elif suffix in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(path)
        except ImportError as exc:
            raise RuntimeError(
                "读取 Excel 文件需要额外依赖 (如 'openpyxl')。 "
                "请运行 'pip install openpyxl' 或将文件转换为 CSV。"
            ) from exc
    else:
        raise ValueError(
            f"不支持的文件类型: {suffix!r}。 仅支持: .csv, .txt, .xlsx, .xls"
        )

    # 验证列名是否存在
    if text_column not in df.columns:
        raise ValueError(f"列 '{text_column}' 在数据集中不存在 (现有列: {list(df.columns)})")
    if label_column not in df.columns:
        raise ValueError(f"列 '{label_column}' 在数据集中不存在 (现有列: {list(df.columns)})")

    # 先转为字符串，便于统一清洗无效值
    df[text_column] = df[text_column].astype(str)
    df[label_column] = df[label_column].astype(str)

    # 丢弃无效标签行：NaN / 空字符串 / "nan" / "none"
    invalid_label_mask = df[label_column].str.strip().str.lower().isin(
        {"", "nan", "none"}
    )
    # 丢弃无效文本行：NaN / 空字符串
    invalid_text_mask = df[text_column].str.strip().eq("")

    before_count = len(df)
    df = df[~(invalid_label_mask | invalid_text_mask)].copy()
    dropped = before_count - len(df)
    if dropped > 0:
        print(
            f"[intent-train] 丢弃了 {dropped} 条无效样本 "
            f"(空文本 / 无效标签: {label_column})."
        )

    texts = df[text_column].values

    unique_labels, counts = np.unique(df[label_column].astype(str).values, return_counts=True)
    print("[intent-train] 标签分布:")
    for label, count in zip(unique_labels, counts):
        print(f"  - {label}: {count} ({count / len(df) * 100:.1f}%)")

    # 将分类标签（如 "returns"）转换为数字（如 0, 1, 2...）
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[label_column].astype(str).values)

    return texts, labels, label_encoder


def build_pipeline(label_encoder: LabelEncoder, classifier_type: str = "xgboost") -> Pipeline:
    """构建意图分类流水线。

    流水线包含：
    - TfidfVectorizer: 提取词级特征，非常适合短查询文本。
    - XGBClassifier / LogisticRegression: 多分类器。
    """
    # 文本转向量：短文本意图分类对 unigram+bigram 往往更稳健。
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    if classifier_type == "xgboost":
        classifier = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            max_depth=4,
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.5,
            n_jobs=4,
        )
    else:
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
        )

    # 使用 ColumnTransformer 专门处理指定的文本列
    text_transformer = ColumnTransformer(
        transformers=[("text", vectorizer, 0)],
        remainder="drop",
    )

    # 将特征工程和分类器打包
    pipeline = Pipeline(
        steps=[
            ("features", text_transformer),
            ("clf", classifier),
        ]
    )

    # 将标签编码器挂载到流水线上，方便预测后能还原回中文/英文标签名
    pipeline.label_encoder = label_encoder  # type: ignore[attr-defined]
    return pipeline


def main() -> int:
    """意图分类模型训练入口。"""
    args = parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)

    print(f"[intent-train] 正在加载数据: {data_path}")
    X, y, label_encoder = load_dataset(
        data_path,
        args.text_column,
        args.label_column,
        args.encoding,
    )

    # 划分训练集和验证集，stratify=y 确保各类意图在两个集合中比例一致
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    print(f"[intent-train] 训练样本数: {len(X_train)}, 验证样本数: {len(X_val)}")

    # 构建并训练流水线
    pipeline = build_pipeline(label_encoder, args.classifier)
    # 因为 ColumnTransformer 期望输入是二维数组，所以需要 reshape
    fit_kwargs = {}
    if args.classifier == "xgboost":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        fit_kwargs["clf__sample_weight"] = sample_weight
    pipeline.fit(X_train.reshape(-1, 1), y_train, **fit_kwargs)

    print("[intent-train] 正在验证集上评估模型...")
    y_pred = pipeline.predict(X_val.reshape(-1, 1))
    
    # 生成详细的评估报告（精确率、召回率、F1等）
    report = classification_report(
        y_val,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
    )
    print(report)
    print(f"[intent-train] classifier={args.classifier}")

    # 确保输出目录存在
    output_dir = output_path.parent
    os.makedirs(output_dir, exist_ok=True)

    # 保存模型文件
    joblib.dump(pipeline, output_path)
    print(f"[intent-train] 模型已保存至: {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
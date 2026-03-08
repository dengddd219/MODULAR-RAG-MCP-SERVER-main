"""Query Complexity Classifier 训练脚本。

此脚本训练一个二分类模型（简单询问 vs 复杂询问），
并将训练好的流水线序列化保存到 ``data/models/query_complexity_classifier.pkl``。

设计目标:
- 本地离线训练
- 二分类任务：simple（简单）vs complex（复杂）
- 使用 TF-IDF + XGBoost 或 LogisticRegression
- 快速推理，适合实时路由决策

默认训练数据格式
----------------------------
脚本期望一个至少包含两列的 CSV 文件：
- ``text``: 原始用户查询（字符串）
- ``complexity``: 类别标签（字符串），取值为 "simple" 或 "complex"
  或者使用 "label" 列，取值为 0（简单）或 1（复杂）

标签列将使用 scikit-learn 的 LabelEncoder 进行编码，
映射关系会存储在流水线中，供运行时预测使用。
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


DEFAULT_OUTPUT_PATH = Path("data/models/query_complexity_classifier.pkl")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="训练查询复杂度分类器（简单 vs 复杂）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data/training/query_complexity_dataset.csv",
        help="训练数据 CSV 文件路径（默认: data/training/query_complexity_dataset.csv）",
    )
    
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="文本列名（默认: text）",
    )
    
    parser.add_argument(
        "--label-column",
        type=str,
        default="complexity",
        help="标签列名（默认: complexity，可选: label）",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"训练好的模型保存位置（默认: {DEFAULT_OUTPUT_PATH}）。",
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="验证集比例（默认: 0.2）",
    )
    
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="CSV 文件编码（默认: utf-8）",
    )
    
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["xgboost", "logistic"],
        default="xgboost",
        help="分类器类型：xgboost（默认）或 logistic",
    )
    
    return parser.parse_args()


def load_dataset(
    path: Path,
    text_column: str,
    label_column: str,
    encoding: str,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """加载训练数据集。
    
    Args:
        path: CSV 文件路径
        text_column: 文本列名
        label_column: 标签列名
        encoding: 文件编码
        
    Returns:
        (texts, labels, label_encoder) 元组
    """
    if not path.exists():
        raise FileNotFoundError(f"训练数据文件不存在: {path}")
    
    df = pd.read_csv(path, encoding=encoding)
    
    if text_column not in df.columns:
        raise ValueError(f"文本列 '{text_column}' 不存在。可用列: {df.columns.tolist()}")
    
    if label_column not in df.columns:
        raise ValueError(f"标签列 '{label_column}' 不存在。可用列: {df.columns.tolist()}")
    
    # 提取文本和标签
    texts = df[text_column].fillna("").astype(str).values
    raw_labels = df[label_column].fillna("").astype(str).values
    
    # 处理标签：支持 "simple"/"complex" 或 0/1
    normalized_labels = []
    for label in raw_labels:
        label_str = str(label).strip().lower()
        if label_str in ["simple", "简单", "0"]:
            normalized_labels.append("simple")
        elif label_str in ["complex", "复杂", "1"]:
            normalized_labels.append("complex")
        else:
            # 尝试转换为数字
            try:
                label_num = int(label_str)
                normalized_labels.append("simple" if label_num == 0 else "complex")
            except ValueError:
                raise ValueError(f"无效的标签值: {label}。期望: simple/complex 或 0/1")
    
    # 编码标签
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(normalized_labels)
    
    print(f"[complexity-train] 加载了 {len(texts)} 个样本")
    print(f"[complexity-train] 标签分布:")
    unique, counts = np.unique(normalized_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  - {label}: {count} ({count/len(texts)*100:.1f}%)")
    
    return texts, labels, label_encoder


def build_pipeline(label_encoder: LabelEncoder, classifier_type: str = "xgboost") -> Pipeline:
    """构建查询复杂度分类流水线。
    
    流水线包含：
    - TfidfVectorizer: 提取词级特征
    - XGBClassifier 或 LogisticRegression: 二分类器
    
    Args:
        label_encoder: 标签编码器
        classifier_type: 分类器类型（"xgboost" 或 "logistic"）
    """
    # 文本转向量
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # 使用 1-gram 和 2-gram
        min_df=2,            # 至少在 2 个样本中出现
        max_df=0.95,         # 过滤掉超过 95% 文档的高频词
        sublinear_tf=True,   # 使用 sublinear TF 缩放
    )
    
    # 选择分类器
    if classifier_type == "xgboost":
        classifier = XGBClassifier(
            objective="binary:logistic",  # 二分类任务
            eval_metric="logloss",
            max_depth=5,
            n_estimators=200,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            n_jobs=4,
        )
    else:  # logistic
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(
            max_iter=1000,
            C=1.0,
            penalty="l2",
            solver="lbfgs",
        )
    
    # 使用 ColumnTransformer 处理文本列
    text_transformer = ColumnTransformer(
        transformers=[("text", vectorizer, 0)],
        remainder="drop",
    )
    
    # 构建流水线
    pipeline = Pipeline(
        steps=[
            ("features", text_transformer),
            ("clf", classifier),
        ]
    )
    
    # 将标签编码器挂载到流水线上
    pipeline.label_encoder = label_encoder  # type: ignore[attr-defined]
    
    return pipeline


def main() -> int:
    """查询复杂度分类模型训练入口。"""
    args = parse_args()
    
    data_path = Path(args.data)
    output_path = Path(args.output)
    
    print(f"[complexity-train] 正在加载数据: {data_path}")
    X, y, label_encoder = load_dataset(
        data_path,
        args.text_column,
        args.label_column,
        args.encoding,
    )
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )
    
    print(f"[complexity-train] 训练样本数: {len(X_train)}, 验证样本数: {len(X_val)}")
    
    # 构建并训练流水线
    print(f"[complexity-train] 使用 {args.classifier} 分类器")
    pipeline = build_pipeline(label_encoder, args.classifier)
    pipeline.fit(X_train.reshape(-1, 1), y_train)
    
    print("[complexity-train] 正在验证集上评估模型...")
    y_pred = pipeline.predict(X_val.reshape(-1, 1))
    
    # 生成详细的评估报告
    report = classification_report(
        y_val,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
    )
    print(report)
    
    # 显示混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    print("\n[complexity-train] 混淆矩阵:")
    print(f"               预测")
    print(f"            Simple  Complex")
    print(f"实际 Simple    {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"     Complex   {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    # 确保输出目录存在
    output_dir = output_path.parent
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型文件
    joblib.dump(pipeline, output_path)
    print(f"\n[complexity-train] 模型已保存至: {output_path.resolve()}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


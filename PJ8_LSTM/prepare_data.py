# -*- coding: utf-8 -*-
"""
数据准备与预处理脚本（工程化改造版）
- 参照 wanzheng.py 结构，支持 CONFIG 配置、三切分、缓存复用、元数据记录
- 输出：train.csv / val.csv / test.csv / vocab.json / metadata.json
"""

import os
import json
import random
from collections import Counter
from typing import Any, Dict, List, Tuple

import jieba
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================== CONFIG ==============================
# 所有可调整参数集中在这里
CONFIG: Dict[str, Any] = {
    # 数据相关
    "dataset_name": "XiangPan/waimai_10k",
    "hf_cache_dir": "./hf_datasets_cache",   # HuggingFace 本地缓存目录
    "processed_data_dir": "processed_data",  # 输出目录
    "force_prepare": False,  # True=强制重新生成；False=文件存在则直接复用
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "min_freq": 2,
    "label_names": ["负向", "正向"],  # 与标签 0/1 对应
    "seed": 42,
}
# ====================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)


def dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    自动探测文本列与标签列，统一重命名为 text / label。
    """
    text_candidates = ["review", "text", "content", "sentence", "review_content"]
    text_col = None
    for col in text_candidates:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        for col in df.columns:
            if col != "label" and df[col].dtype == object:
                text_col = col
                break
    if text_col is None:
        raise ValueError(f"未找到文本列，当前列: {list(df.columns)}")

    df = df.rename(columns={text_col: "text"})
    if "label" not in df.columns:
        label_candidates = [
            col for col in df.columns if "label" in col.lower() or col == "labels"
        ]
        if not label_candidates:
            raise ValueError(f"未找到标签列，当前列: {list(df.columns)}")
        df = df.rename(columns={label_candidates[0]: "label"})

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].dropna(subset=["text", "label"])
    return df.reset_index(drop=True)


def normalize_labels(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    将原始标签统一为 0/1，保留 raw_label 列，并生成 metadata。
    """
    raw_labels = list(pd.unique(df["label"]))
    if set(raw_labels).issubset({0, 1}):
        label_map = {0: 0, 1: 1}
    elif set(raw_labels).issubset({1, 2}):
        label_map = {1: 0, 2: 1}
    else:
        sorted_labels = sorted(raw_labels)
        label_map = {label: idx for idx, label in enumerate(sorted_labels)}

    df = df.copy()
    df["raw_label"] = df["label"]
    df["label"] = df["label"].map(label_map).astype(int)

    num_classes = int(df["label"].nunique())
    label_names = cfg["label_names"]
    if len(label_names) != num_classes:
        label_names = [str(i) for i in range(num_classes)]

    meta = {
        "label_map": {str(k): int(v) for k, v in label_map.items()},
        "label_names": label_names,
        "num_classes": num_classes,
        "label_distribution": {
            str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()
        },
    }
    return df, meta


def download_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    从 HuggingFace 下载，合并所有 split，并保留 source_split 来源标记。
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请先安装: pip install datasets")

    print(f"正在加载数据集: {cfg['dataset_name']}")
    ds = load_dataset(cfg["dataset_name"], cache_dir=cfg["hf_cache_dir"])
    frames = []
    for split_name in ds.keys():
        split_df = ds[split_name].to_pandas()
        split_df["source_split"] = split_name
        frames.append(split_df)
    df = pd.concat(frames, ignore_index=True)
    df = normalize_columns(df)
    print(f"原始数据读取完成，共 {len(df)} 条")
    return df


def split_train_val_test(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按比例 7:1.5:1.5 分层划分为 train / val / test。
    """
    ratio_sum = cfg["train_ratio"] + cfg["val_ratio"] + cfg["test_ratio"]
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1")

    train_val_df, test_df = train_test_split(
        df,
        test_size=cfg["test_ratio"],
        random_state=cfg["seed"],
        stratify=df["label"],
    )
    val_size_in_train_val = cfg["val_ratio"] / (cfg["train_ratio"] + cfg["val_ratio"])
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_in_train_val,
        random_state=cfg["seed"],
        stratify=train_val_df["label"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_vocab(train_df: pd.DataFrame, min_freq: int) -> Dict[str, int]:
    """
    仅基于训练集分词、统计词频、过滤低频词，构建 word2id。
    """
    print("正在使用 jieba 对训练集分词并统计词频...")
    all_tokens: List[str] = []
    for text in train_df["text"].astype(str):
        all_tokens.extend(jieba.lcut(text.strip()))

    word_freq = Counter(token for token in all_tokens if token and token.strip())
    vocab_words = [word for word, count in word_freq.items() if count >= min_freq]
    vocab_words = sorted(vocab_words, key=lambda x: (-word_freq[x], x))

    word2id = {"<PAD>": 0, "<UNK>": 1}
    for idx, word in enumerate(vocab_words, start=2):
        word2id[word] = idx
    return word2id


def processed_data_is_complete(paths: Dict[str, str]) -> bool:
    """
    已处理数据完整时直接复用，不下载、不划分、不重建词表。
    """
    required_files = list(paths.values())
    if not all(os.path.isfile(path) for path in required_files):
        return False

    try:
        for split_name in ("train_csv", "val_csv", "test_csv"):
            df = pd.read_csv(paths[split_name], nrows=1)
            if not {"text", "label"}.issubset(df.columns):
                return False

        word2id = read_json(paths["vocab_json"])
        metadata = read_json(paths["metadata_json"])
        if "<PAD>" not in word2id or "<UNK>" not in word2id:
            return False
        required_meta_keys = {"train_size", "val_size", "test_size", "vocab_size", "num_classes"}
        if not required_meta_keys.issubset(metadata.keys()):
            return False
    except Exception:
        return False

    return True


def prepare_data(cfg: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    """
    主控函数：
      1. 检查缓存文件是否存在，存在则直接复用；
      2. 否则下载 → 清洗 → 划分 → 构建词表 → 保存全部产物。
    返回字典，包含 paths / word2id / metadata，可直接被训练脚本消费。
    """
    data_dir = os.path.join(base_dir, cfg["processed_data_dir"])
    os.makedirs(data_dir, exist_ok=True)

    paths = {
        "train_csv": os.path.join(data_dir, "train.csv"),
        "val_csv": os.path.join(data_dir, "val.csv"),
        "test_csv": os.path.join(data_dir, "test.csv"),
        "vocab_json": os.path.join(data_dir, "vocab.json"),
        "metadata_json": os.path.join(data_dir, "metadata.json"),
    }

    # 缓存复用逻辑：processed_data 完整时直接跳过准备流程。
    if not cfg["force_prepare"] and processed_data_is_complete(paths):
        print(f"检测到已有处理后数据，直接复用: {data_dir}")
        word2id = read_json(paths["vocab_json"])
        metadata = read_json(paths["metadata_json"])
        return {"paths": paths, "word2id": word2id, "metadata": metadata}

    # 下载与清洗
    df = download_dataset(cfg)
    df, metadata = normalize_labels(df, cfg)

    # 划分
    train_df, val_df, test_df = split_train_val_test(df, cfg)

    # 构建词表（仅基于训练集，防止泄漏）
    word2id = build_vocab(train_df, cfg["min_freq"])

    # 保存 CSV
    train_df.to_csv(paths["train_csv"], index=False, encoding="utf-8")
    val_df.to_csv(paths["val_csv"], index=False, encoding="utf-8")
    test_df.to_csv(paths["test_csv"], index=False, encoding="utf-8")
    dump_json(paths["vocab_json"], word2id)

    # 更新并保存元数据
    metadata.update({
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "vocab_size": int(len(word2id)),
        "min_freq": cfg["min_freq"],
        "train_ratio": cfg["train_ratio"],
        "val_ratio": cfg["val_ratio"],
        "test_ratio": cfg["test_ratio"],
    })
    dump_json(paths["metadata_json"], metadata)

    print(
        f"数据准备完成: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
        f"vocab={len(word2id)}"
    )
    return {"paths": paths, "word2id": word2id, "metadata": metadata}


def main() -> None:
    set_seed(CONFIG["seed"])
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    data_info = prepare_data(CONFIG, base_dir)

    # 打印摘要
    meta = data_info["metadata"]
    print("\n===== 数据准备结果 =====")
    print(f"训练集: {meta['train_size']} 条")
    print(f"验证集: {meta['val_size']} 条")
    print(f"测试集: {meta['test_size']} 条")
    print(f"词表大小: {meta['vocab_size']}")
    print(f"标签映射: {meta['label_map']}")
    print(f"标签名称: {meta['label_names']}")
    print(f"标签分布: {meta['label_distribution']}")
    print("========================")


if __name__ == "__main__":
    main()

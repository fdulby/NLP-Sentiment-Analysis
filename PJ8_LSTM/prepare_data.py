# -*- coding: utf-8 -*-
"""
数据准备与预处理脚本
- 从网络下载 waimai_10k 中文外卖评价情感数据集
- 8:2 划分训练集与测试集，保存为 train.csv / test.csv（列：label, text）
- 使用 jieba 分词构建词表，保存为 vocab.json（含 <PAD>、<UNK>）
"""

import os
import json
import jieba
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# 使用 HuggingFace datasets 下载 waimai_10k
try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("请先安装: pip install datasets")


# ========== 1. 从网络下载 waimai_10k 数据集 ==========

def download_waimai_10k():
    """
    从 HuggingFace 下载 waimai_10k 数据集。
    返回 pandas DataFrame，列为 label, text。
    """
    print("正在从 HuggingFace 下载 waimai_10k 数据集...")
    ds = load_dataset("XiangPan/waimai_10k")
    # 通常为 train split 或全量在 "train"
    if "train" in ds:
        data = ds["train"]
    else:
        # 若只有单一 split，取第一个
        split_name = list(ds.keys())[0]
        data = ds[split_name]

    # 统一转为 pandas，列名可能为 review/label 或 text/label 等
    df = data.to_pandas()
    # 检测文本列：优先 review，其次 text, content, sentence
    text_candidates = ["review", "text", "content", "sentence", "review_content"]
    text_col = None
    for c in text_candidates:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        # 取第一个非 label 的列作为文本
        for c in df.columns:
            if c != "label" and df[c].dtype == object:
                text_col = c
                break
    if text_col is None:
        raise ValueError(f"未找到文本列，当前列: {list(df.columns)}")

    # 统一列名为 text, label
    df = df.rename(columns={text_col: "text"})
    if "label" not in df.columns:
        label_candidates = [c for c in df.columns if "label" in c.lower() or c in ("label", "labels")]
        if label_candidates:
            df = df.rename(columns={label_candidates[0]: "label"})
        else:
            raise ValueError(f"未找到标签列，当前列: {list(df.columns)}")

    # 只保留 text, label
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].dropna(subset=["text", "label"]).reset_index(drop=True)

    # 标签统一为 0/1（原始可能为 0/1 或 1/2 或 -1/1）
    labels = df["label"].unique()
    if set(labels).issubset({0, 1}):
        df["label"] = df["label"].astype(int)
    elif set(labels).issubset({1, 2}):
        df["label"] = (df["label"].astype(int) - 1)  # 1->0, 2->1
    else:
        # 映射：较小/负的为 0，较大/正的为 1
        sorted_labels = sorted(labels)
        label_map = {v: i for i, v in enumerate(sorted_labels)}
        df["label"] = df["label"].map(label_map)

    print(f"已加载 waimai_10k: 共 {len(df)} 条，标签分布: {df['label'].value_counts().to_dict()}")
    return df


# ========== 2. 划分并保存 train.csv / test.csv，构建词表 vocab.json ==========

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    df = download_waimai_10k()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    print(f"已保存: {train_path} ({len(train_df)} 条), {test_path} ({len(test_df)} 条)")

    # ========== 3. 使用 jieba 分词，构建词表 ==========
    print("正在使用 jieba 对训练集分词并统计词频...")
    all_tokens = []
    for text in train_df["text"].astype(str):
        all_tokens.extend(jieba.lcut(text.strip()))

    word_freq = Counter(w for w in all_tokens if w and w.strip())
    min_freq = 2
    vocab_words = [w for w, c in word_freq.items() if c >= min_freq]
    vocab_words = sorted(vocab_words, key=lambda x: (-word_freq[x], x))

    word2id = {"<PAD>": 0, "<UNK>": 1}
    for i, w in enumerate(vocab_words):
        word2id[w] = i + 2

    vocab_path = os.path.join(base_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(word2id, f, ensure_ascii=False, indent=2)
    print(f"词表已保存: {vocab_path}，词表大小: {len(word2id)}")
    print("数据准备完成。")


if __name__ == "__main__":
    main()

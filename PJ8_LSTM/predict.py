# -*- coding: utf-8 -*-
"""
加载 train.py 训练并保存的 checkpoint，对输入句子做情感预测（0=负向，1=正向）。

说明：
  - 网络结构类 BiLSTMClassifier 与 train.py 中**共用**（见 `from train import ...`），请勿在本文件
    再复制一份模型类，以免与训练时 `state_dict` 键名不一致导致无法加载。
  - 在 train.py 中实现 Bi-LSTM 并成功保存 `model_best.pt` 后，本脚本即可用同一套参数重建网络并推理。
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

import jieba
import torch
import torch.nn as nn

# 与 train.py 使用同一模型定义，保证 load_state_dict 键一致
from train import BiLSTMClassifier, get_device


def text_to_ids(
    text: str,
    word2id: Dict[str, int],
    max_len: int,
) -> torch.Tensor:
    """与 WaimaiDataset 中分词、截断、填充规则保持一致。"""
    pad_id = word2id.get("<PAD>", 0)
    unk_id = word2id.get("<UNK>", 1)
    tokens = jieba.lcut(text.strip())
    ids: List[int] = [word2id.get(t, unk_id) for t in tokens if t and t.strip()]
    if len(ids) > max_len:
        ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(pad_id)
    return torch.tensor([ids], dtype=torch.long)


def load_model(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, int], int]:
    # 兼容旧版 PyTorch：不传 weights_only
    ckpt = torch.load(ckpt_path, map_location=device)
    word2id: Dict[str, int] = ckpt["word2id"]
    hp = ckpt["hparams"]
    vocab_size = len(word2id)
    pad_idx = word2id.get("<PAD>", 0)
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=hp["embed_dim"],
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
        num_classes=2,
        dropout=hp["dropout"],
        pad_idx=pad_idx,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    max_len = int(hp.get("max_len", 128))
    return model, word2id, max_len


@torch.no_grad()
def predict_text(
    model: nn.Module,
    text: str,
    word2id: Dict[str, int],
    max_len: int,
    device: torch.device,
) -> Tuple[int, List[float]]:
    x = text_to_ids(text, word2id, max_len).to(device)
    logits = model(x)
    proba = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
    pred = int(logits.argmax(dim=1).item())
    return pred, proba


def main() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    p = argparse.ArgumentParser(description="Bi-LSTM 情感预测（依赖 train.py 中的模型类）")
    p.add_argument("--ckpt", type=str, default="model_best.pt", help="train.py 保存的权重")
    p.add_argument("--text", type=str, default=None, help="单句文本；不填则从标准输入读一行")
    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        print("未找到检查点文件:", args.ckpt, file=sys.stderr)
        print("请先完成 train.py 中的模型并训练，成功保存权重后再运行本脚本。", file=sys.stderr)
        sys.exit(1)

    device = get_device()
    model, word2id, max_len = load_model(args.ckpt, device)

    if args.text is not None and len(args.text) > 0:
        t = args.text
    else:
        try:
            t = input("请输入一句评论（回车结束）: ").strip()
        except EOFError:
            print("", file=sys.stderr)
            sys.exit(1)
    if not t:
        print("空输入", file=sys.stderr)
        sys.exit(1)

    label, proba = predict_text(model, t, word2id, max_len, device)
    name = "正向" if label == 1 else "负向"
    print(f"预测: {label}（{name}）")
    print(f"P(负向)={proba[0]:.4f}  P(正向)={proba[1]:.4f}")


if __name__ == "__main__":
    main()

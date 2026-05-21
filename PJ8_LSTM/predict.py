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


def tokenize(text: str) -> List[str]:
    if hasattr(jieba, "lcut"):
        return jieba.lcut(text)
    return list(jieba.cut(text))


def text_to_ids(
        text: str,
        word2id: Dict[str, int],
        max_len: int,
) -> torch.Tensor:
    """与 WaimaiDataset 中分词、截断、填充规则保持一致。"""
    pad_id = word2id.get("<PAD>", 0)
    unk_id = word2id.get("<UNK>", 1)
    tokens = tokenize(text.strip())
    ids: List[int] = [word2id.get(t, unk_id) for t in tokens if t and t.strip()]
    if len(ids) > max_len:
        ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(pad_id)
    return torch.tensor([ids], dtype=torch.long)


def load_model(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, int], int]:
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
    # 修改点 1：移除 os.chdir(base)，避免它强制把工作目录切走，
    # 这样我们在终端敲 runs/xxx.pt 时，相对路径才不会失效。

    p = argparse.ArgumentParser(description="Bi-LSTM 情感预测（支持循环交互与 runs 目录）")
    # 修改点 2：把默认路径改到 runs 文件夹下（这里假设你有一个默认的，也可以不设默认值）
    p.add_argument("--ckpt", type=str, default="runs/model_best.pt", help="找最佳参数")
    p.add_argument("--text", type=str, default=None, help="单句文本；若不填则进入连续交互模式")
    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"未找到检查点文件: {args.ckpt}", file=sys.stderr)
        print("请检查路径是否正确。例如：runs/你的模型文件名.pt", file=sys.stderr)
        sys.exit(1)

    print(f"正在加载模型权重: {args.ckpt} ...")
    device = get_device()
    model, word2id, max_len = load_model(args.ckpt, device)
    print("模型加载成功！")

    # 修改点 3：如果命令行直接传了 --text，就只预测单句（保留原功能）
    if args.text is not None and len(args.text) > 0:
        label, proba = predict_text(model, args.text, word2id, max_len, device)
        name = "正向" if label == 1 else "负向"
        print(f"\n输入: {args.text}")
        print(f"预测: {label}（{name}）")
        print(f"P(负向)={proba[0]:.4f}  P(正向)={proba[1]:.4f}")
        return

    # 修改点 4：如果没有传 --text，进入 while True 循环，等待用户源源不断地输入
    print("\n" + "=" * 40)
    print(" 进入智能情感判别系统（输入 'q' 或 'exit' 退出）")
    print("=" * 40)

    while True:
        try:
            t = input("\n请输入一句评论 ▷ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n程序平稳退出。")
            break

        if not t:
            print("⚠输入不能为空，请重新输入。")
            continue

        if t.lower() in ["q", "exit", "quit"]:
            print("👋 谢谢使用，再见！")
            break

        # 开始推理
        label, proba = predict_text(model, t, word2id, max_len, device)
        name = "正向" if label == 1 else "负向"

        # 结果
        print(f" 预测结果: {label} —— 【{name}】")
        print(f" ［详细概率］ P(负向): {proba[0]:.4f} | P(正向): {proba[1]:.4f}")


if __name__ == "__main__":
    main()
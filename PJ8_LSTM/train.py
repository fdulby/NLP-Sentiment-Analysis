# -*- coding: utf-8 -*-
"""
Bi-LSTM 中文情感分析训练脚本。
需先运行 prepare_data.py 生成 train.csv / test.csv / vocab.json。

本文件中 BiLSTMClassifier 的循环部分需你自行补全，禁止使用 nn.LSTM 等；详见类内说明。
predict.py 从本文件导入同一 BiLSTMClassifier，请勿在 predict 中再复制一份模型类。
"""
import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import jieba
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 复现实验的随机性（DataLoader 外仍有不确定因素时，可再设 torch.backends 等）
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


class WaimaiDataset(Dataset):
    """分词 -> id 序列；截断/填充为 max_len。"""

    def __init__(self, csv_path: str, word2id: Dict[str, int], max_len: int) -> None:
        self.df = pd.read_csv(csv_path)
        self.word2id = word2id
        self.max_len = max_len
        self.pad_id = word2id.get("<PAD>", 0)
        self.unk_id = word2id.get("<UNK>", 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = str(self.df.iloc[idx]["text"])
        label = int(self.df.iloc[idx]["label"])
        tokens = jieba.lcut(text.strip())
        ids: List[int] = [self.word2id.get(t, self.unk_id) for t in tokens if t and t.strip()]
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        while len(ids) < self.max_len:
            ids.append(self.pad_id)
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class BiLSTMClassifier(nn.Module):
    """
    目标结构（与《实验指导》一致，由你补全实现）：

        (batch, seq) -> Embedding
        -> **双向 LSTM 若干层**（须自行用基础层实现，见下方约束）
        -> 取**最后一层**在**最后一个时间步**上的「前向隐状态」与「后向隐状态」
        -> 拼接 (batch, 2*hidden_dim)
        -> Dropout -> Linear -> (batch, num_classes) 的 logits
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.pad_idx = pad_idx

        # ---------- 已给出：词向量层（可保留，勿删除）----------
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # ---------- 须补全：Bi-LSTM 部分 ----------
        # 要求通过自行实现 LSTM 内部计算（或等价展开）完成双向、多层（num_layers）逻辑，
        # 并能在 forward 中得到「最后一层」前向/后向在整句末时刻的隐表示用于拼接。
        #
        # 【禁止使用】（须自行展开门控与时序，不得直接依赖封装好的整条序列层）：
        #   - nn.LSTM, nn.LSTMCell
        #   - nn.RNN, nn.RNNCell, nn.GRU, nn.GRUCell
        #   以及任何封装好的「整条序列」循环层（若《实验指导》与课程总说明有冲突，以总说明为准）。
        #
        # 【允许使用】（示例，不限于此）：
        #   - nn.Linear, nn.Parameter
        #   - nn.Dropout, nn.ModuleList 等组织模块的方式
        #   - torch.tanh, torch.sigmoid, torch.cat 等张量运算
        #   请用上述基础组件自行组出 LSTM 的输入门/遗忘门/输出门/候选与状态更新（参考教材公式）。
        #
        # 提示：对 padding 位置，应在计算中掩蔽或使该时间步不污染最终用于分类的表示（与实验指导中 max_len
        #  填充约定一致；实现方式不唯一）。

        # TODO: 在此声明你为 Bi-LSTM 各层、各向所需的可学习参数（例如各门的 nn.Linear 与 nn.Parameter 等）
        # 采用“一个 Linear 同时计算 4 个门”的写法：
        # concat([h_prev, x_t]) -> Linear(hidden + input, 4*hidden)
        # 然后 chunk 成 f, i, g, o 四部分
        self.fw_cells = nn.ModuleList()
        self.bw_cells = nn.ModuleList()

        for layer_idx in range(num_layers):
            input_size = embed_dim if layer_idx == 0 else hidden_dim * 2

            fw_gate = nn.Linear(input_size + hidden_dim, 4 * hidden_dim)
            bw_gate = nn.Linear(input_size + hidden_dim, 4 * hidden_dim)

            # 我们这里的门顺序是: f, i, g, o
            # 所以前 hidden_dim 段对应 forget gate，可把 forget bias 设为 1.0
            with torch.no_grad():
                fw_gate.bias[:hidden_dim].fill_(1.0)
                bw_gate.bias[:hidden_dim].fill_(1.0)

            self.fw_cells.append(fw_gate)
            self.bw_cells.append(bw_gate)

        # ---------- 已给出：分类头（在拿到 concat 后的句向量 h 后使用，形状为 batch × (2*hidden_dim)）---
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        # cell
        def _lstm_cell(
                self,
                x_t: torch.Tensor,
                state: Tuple[torch.Tensor, torch.Tensor],
                gate_layer: nn.Linear,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            x_t:     (batch, input_size)
            state:   (h_prev, c_prev)，每个都是 (batch, hidden_dim)
            gate_layer: 一个 Linear(input_size + hidden_dim, 4*hidden_dim)
            """
            h_prev, c_prev = state

            concat = torch.cat([h_prev, x_t], dim=-1)  # (batch, hidden+input)
            gates_out = gate_layer(concat)  # (batch, 4*hidden)

            # 门顺序：f, i, g, o
            f_pre, i_pre, g_pre, o_pre = gates_out.chunk(4, dim=-1)

            f_t = torch.sigmoid(f_pre)
            i_t = torch.sigmoid(i_pre)
            g_t = torch.tanh(g_pre)
            o_t = torch.sigmoid(o_pre)

            c_t = f_t * c_prev + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            return h_t, c_t

        # 单向
        def _run_one_direction(
                self,
                seq_inputs: torch.Tensor,
                lengths: torch.Tensor,
                gate_layer: nn.Linear,
                reverse: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            seq_inputs: (batch, seq_len, input_size)
            lengths:    (batch,) 每个样本的真实长度（非 PAD 个数）
            gate_layer: 当前层当前方向的门控 Linear
            reverse:    是否反向扫描

            返回:
                outputs: (batch, seq_len, hidden_dim) 该方向每个时间步的输出
                h_last:  (batch, hidden_dim) 该方向整句扫描后的最终隐状态
            """
            batch_size, seq_len, _ = seq_inputs.size()
            device = seq_inputs.device
            dtype = seq_inputs.dtype

            h_t = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
            c_t = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

            outputs = [None] * seq_len
            time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

            for t in time_indices:
                x_t = seq_inputs[:, t, :]  # (batch, input_size)

                h_new, c_new = self._lstm_cell(x_t, (h_t, c_t), gate_layer)

                # 只在真实 token 位置更新；PAD 位置保持旧状态不变
                valid_mask = (t < lengths).unsqueeze(1).to(dtype)  # (batch, 1)

                h_t = valid_mask * h_new + (1.0 - valid_mask) * h_t
                c_t = valid_mask * c_new + (1.0 - valid_mask) * c_t

                outputs[t] = h_t.unsqueeze(1)  # 保持输出按原始时间顺序存放

            outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim)
            return outputs, h_t


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) 的 token id。

        须返回: (batch, num_classes) 的 logits，供 CrossEntropyLoss 使用。

        实现步骤提示（你应在代码中落实，并删掉本函数的 NotImplementedError）：
          1) emb = self.embedding(x)
          2) 对 emb 做双向、num_layers 层 LSTM 前向，得到最后一步的前向隐状态 h_f 与后向隐状态 h_b
          3) h = concat(h_f, h_b, dim=1)
          4) h = self.dropout(h)
          5) return self.fc(h)

        注意：「最后一步」在存在 PAD 时应对真实结束位置取隐状态，而不是固定取下标 max_len-1
        （若你暂时简化实现，也应在报告中说明）。
        """
        # 真实长度：非 PAD token 数
        lengths = (x != self.pad_idx).sum(dim=1)  # (batch,)

        # 1) Embedding
        layer_input = self.embedding(x)  # (batch, seq_len, embed_dim)

        # 2) 双向、多层时序展开
        for layer_idx in range(self.num_layers):
            fw_outputs, fw_last = self._run_one_direction(
                layer_input, lengths, self.fw_cells[layer_idx], reverse=False
            )
            bw_outputs, bw_last = self._run_one_direction(
                layer_input, lengths, self.bw_cells[layer_idx], reverse=True
            )

            # 当前层双向输出，作为下一层输入
            layer_input = torch.cat([fw_outputs, bw_outputs], dim=-1)  # (batch, seq_len, 2*hidden_dim)

        # 3)
        h = torch.cat([fw_last, bw_last], dim=1)  # (batch, 2*hidden_dim)

        # 4) Dropout
        h = self.dropout(h)

        # 5) 分类 logits
        logits = self.fc(h)  # (batch, num_classes)
        return logits


        #raise NotImplementedError(
            #"请在本类中补全 __init__ 中 Bi-LSTM 相关参数 与 本 forward 的完整前向过程；"
            #"禁止使用 nn.LSTM 等封装的循环层，详见类内注释。"
        #)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / max(n, 1)


def main() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    p = argparse.ArgumentParser(description="Bi-LSTM 情感分析训练")
    p.add_argument("--train_csv", type=str, default="train.csv")
    p.add_argument("--test_csv", type=str, default="test.csv")
    p.add_argument("--vocab", type=str, default="vocab.json")
    p.add_argument("--save", type=str, default="model_best.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for path in (args.train_csv, args.test_csv, args.vocab):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"缺少文件: {path}，请先运行 prepare_data.py")

    set_seed(args.seed)
    device = get_device()
    print("设备:", device)

    word2id = load_vocab(args.vocab)
    vocab_size = len(word2id)
    pad_idx = word2id.get("<PAD>", 0)

    train_set = WaimaiDataset(args.train_csv, word2id, args.max_len)
    test_set = WaimaiDataset(args.test_csv, word2id, args.max_len)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout,
        pad_idx=pad_idx,
    ).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, crit, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:3d} | train_loss: {tr_loss:.4f} | "
            f"test_loss: {te_loss:.4f} | test_acc: {te_acc:.4f}"
        )
        if te_acc >= best_acc:
            best_acc = te_acc
            payload = {
                "state_dict": model.state_dict(),
                "word2id": word2id,
                "hparams": {
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "max_len": args.max_len,
                },
            }
            torch.save(payload, args.save)
            print(f"  -> 已保存更优模型 (test_acc={te_acc:.4f}) 至 {args.save}")

    print(f"训练结束。最佳测试集准确率: {best_acc:.4f}，模型: {args.save}")


if __name__ == "__main__":
    main()

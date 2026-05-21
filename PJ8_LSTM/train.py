# -*- coding: utf-8 -*-
"""
Bi-LSTM 中文情感分析训练脚本。
需先运行 prepare_data.py 生成 processed_data/train.csv / val.csv / test.csv / vocab.json。
本文件中 BiLSTMClassifier 的循环部分需你自行补全，禁止使用 nn.LSTM 等；详见类内说明。
predict.py 从本文件导入同一 BiLSTMClassifier，请勿在 predict 中再复制一份模型类。
"""

import copy
import json
import os
import random
from typing import Any, Dict, List, Tuple

import jieba
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset


# ============================== CONFIG ==============================
CONFIG: Dict[str, Any] = {
    # 数据文件
    "processed_data_dir": "processed_data",
    "train_file": "train.csv",
    "val_file": "val.csv",
    "test_file": "test.csv",
    "vocab_file": "vocab.json",
    "metadata_file": "metadata.json",

    # 输出目录
    "output_root": "runs",
    "run_name_keys": [
        "lr",
        "batch_size",
        "hidden_dim",
        "dropout",
        "max_len",
        "num_layers",
        "embed_dim",
        "use_class_weights",
        "early_stopping_patience",
    ],

    # 随机性
    "seed": 42,

    # 模型超参数
    "max_len": 64,
    "embed_dim": 64,
    "hidden_dim": 64,
    "num_layers": 1,
    "dropout": 0.5,

    # 训练超参数
    "epochs": 30,
    "batch_size": 64,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "num_workers": 0,
    "early_stopping_patience": 4,
    "early_stopping_min_delta": 1e-4,
    "use_class_weights": True,

    # 类别名称：None 表示优先读 metadata.json；如果没有 metadata，则自动按 label 数量生成。
    "label_names": None,

    # 批量实验：每组只写相对默认 CONFIG 需要覆盖的超参数。
    "experiments": [
        {"lr": 5e-4, "dropout": 0.5, "hidden_dim": 16, "embed_dim": 32},
        {"lr": 5e-4, "dropout": 0.5, "hidden_dim": 16, "embed_dim": 64},
        {"lr": 5e-4, "dropout": 0.5, "hidden_dim": 32, "embed_dim": 32},
        {"lr": 5e-4, "dropout": 0.5, "hidden_dim": 32, "embed_dim": 64},
        {"lr": 5e-4, "dropout": 0.7, "hidden_dim": 16, "embed_dim": 32},
        {"lr": 5e-4, "dropout": 0.7, "hidden_dim": 16, "embed_dim": 64},
        {"lr": 5e-4, "dropout": 0.7, "hidden_dim": 32, "embed_dim": 32},
        {"lr": 5e-4, "dropout": 0.7, "hidden_dim": 32, "embed_dim": 64},
        {"lr": 1e-4, "dropout": 0.5, "hidden_dim": 16, "embed_dim": 32},
        {"lr": 1e-4, "dropout": 0.5, "hidden_dim": 16, "embed_dim": 64},
        {"lr": 1e-4, "dropout": 0.5, "hidden_dim": 32, "embed_dim": 32},
        {"lr": 1e-4, "dropout": 0.5, "hidden_dim": 32, "embed_dim": 64},
        {"lr": 1e-4, "dropout": 0.7, "hidden_dim": 16, "embed_dim": 32},
        {"lr": 1e-4, "dropout": 0.7, "hidden_dim": 16, "embed_dim": 64},
        {"lr": 1e-4, "dropout": 0.7, "hidden_dim": 32, "embed_dim": 32},
        {"lr": 1e-4, "dropout": 0.7, "hidden_dim": 32, "embed_dim": 64},
        {"lr": 5e-5, "dropout": 0.5, "hidden_dim": 16, "embed_dim": 32},
        {"lr": 5e-5, "dropout": 0.5, "hidden_dim": 16, "embed_dim": 64},
        {"lr": 5e-5, "dropout": 0.5, "hidden_dim": 32, "embed_dim": 32},
        {"lr": 5e-5, "dropout": 0.5, "hidden_dim": 32, "embed_dim": 64},
        {"lr": 5e-5, "dropout": 0.7, "hidden_dim": 16, "embed_dim": 32},
        {"lr": 5e-5, "dropout": 0.7, "hidden_dim": 16, "embed_dim": 64},
        {"lr": 5e-5, "dropout": 0.7, "hidden_dim": 32, "embed_dim": 32},
        {"lr": 5e-5, "dropout": 0.7, "hidden_dim": 32, "embed_dim": 64},
    ],
}
# ====================================================================


# 复现实验的随机性（DataLoader 外仍有不确定因素时，可再设 torch.backends 等）
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def tokenize(text: str) -> List[str]:
    if hasattr(jieba, "lcut"):
        return jieba.lcut(text)
    return list(jieba.cut(text))


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
        tokens = tokenize(text.strip())
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
    criterion: nn.Module,
) -> Dict[str, Any]:
    model.eval()
    total = 0
    loss_sum = 0.0
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[List[float]] = []
    all_indices: List[int] = []
    offset = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        prob = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

        batch_size = y.size(0)
        loss_sum += loss.item() * y.size(0)
        total += batch_size
        all_true.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        all_prob.extend(prob.cpu().tolist())
        all_indices.extend(range(offset, offset + batch_size))
        offset += batch_size

    return {
        "loss": loss_sum / max(total, 1),
        "acc": accuracy_score(all_true, all_pred) if all_true else 0.0,
        "true": all_true,
        "pred": all_pred,
        "prob": all_prob,
        "indices": all_indices,
    }


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
    grad_clip: float = 0.0,
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
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / max(n, 1)


def format_value_for_name(value: Any) -> str:
    text = str(value)
    return text.replace(".", "p").replace("-", "m").replace("/", "_")


def make_run_name(cfg: Dict[str, Any]) -> str:
    parts = []
    for key in cfg["run_name_keys"]:
        parts.append(f"{key}_{format_value_for_name(cfg[key])}")
    return "_".join(parts)


def make_experiment_configs(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    experiments = base_cfg.get("experiments") or [{}]
    configs: List[Dict[str, Any]] = []
    for overrides in experiments:
        cfg = copy.deepcopy(base_cfg)
        cfg.pop("experiments", None)
        cfg.update(overrides)
        configs.append(cfg)
    return configs


def build_data_info(cfg: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    data_dir = os.path.join(base_dir, cfg["processed_data_dir"])
    paths = {
        "train_csv": os.path.join(data_dir, cfg["train_file"]),
        "val_csv": os.path.join(data_dir, cfg["val_file"]),
        "test_csv": os.path.join(data_dir, cfg["test_file"]),
        "vocab_json": os.path.join(data_dir, cfg["vocab_file"]),
        "metadata_json": os.path.join(data_dir, cfg["metadata_file"]),
    }

    required_paths = [paths["train_csv"], paths["val_csv"], paths["test_csv"], paths["vocab_json"]]
    for path in required_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"缺少文件: {path}，请先运行 prepare_data.py")

    metadata = read_json(paths["metadata_json"]) if os.path.isfile(paths["metadata_json"]) else {}
    word2id = load_vocab(paths["vocab_json"])

    label_values = set()
    for split_path in (paths["train_csv"], paths["val_csv"], paths["test_csv"]):
        df = pd.read_csv(split_path, usecols=["label"])
        label_values.update(int(v) for v in df["label"].dropna().unique().tolist())
    labels = sorted(label_values)
    if not labels:
        raise ValueError("未在 processed_data 的 CSV 文件中找到 label。")
    if labels != list(range(len(labels))):
        raise ValueError(f"label 必须从 0 连续编号，当前标签为: {labels}")

    num_classes = max(int(metadata.get("num_classes", len(labels))), len(labels))
    label_names = cfg.get("label_names") or metadata.get("label_names")
    if not label_names or len(label_names) != num_classes:
        label_names = [f"class_{idx}" for idx in range(num_classes)]

    metadata.update(
        {
            "num_classes": num_classes,
            "label_names": label_names,
            "detected_labels": labels,
        }
    )
    return {
        "paths": paths,
        "word2id": word2id,
        "metadata": metadata,
    }


def create_loaders(data_info: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    word2id = data_info["word2id"]
    paths = data_info["paths"]
    train_set = WaimaiDataset(paths["train_csv"], word2id, cfg["max_len"])
    val_set = WaimaiDataset(paths["val_csv"], word2id, cfg["max_len"])
    test_set = WaimaiDataset(paths["test_csv"], word2id, cfg["max_len"])

    loader_kwargs = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def compute_class_weights(train_csv: str, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = pd.read_csv(train_csv, usecols=["label"])["label"].astype(int)
    counts = labels.value_counts().reindex(range(num_classes), fill_value=0).sort_index()
    if (counts == 0).any():
        missing = counts[counts == 0].index.tolist()
        raise ValueError(f"训练集中缺少类别: {missing}")

    total = float(counts.sum())
    weights = [total / (num_classes * float(count)) for count in counts.tolist()]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def is_validation_better(
    val_acc: float,
    val_loss: float,
    best_val_acc: float,
    best_val_loss: float,
    min_delta: float,
) -> bool:
    if val_acc > best_val_acc + min_delta:
        return True
    if abs(val_acc - best_val_acc) <= min_delta and val_loss < best_val_loss - min_delta:
        return True
    return False


def build_test_summary(eval_result: Dict[str, Any], label_names: List[str]) -> Dict[str, Any]:
    y_true = eval_result["true"]
    y_pred = eval_result["pred"]
    y_prob = eval_result["prob"]
    labels = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prob_pos = [row[1] for row in y_prob] if y_prob and len(y_prob[0]) > 1 else y_pred

    summary: Dict[str, Any] = {
        "loss": float(eval_result["loss"]),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "ap": float(average_precision_score(y_true, prob_pos)) if y_true and len(set(y_true)) > 1 else 0.0,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)) if y_true else 0.0,
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if y_true else 0.0,
        "per_class": [],
        "confusion_matrix": cm.tolist(),
    }

    precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    supports = [int(sum(1 for label in y_true if label == class_idx)) for class_idx in labels]
    for class_idx, name in enumerate(label_names):
        summary["per_class"].append(
            {
                "label": f"class_{class_idx}",
                "label_name": name,
                "precision": float(precisions[class_idx]),
                "recall": float(recalls[class_idx]),
                "f1": float(f1s[class_idx]),
                "support": supports[class_idx],
            }
        )

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        summary["binary_table"] = {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        }
    else:
        summary["binary_table"] = {}
    return summary


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    test_summary: Dict[str, Any],
    save_path: str,
) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [4.5, 1.2]})

    ax = axes[0]
    ax.plot(epochs, train_losses, marker="o", label="train_loss")
    ax.plot(epochs, val_losses, marker="o", label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.set_title("Train / Val Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    ax = axes[1]
    ax.axis("off")
    class_f1 = {row["label"]: row["f1"] for row in test_summary["per_class"]}
    summary_items = [
        ("Accuracy", test_summary["accuracy"]),
        ("Class 0 F1", class_f1.get("class_0", 0.0)),
        ("Class 1 F1", class_f1.get("class_1", 0.0)),
    ]
    x_positions = [0.17, 0.5, 0.83]
    for x, (name, value) in zip(x_positions, summary_items):
        ax.text(
            x,
            0.62,
            f"{value:.4f}",
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            x,
            0.28,
            name,
            ha="center",
            va="center",
            fontsize=11,
            color="#444444",
            transform=ax.transAxes,
        )
    ax.axhline(0.02, color="#dddddd", linewidth=1)
    ax.set_title("Test Summary", pad=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_wrong_predictions(
    test_csv: str,
    eval_result: Dict[str, Any],
    label_names: List[str],
    save_path: str,
) -> int:
    test_df = pd.read_csv(test_csv)
    rows = []
    for idx, true_label, pred_label, prob in zip(
        eval_result["indices"],
        eval_result["true"],
        eval_result["pred"],
        eval_result["prob"],
    ):
        if true_label == pred_label:
            continue
        row = test_df.iloc[int(idx)].to_dict()
        row["true_label"] = int(true_label)
        row["true_name"] = label_names[int(true_label)]
        row["pred_label"] = int(pred_label)
        row["pred_name"] = label_names[int(pred_label)]
        for class_idx, class_prob in enumerate(prob):
            row[f"prob_{class_idx}_{label_names[class_idx]}"] = float(class_prob)
        rows.append(row)

    wrong_df = pd.DataFrame(rows)
    wrong_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return len(wrong_df)


def run_one_experiment(cfg: Dict[str, Any], data_info: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    set_seed(cfg["seed"])
    device = get_device()
    run_name = make_run_name(cfg)
    run_dir = os.path.join(base_dir, cfg["output_root"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    paths = {
        "best_model": os.path.join(run_dir, "best_model.pt"),
        "loss_curve": os.path.join(run_dir, "loss_curve.png"),
        "wrong": os.path.join(run_dir, "wrong.csv"),
    }

    word2id = data_info["word2id"]
    metadata = data_info["metadata"]
    label_names = metadata["label_names"]
    num_classes = int(metadata["num_classes"])
    pad_idx = word2id.get("<PAD>", 0)

    train_loader, val_loader, test_loader = create_loaders(data_info, cfg)
    class_weights = None
    if cfg["use_class_weights"]:
        class_weights = compute_class_weights(
            train_csv=data_info["paths"]["train_csv"],
            num_classes=num_classes,
            device=device,
        )
        print(
            "类别权重:",
            {label_names[idx]: round(float(weight), 4) for idx, weight in enumerate(class_weights.cpu())},
        )

    model = BiLSTMClassifier(
        vocab_size=len(word2id),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=num_classes,
        dropout=cfg["dropout"],
        pad_idx=pad_idx,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch = cfg["epochs"]

    print(f"\n开始实验: {run_name}")
    print(f"设备: {device}")
    print(f"结果目录: {run_dir}")

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=cfg["grad_clip"],
        )
        val_result = evaluate(model, val_loader, device, criterion)
        val_loss = float(val_result["loss"])
        val_acc = float(val_result["acc"])
        train_losses.append(float(train_loss))
        val_losses.append(val_loss)

        is_better = is_validation_better(
            val_acc=val_acc,
            val_loss=val_loss,
            best_val_acc=best_val_acc,
            best_val_loss=best_val_loss,
            min_delta=cfg["early_stopping_min_delta"],
        )
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "word2id": word2id,
                    "hparams": {
                        "max_len": cfg["max_len"],
                        "embed_dim": cfg["embed_dim"],
                        "hidden_dim": cfg["hidden_dim"],
                        "num_layers": cfg["num_layers"],
                        "dropout": cfg["dropout"],
                        "num_classes": num_classes,
                        "pad_idx": pad_idx,
                    },
                    "label_names": label_names,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "config": cfg,
                },
                paths["best_model"],
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"best_epoch={best_epoch} | "
            f"no_improve={epochs_without_improvement}/{cfg['early_stopping_patience']}"
        )

        if epochs_without_improvement >= cfg["early_stopping_patience"]:
            stopped_early = True
            stop_epoch = epoch
            print(
                f"早停触发: 连续 {cfg['early_stopping_patience']} 轮验证集无提升，"
                f"停止在 epoch {epoch}，最佳 epoch 为 {best_epoch}。"
            )
            break

    checkpoint = torch.load(paths["best_model"], map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_result = evaluate(model, test_loader, device, criterion)
    test_summary = build_test_summary(test_result, label_names)
    test_loss = float(test_summary["loss"])
    test_acc = float(test_summary["accuracy"])
    wrong_count = save_wrong_predictions(
        test_csv=data_info["paths"]["test_csv"],
        eval_result=test_result,
        label_names=label_names,
        save_path=paths["wrong"],
    )
    plot_loss_curve(train_losses, val_losses, test_summary, paths["loss_curve"])

    summary = {
        "run_name": run_name,
        "run_dir": run_dir,
        "best": {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
        },
        "training_control": {
            "stopped_early": stopped_early,
            "stop_epoch": stop_epoch,
        },
        "final_test": {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1_macro": test_summary["f1_macro"],
            "test_ap": test_summary["ap"],
            "wrong_count": wrong_count,
        },
        "output_files": paths,
    }

    print(
        f"实验完成: {run_name} | "
        f"best_val_acc={best_val_acc:.4f} | "
        f"test_acc={test_acc:.4f} | "
        f"错误样本数={wrong_count}"
    )
    return summary


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    data_info = build_data_info(CONFIG, base_dir)
    experiment_configs = make_experiment_configs(CONFIG)

    summaries = []
    for cfg in experiment_configs:
        summaries.append(run_one_experiment(cfg, data_info, base_dir))

    print("\n全部实验完成。")
    for summary in summaries:
        best = summary["best"]
        final_test = summary["final_test"]
        print(
            f"{summary['run_name']} | "
            f"best_val_acc={best['best_val_acc']:.4f} | "
            f"test_acc={final_test['test_acc']:.4f} | "
            f"run_dir={summary['run_dir']}"
        )


if __name__ == "__main__":
    main()

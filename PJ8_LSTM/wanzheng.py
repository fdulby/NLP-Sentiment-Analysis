# -*- coding: utf-8 -*-
"""
一键完成 waimai_10k 情感分类实验：
1. 下载/读取数据并划分 train/val/test
2. 基于 train 构建词表
3. 训练手写 Bi-LSTM
4. 用 val 选择最佳模型参数
5. 在 test 上输出错误样本与混淆矩阵

使用方式：
    直接修改下方 CONFIG 后运行：
    python wanzheng.py
"""

import copy
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Tuple

import jieba
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset
except ImportError as exc:
    raise ImportError("请先安装 datasets: pip install datasets") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("请先安装 matplotlib: pip install matplotlib") from exc


# ============================== CONFIG ==============================
# 你主要需要改这里。若想一次跑多组参数，把不同参数写到 experiments 中。
CONFIG: Dict[str, Any] = {
    # 数据相关
    "dataset_name": "XiangPan/waimai_10k",
    "hf_cache_dir": "/root/autodl-tmp/hf_datasets_cache",
    "processed_data_dir": "processed_data",
    "force_prepare": False,  # True: 每次都重新下载/划分；False: 已有数据则复用
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "min_freq": 2,
    "label_names": ["负向", "正向"],

    # 结果相关
    "output_root": "runs",
    "run_name_keys": ["lr", "batch_size", "hidden_dim", "dropout", "max_len"],

    # 训练随机性
    "seed": 42,

    # 模型参数
    "max_len": 128,
    "embed_dim": 128,
    "hidden_dim": 128,
    "num_layers": 1,
    "dropout": 0.3,

    # 训练参数
    "epochs": 15,
    "batch_size": 64,
    "lr": 5e-4,
    "weight_decay": 0.0,
    "grad_clip": 5.0,
    "num_workers": 0,

    # 如果只想跑一组参数，保持空列表即可。
    # 如果想一次跑多组，可以这样写：
     "experiments": [
         {"lr": 1e-3, "batch_size": 64, "hidden_dim": 128, "dropout": 0.5,"num_layers": 1,"embed_dim": 128},
         {"lr": 5e-4, "batch_size": 64, "hidden_dim": 128, "dropout": 0.5,"num_layers": 1,"embed_dim": 128},
         {"lr": 1e-4, "batch_size": 64, "hidden_dim": 128, "dropout": 0.5,"num_layers": 1,"embed_dim": 128},
         {"lr": 5e-5, "batch_size": 64, "hidden_dim": 128, "dropout": 0.5,"num_layers": 1,"embed_dim": 128},
         {"lr": 5e-4, "batch_size": 64, "hidden_dim": 128, "dropout": 0.5,"num_layers": 1,"embed_dim": 64},
         {"lr": 5e-4, "batch_size": 64, "hidden_dim": 128, "dropout": 0.5,"num_layers": 2,"embed_dim": 128},
         {"lr": 5e-4, "batch_size": 64, "hidden_dim": 128, "dropout": 0.5,"num_layers": 2,"embed_dim": 64},
     ],
    #"experiments": [],
}
# ====================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.10g}"
        if "e" in text or "E" in text:
            text = f"{value:.10f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace(".", "p").replace("-", "m").replace("/", "_")


def make_run_name(cfg: Dict[str, Any]) -> str:
    short_names = {"batch_size": "batch"}
    parts = []
    for key in cfg["run_name_keys"]:
        name = short_names.get(key, key)
        parts.append(f"{name}_{safe_value(cfg[key])}")
    return "_".join(parts)


def dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def prepare_data(cfg: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    data_dir = os.path.join(base_dir, cfg["processed_data_dir"])
    os.makedirs(data_dir, exist_ok=True)

    paths = {
        "train_csv": os.path.join(data_dir, "train.csv"),
        "val_csv": os.path.join(data_dir, "val.csv"),
        "test_csv": os.path.join(data_dir, "test.csv"),
        "vocab_json": os.path.join(data_dir, "vocab.json"),
        "metadata_json": os.path.join(data_dir, "metadata.json"),
    }

    required_files = list(paths.values())
    if not cfg["force_prepare"] and all(os.path.isfile(path) for path in required_files):
        print(f"检测到已有处理后数据，直接复用: {data_dir}")
        word2id = read_json(paths["vocab_json"])
        metadata = read_json(paths["metadata_json"])
        return {"paths": paths, "word2id": word2id, "metadata": metadata}

    df = download_dataset(cfg)
    df, metadata = normalize_labels(df, cfg)
    train_df, val_df, test_df = split_train_val_test(df, cfg)
    word2id = build_vocab(train_df, cfg["min_freq"])

    train_df.to_csv(paths["train_csv"], index=False, encoding="utf-8")
    val_df.to_csv(paths["val_csv"], index=False, encoding="utf-8")
    test_df.to_csv(paths["test_csv"], index=False, encoding="utf-8")
    dump_json(paths["vocab_json"], word2id)

    metadata.update(
        {
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "vocab_size": int(len(word2id)),
            "min_freq": cfg["min_freq"],
            "train_ratio": cfg["train_ratio"],
            "val_ratio": cfg["val_ratio"],
            "test_ratio": cfg["test_ratio"],
        }
    )
    dump_json(paths["metadata_json"], metadata)

    print(
        "数据准备完成: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, "
        f"vocab={len(word2id)}"
    )
    return {"paths": paths, "word2id": word2id, "metadata": metadata}


class WaimaiDataset(Dataset):
    def __init__(self, csv_path: str, word2id: Dict[str, int], max_len: int) -> None:
        self.df = pd.read_csv(csv_path)
        self.word2id = word2id
        self.max_len = max_len
        self.pad_id = word2id.get("<PAD>", 0)
        self.unk_id = word2id.get("<UNK>", 1)

    def __len__(self) -> int:
        return len(self.df)

    def encode_text(self, text: str) -> torch.Tensor:
        tokens = jieba.lcut(text.strip())
        ids = [self.word2id.get(token, self.unk_id) for token in tokens if token and token.strip()]
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        while len(ids) < self.max_len:
            ids.append(self.pad_id)
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text = str(self.df.iloc[idx]["text"])
        label = int(self.df.iloc[idx]["label"])
        x = self.encode_text(text)
        y = torch.tensor(label, dtype=torch.long)
        sample_idx = torch.tensor(idx, dtype=torch.long)
        return x, y, sample_idx


class BiLSTMClassifier(nn.Module):
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
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fw_cells = nn.ModuleList()
        self.bw_cells = nn.ModuleList()

        for layer_idx in range(num_layers):
            input_size = embed_dim if layer_idx == 0 else hidden_dim * 2
            fw_gate = nn.Linear(input_size + hidden_dim, 4 * hidden_dim)
            bw_gate = nn.Linear(input_size + hidden_dim, 4 * hidden_dim)
            with torch.no_grad():
                fw_gate.bias[:hidden_dim].fill_(1.0)
                bw_gate.bias[:hidden_dim].fill_(1.0)
            self.fw_cells.append(fw_gate)
            self.bw_cells.append(bw_gate)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def _lstm_cell(
        self,
        x_t: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        gate_layer: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        concat = torch.cat([h_prev, x_t], dim=-1)
        f_pre, i_pre, g_pre, o_pre = gate_layer(concat).chunk(4, dim=-1)

        f_t = torch.sigmoid(f_pre)
        i_t = torch.sigmoid(i_pre)
        g_t = torch.tanh(g_pre)
        o_t = torch.sigmoid(o_pre)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def _run_one_direction(
        self,
        seq_inputs: torch.Tensor,
        lengths: torch.Tensor,
        gate_layer: nn.Linear,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = seq_inputs.size()
        device = seq_inputs.device
        dtype = seq_inputs.dtype

        h_t = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        outputs = [None] * seq_len
        time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        for t in time_indices:
            x_t = seq_inputs[:, t, :]
            h_new, c_new = self._lstm_cell(x_t, (h_t, c_t), gate_layer)
            valid_mask = (t < lengths).unsqueeze(1).to(dtype)
            h_t = valid_mask * h_new + (1.0 - valid_mask) * h_t
            c_t = valid_mask * c_new + (1.0 - valid_mask) * c_t
            outputs[t] = h_t.unsqueeze(1)

        return torch.cat(outputs, dim=1), h_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = (x != self.pad_idx).sum(dim=1)
        layer_input = self.embedding(x)

        for layer_idx in range(self.num_layers):
            fw_outputs, fw_last = self._run_one_direction(
                layer_input, lengths, self.fw_cells[layer_idx], reverse=False
            )
            bw_outputs, bw_last = self._run_one_direction(
                layer_input, lengths, self.bw_cells[layer_idx], reverse=True
            )
            layer_input = torch.cat([fw_outputs, bw_outputs], dim=-1)

        h = torch.cat([fw_last, bw_last], dim=1)
        h = self.dropout(h)
        return self.fc(h)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total += y.size(0)
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[List[float]] = []
    all_indices: List[int] = []

    for x, y, indices in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        prob = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        all_true.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        all_prob.extend(prob.cpu().tolist())
        all_indices.extend(indices.cpu().tolist())

    return {
        "loss": total_loss / max(total, 1),
        "acc": accuracy_score(all_true, all_pred) if all_true else 0.0,
        "true": all_true,
        "pred": all_pred,
        "prob": all_prob,
        "indices": all_indices,
    }


def plot_loss_curve(train_losses: List[float], val_losses: List[float], save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, marker="o", label="train_loss")
    plt.plot(epochs, val_losses, marker="o", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Val Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    label_names: List[str],
    save_path: str,
) -> List[List[int]]:
    labels = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Test Confusion Matrix")
    plt.colorbar()
    tick_marks = list(range(len(label_names)))
    plt.xticks(tick_marks, label_names)
    plt.yticks(tick_marks, label_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = cm.max() / 2.0 if cm.size > 0 and cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return cm.tolist()


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


def create_loaders(data_info: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    paths = data_info["paths"]
    word2id = data_info["word2id"]
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


def run_one_experiment(cfg: Dict[str, Any], data_info: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    set_seed(cfg["seed"])
    device = get_device()
    run_name = make_run_name(cfg)
    run_dir = os.path.join(base_dir, cfg["output_root"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    paths = {
        "best_model": os.path.join(run_dir, "best_model.pt"),
        "loss_curve": os.path.join(run_dir, "loss_curve.png"),
        "experiment_config": os.path.join(run_dir, "experiment_config.json"),
        "wrong_predictions": os.path.join(run_dir, "wrong_predictions.csv"),
        "confusion_matrix": os.path.join(run_dir, "confusion_matrix.png"),
    }

    metadata = data_info["metadata"]
    label_names = metadata["label_names"]
    num_classes = int(metadata["num_classes"])
    word2id = data_info["word2id"]
    pad_idx = word2id.get("<PAD>", 0)

    config_payload = {
        "run_name": run_name,
        "run_dir": run_dir,
        "device": str(device),
        "config": cfg,
        "data": metadata,
        "output_files": paths,
    }
    dump_json(paths["experiment_config"], config_payload)

    train_loader, val_loader, test_loader = create_loaders(data_info, cfg)
    model = BiLSTMClassifier(
        vocab_size=len(word2id),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_classes=num_classes,
        dropout=cfg["dropout"],
        pad_idx=pad_idx,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
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

    print(f"\n开始实验: {run_name}")
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
        val_result = evaluate(model, val_loader, criterion, device)
        val_loss = float(val_result["loss"])
        val_acc = float(val_result["acc"])

        train_losses.append(float(train_loss))
        val_losses.append(val_loss)

        is_better = val_acc > best_val_acc or (
            abs(val_acc - best_val_acc) < 1e-12 and val_loss < best_val_loss
        )
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
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

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"best_epoch={best_epoch}"
        )

    plot_loss_curve(train_losses, val_losses, paths["loss_curve"])

    checkpoint = torch.load(paths["best_model"], map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_result = evaluate(model, test_loader, criterion, device)
    test_acc = float(test_result["acc"])
    test_loss = float(test_result["loss"])

    wrong_count = save_wrong_predictions(
        test_csv=data_info["paths"]["test_csv"],
        eval_result=test_result,
        label_names=label_names,
        save_path=paths["wrong_predictions"],
    )
    cm = plot_confusion_matrix(
        y_true=test_result["true"],
        y_pred=test_result["pred"],
        label_names=label_names,
        save_path=paths["confusion_matrix"],
    )

    config_payload.update(
        {
            "best": {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
            },
            "final_test": {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "wrong_count": wrong_count,
                "confusion_matrix": cm,
            },
            "loss_history": {
                "train_loss": train_losses,
                "val_loss": val_losses,
            },
        }
    )
    dump_json(paths["experiment_config"], config_payload)

    print(
        f"实验完成: {run_name} | "
        f"best_val_acc={best_val_acc:.4f} | "
        f"test_acc={test_acc:.4f} | "
        f"错误样本数={wrong_count}"
    )
    return config_payload


def make_experiment_configs(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    experiments = base_cfg.get("experiments") or [{}]
    configs = []
    for overrides in experiments:
        cfg = copy.deepcopy(base_cfg)
        cfg.pop("experiments", None)
        cfg.update(overrides)
        configs.append(cfg)
    return configs


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    data_info = prepare_data(CONFIG, base_dir)
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

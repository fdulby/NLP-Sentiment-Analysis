# -*- coding: utf-8 -*-
"""
TextCNN baseline for the sentiment analysis project.

Reads data from ../processed_data and writes one metrics figure per experiment to:
    other/TextCNN/<hyperparameter-run-name>/metrics.png
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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset


CONFIG: Dict[str, Any] = {
    "processed_data_dir": "processed_data",
    "train_file": "train.csv",
    "val_file": "val.csv",
    "test_file": "test.csv",
    "vocab_file": "vocab.json",
    "metadata_file": "metadata.json",
    "output_root": os.path.join("other", "TextCNN"),
    "run_name_keys": [
        "lr",
        "batch_size",
        "num_filters",
        "filter_sizes",
        "dropout",
        "max_len",
        "embed_dim",
        "use_class_weights",
    ],
    "seed": 42,
    "max_len": 64,
    "embed_dim": 64,
    "num_filters": 128,
    "filter_sizes": [3, 4, 5],
    "dropout": 0.5,
    "epochs": 20,
    "batch_size": 64,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "num_workers": 0,
    "early_stopping_patience": 4,
    "early_stopping_min_delta": 1e-4,
    "use_class_weights": True,
    "label_names": None,
    "experiments": [],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> List[str]:
    if hasattr(jieba, "lcut"):
        return jieba.lcut(text)
    return list(jieba.cut(text))


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class WaimaiDataset(Dataset):
    def __init__(self, csv_path: str, word2id: Dict[str, int], max_len: int) -> None:
        self.df = pd.read_csv(csv_path)
        self.word2id = word2id
        self.max_len = max_len
        self.pad_id = word2id.get("<PAD>", 0)
        self.unk_id = word2id.get("<UNK>", 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        tokens = tokenize(str(row["text"]).strip())
        ids = [self.word2id.get(token, self.unk_id) for token in tokens if token and token.strip()]
        ids = ids[: self.max_len]
        ids.extend([self.pad_id] * (self.max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(int(row["label"]), dtype=torch.long)


class TextCNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_filters: int,
        filter_sizes: List[int],
        num_classes: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, kernel_size=(kernel_size, embed_dim)) for kernel_size in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).unsqueeze(1)
        conv_outputs = []
        for conv in self.convs:
            feature = F.relu(conv(emb)).squeeze(3)
            pooled = F.max_pool1d(feature, kernel_size=feature.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        h = torch.cat(conv_outputs, dim=1)
        return self.fc(self.dropout(h))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_value_for_name(value: Any) -> str:
    text = str(value)
    return text.replace(".", "p").replace("-", "m").replace("/", "_").replace(" ", "")


def make_run_name(cfg: Dict[str, Any]) -> str:
    return "_".join(f"{key}_{format_value_for_name(cfg[key])}" for key in cfg["run_name_keys"])


def make_experiment_configs(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    experiments = base_cfg.get("experiments") or [{}]
    configs = []
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
    for key in ("train_csv", "val_csv", "test_csv", "vocab_json"):
        if not os.path.isfile(paths[key]):
            raise FileNotFoundError(f"缺少文件: {paths[key]}")

    word2id = read_json(paths["vocab_json"])
    metadata = read_json(paths["metadata_json"]) if os.path.isfile(paths["metadata_json"]) else {}
    labels = set()
    for split_path in (paths["train_csv"], paths["val_csv"], paths["test_csv"]):
        labels.update(pd.read_csv(split_path, usecols=["label"])["label"].astype(int).unique().tolist())
    num_classes = max(int(metadata.get("num_classes", len(labels))), len(labels))
    label_names = cfg.get("label_names") or metadata.get("label_names")
    if not label_names or len(label_names) != num_classes:
        label_names = [f"class_{idx}" for idx in range(num_classes)]
    metadata.update({"num_classes": num_classes, "label_names": label_names})
    return {"paths": paths, "word2id": word2id, "metadata": metadata}


def create_loaders(data_info: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    word2id = data_info["word2id"]
    paths = data_info["paths"]
    kwargs = {"batch_size": cfg["batch_size"], "num_workers": cfg["num_workers"]}
    train_loader = DataLoader(WaimaiDataset(paths["train_csv"], word2id, cfg["max_len"]), shuffle=True, **kwargs)
    val_loader = DataLoader(WaimaiDataset(paths["val_csv"], word2id, cfg["max_len"]), shuffle=False, **kwargs)
    test_loader = DataLoader(WaimaiDataset(paths["test_csv"], word2id, cfg["max_len"]), shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


def compute_class_weights(train_csv: str, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = pd.read_csv(train_csv, usecols=["label"])["label"].astype(int)
    counts = labels.value_counts().reindex(range(num_classes), fill_value=0).sort_index()
    if (counts == 0).any():
        raise ValueError(f"训练集中缺少类别: {counts[counts == 0].index.tolist()}")
    total = float(counts.sum())
    weights = [total / (num_classes * float(count)) for count in counts.tolist()]
    return torch.tensor(weights, dtype=torch.float32, device=device)


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
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        prob = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        y_prob.extend(prob.cpu().tolist())
    return build_metrics(y_true, y_pred, y_prob, total_loss / max(total, 1))


def build_metrics(y_true: List[int], y_pred: List[int], y_prob: List[List[float]], loss: float) -> Dict[str, float]:
    prob_pos = [row[1] for row in y_prob] if y_prob and len(y_prob[0]) > 1 else y_pred
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
        "f1_positive": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)) if y_true else 0.0,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)) if y_true else 0.0,
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if y_true else 0.0,
        "ap": float(average_precision_score(y_true, prob_pos)) if y_true and len(set(y_true)) > 1 else 0.0,
    }


def is_better(current: Dict[str, float], best_acc: float, best_loss: float, min_delta: float) -> bool:
    if current["accuracy"] > best_acc + min_delta:
        return True
    if abs(current["accuracy"] - best_acc) <= min_delta and current["loss"] < best_loss - min_delta:
        return True
    return False


def plot_metrics(history: Dict[str, List[float]], test_metrics: Dict[str, float], save_path: str, title: str) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0][0]
    ax.plot(epochs, history["train_loss"], marker="o", label="train_loss")
    ax.plot(epochs, history["val_loss"], marker="o", label="val_loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    ax = axes[0][1]
    for key in ("f1_positive", "f1_macro", "f1_micro", "f1_weighted"):
        ax.plot(epochs, history[key], marker="o", label=f"val_{key}")
        ax.axhline(test_metrics[key], linestyle="--", alpha=0.35, label=f"test_{key}")
    ax.set_title("F1 Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    ax = axes[1][0]
    for key in ("precision_macro", "recall_macro"):
        ax.plot(epochs, history[key], marker="o", label=f"val_{key}")
        ax.axhline(test_metrics[key], linestyle="--", alpha=0.35, label=f"test_{key}")
    ax.set_title("Precision / Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    ax = axes[1][1]
    for key in ("accuracy", "ap"):
        ax.plot(epochs, history[key], marker="o", label=f"val_{key}")
        ax.axhline(test_metrics[key], linestyle="--", alpha=0.35, label=f"test_{key}")
    ax.set_title("Accuracy / AP")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def run_one_experiment(cfg: Dict[str, Any], data_info: Dict[str, Any], base_dir: str) -> Dict[str, float]:
    set_seed(cfg["seed"])
    device = get_device()
    run_name = make_run_name(cfg)
    run_dir = os.path.join(base_dir, cfg["output_root"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.png")

    train_loader, val_loader, test_loader = create_loaders(data_info, cfg)
    word2id = data_info["word2id"]
    metadata = data_info["metadata"]
    pad_idx = word2id.get("<PAD>", 0)
    num_classes = int(metadata["num_classes"])

    class_weights = None
    if cfg["use_class_weights"]:
        class_weights = compute_class_weights(data_info["paths"]["train_csv"], num_classes, device)

    model = TextCNNClassifier(
        vocab_size=len(word2id),
        embed_dim=cfg["embed_dim"],
        num_filters=cfg["num_filters"],
        filter_sizes=cfg["filter_sizes"],
        num_classes=num_classes,
        dropout=cfg["dropout"],
        pad_idx=pad_idx,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    history = {
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
        "precision_macro": [],
        "recall_macro": [],
        "f1_positive": [],
        "f1_macro": [],
        "f1_micro": [],
        "f1_weighted": [],
        "ap": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_acc = -1.0
    best_loss = float("inf")
    no_improve = 0

    print(f"\nTextCNN experiment: {run_name}")
    print(f"device={device} output={run_dir}")
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg["grad_clip"])
        val_metrics = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        for key in history:
            if key not in {"train_loss", "val_loss"}:
                history[key].append(val_metrics[key])

        if is_better(val_metrics, best_acc, best_loss, cfg["early_stopping_min_delta"]):
            best_acc = val_metrics["accuracy"]
            best_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} | val_ap={val_metrics['ap']:.4f}"
        )
        if no_improve >= cfg["early_stopping_patience"]:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device)
    plot_metrics(history, test_metrics, metrics_path, f"TextCNN - {run_name}")
    print(
        f"saved {metrics_path} | test_acc={test_metrics['accuracy']:.4f} | "
        f"test_f1_macro={test_metrics['f1_macro']:.4f} | test_ap={test_metrics['ap']:.4f}"
    )
    return test_metrics


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    os.chdir(base_dir)
    data_info = build_data_info(CONFIG, base_dir)
    for cfg in make_experiment_configs(CONFIG):
        run_one_experiment(cfg, data_info, base_dir)


if __name__ == "__main__":
    main()

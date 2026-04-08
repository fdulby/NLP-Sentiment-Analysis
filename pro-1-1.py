import pandas as pd
import jieba
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import time
import numpy as np
import random

# ================= 全局配置 =================
CONFIG = {
    # 【原始数据集路径】
    "raw_data_path": r"D:\LSTM\data_online\online_shopping_10_cats.csv",

    # 【预训练词向量路径】
    # 将你下载解压好的 txt 文件路径填在这里
    "pretrained_path": r"D:\LSTM\pretrained\sgns.weibo.word.txt",
    "pretrained_cache": r"D:\LSTM\data_online\embedding_matrix.npy",  # 自动生成的缓存，秒开用

    # 【工作目录及生成文件存储路径】
    "base_dir": r"D:\LSTM\data_online",
    "train_path": r"D:\LSTM\data_online\train_processed.csv",
    "dev_path": r"D:\LSTM\data_online\dev_processed.csv",
    "test_path": r"D:\LSTM\data_online\test_processed.csv",
    "vocab_path": r"D:\LSTM\data_online\vocab.json",
    "model_save": r"D:\LSTM\data_online\best_model.pth",
    "loss_plot": r"D:\LSTM\data_online\loss_curve.png",
    "cm_plot": r"D:\LSTM\data_online\confusion_matrix.png",

    # 【模型超参数】
    "vocab_size_max": 20000,
    "embedding_dim": 300,  # ！！！注意：这里必须改成与你下载的预训练向量维度一致（通常是300）！！！
    "hidden_dim": 128,
    "n_layers": 2,
    "dropout": 0.5,
    "max_len": 64,

    # 【CNN 专属参数】
    "num_filters": 256,  # 卷积核数量
    "kernel_size": 3,  # 卷积窗口大小

    # 【训练超参数 (Windows CUDA 优化版)】
    "batch_size": 512,
    "num_workers": 0,  # Windows 必须为 0
    "pin_memory": True,
    "lr": 0.001,
    "epochs": 50,
    "patience": 8,
    "weight_decay": 1e-4,  # L2正则化系数防过拟合
    "label_smoothing": 0.1,  # 标签平滑防过拟合

    # 【设备】
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True  # 混合精度
}


# ================= 1. 数据集拆分与预处理 =================
def prepare_data():
    os.makedirs(CONFIG["base_dir"], exist_ok=True)
    if os.path.exists(CONFIG["train_path"]) and os.path.exists(CONFIG["test_path"]):
        print("[Prepare] 检测到已处理的数据集，正在直接加载...")
        train_df = pd.read_csv(CONFIG["train_path"]).fillna({"review_tokens": ""})
        dev_df = pd.read_csv(CONFIG["dev_path"]).fillna({"review_tokens": ""})
        test_df = pd.read_csv(CONFIG["test_path"]).fillna({"review_tokens": ""})
        return train_df, dev_df, test_df

    print("[Prepare] 未检测到处理好的数据集，开始从原始文件构建...")
    df = pd.read_csv(CONFIG["raw_data_path"]).dropna(subset=['label', 'review'])

    print("[Prepare] 正在进行jieba分词...")
    df['review_tokens'] = df['review'].astype(str).apply(lambda x: " ".join(jieba.lcut(x)))

    print("[Prepare] 正在按 8:1:1 划分 Train/Dev/Test...")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    train_df[['label', 'review_tokens']].to_csv(CONFIG["train_path"], index=False, encoding='utf-8')
    dev_df[['label', 'review_tokens']].to_csv(CONFIG["dev_path"], index=False, encoding='utf-8')
    test_df[['label', 'review_tokens']].to_csv(CONFIG["test_path"], index=False, encoding='utf-8')
    return train_df, dev_df, test_df


# ================= 2. 词表构建 =================
def build_or_load_vocab(train_df, max_vocab_size):
    if os.path.exists(CONFIG["vocab_path"]):
        with open(CONFIG["vocab_path"], 'r', encoding='utf-8') as f:
            return json.load(f)

    all_words = []
    for text in train_df['review_tokens']:
        if isinstance(text, str):
            all_words.extend(text.split())

    word_counts = Counter(all_words)
    common_words = word_counts.most_common(max_vocab_size)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(common_words):
        vocab[word] = i + 2

    with open(CONFIG["vocab_path"], 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    return vocab


# ================= 3. 高效加载预训练词向量 =================
def load_pretrained_vectors(vocab, pretrained_path, cache_path, embedding_dim):
    """加载预训练词向量，带缓存机制，大幅加快后续启动速度"""

    # 1. 尝试直接加载 Numpy 缓存
    if os.path.exists(cache_path):
        print(f"[Vocab] 检测到词向量缓存，一秒加载: {cache_path}")
        weight_matrix = np.load(cache_path)
        return torch.from_numpy(weight_matrix).float()

    print(f"[Vocab] 未检测到缓存，开始解析原始预训练文件 (可能需要1-3分钟): {pretrained_path}")
    weight_matrix = np.random.normal(scale=0.6, size=(len(vocab), embedding_dim))
    weight_matrix[0] = np.zeros(embedding_dim)  # <PAD> 严格全0

    hits = 0
    try:
        with open(pretrained_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                values = line.rstrip().split(' ')
                # 第一行可能是统计信息 (词数 维度)，跳过
                if len(values) == 2:
                    continue

                word = values[0]
                if word in vocab:
                    idx = vocab[word]
                    vector = np.asarray(values[1:], dtype='float32')
                    if len(vector) == embedding_dim:
                        weight_matrix[idx] = vector
                        hits += 1

        print(f"[Vocab] 预训练向量命中率: {hits}/{len(vocab)} ({hits / len(vocab):.2%})")

        # 2. 保存缓存文件，下次秒开
        np.save(cache_path, weight_matrix)
        print(f"[Vocab] 权重矩阵缓存已保存至: {cache_path}")

    except FileNotFoundError:
        print("[Error] 警告：未找到预训练文件！将退化为【完全随机初始化】。请检查路径配置。")

    return torch.from_numpy(weight_matrix).float()


# ================= 4. 数据集与 DataLoader =================
class SentimentDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = int(self.df.iloc[idx]['label'])
        text = str(self.df.iloc[idx]['review_tokens'])
        tokens = text.split() if text else []
        ids = [self.vocab.get(token, 1) for token in tokens[:self.max_len]]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def get_dataloader(df, vocab, batch_size, shuffle=True):
    return DataLoader(
        SentimentDataset(df, vocab, CONFIG["max_len"]),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"] if CONFIG["device"] == "cuda" else False,
    )


# ================= 5. 模型定义 =================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        scores = self.attention(lstm_outputs).squeeze(2)
        alpha = torch.softmax(scores, dim=1).unsqueeze(2)
        return torch.sum(lstm_outputs * alpha, dim=1)


class SentimentCNNLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout, num_filters, kernel_size,
                 pretrained_weights=None):
        super().__init__()

        # 核心改动：如果有预训练权重，则加载它；否则随机初始化
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.dropout_emb = nn.Dropout(dropout)

        self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(num_filters, hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)

        lstm_out_dim = hidden_dim * 2
        self.attention = Attention(lstm_out_dim)

        self.fc = nn.Linear(lstm_out_dim * 2, 2)
        self.dropout_fc = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout_emb(self.embedding(text))
        conv_out = self.relu(self.conv(embedded.permute(0, 2, 1)))
        output, _ = self.lstm(conv_out.permute(0, 2, 1))

        context = self.attention(output)
        pooled = torch.max(output, dim=1)[0]
        combined = torch.cat((context, pooled), dim=1)
        return self.fc(self.dropout_fc(combined))


# ================= 6. 训练与评估逻辑 =================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds), all_preds, all_labels


def train_model(train_df, dev_df, vocab):
    print(f"\n[Train] 开始训练 | 设备: {CONFIG['device']}")
    print("-" * 60)

    train_loader = get_dataloader(train_df, vocab, CONFIG["batch_size"], shuffle=True)
    dev_loader = get_dataloader(dev_df, vocab, CONFIG["batch_size"], shuffle=False)

    # 1. 加载预训练词向量
    pretrained_weights = load_pretrained_vectors(
        vocab, CONFIG["pretrained_path"], CONFIG["pretrained_cache"], CONFIG["embedding_dim"]
    )

    # 2. 传入模型
    model = SentimentCNNLSTMAttention(
        len(vocab), CONFIG["embedding_dim"], CONFIG["hidden_dim"], CONFIG["n_layers"],
        CONFIG["dropout"], CONFIG["num_filters"], CONFIG["kernel_size"],
        pretrained_weights=pretrained_weights
    ).to(CONFIG["device"])

    # Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    # Weight Decay
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler() if CONFIG["use_amp"] and CONFIG["device"] == "cuda" else None

    history = {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}
    best_dev_acc, best_epoch, patience_counter = 0.0, 0, 0

    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()
        model.train()
        train_losses, train_preds, train_labels = [], [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            optimizer.zero_grad()

            if scaler:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        dev_loss, dev_acc, _, _ = evaluate(model, dev_loader, criterion, CONFIG["device"])
        scheduler.step(dev_loss)

        history['train_loss'].append(np.mean(train_losses))
        history['train_acc'].append(accuracy_score(train_labels, train_preds))
        history['dev_loss'].append(dev_loss)
        history['dev_acc'].append(dev_acc)

        print(f"[Epoch {epoch + 1:2d}] LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | Dev Loss: {dev_loss:.4f} | "
              f"Dev Acc: {dev_acc:.2%} | Time: {time.time() - epoch_start:.2f}s")

        if dev_acc > best_dev_acc:
            best_dev_acc, best_epoch, patience_counter = dev_acc, epoch + 1, 0
            torch.save(model.state_dict(), CONFIG["model_save"])
            print(f"  --> ✓ 保存最佳模型 (Acc: {dev_acc:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"\n[Train] 早停触发！最优轮次: {best_epoch} (Dev Acc: {best_dev_acc:.2%})")
                break

    plot_training_curve(history, best_epoch)
    return best_epoch


def plot_training_curve(history, best_epoch):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b.-', label='Train Loss')
    plt.plot(history['dev_loss'], 'r.-', label='Dev Loss')
    plt.axvline(x=best_epoch - 1, color='g', linestyle='--', alpha=0.7)
    plt.legend();
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'b.-', label='Train Acc')
    plt.plot(history['dev_acc'], 'r.-', label='Dev Acc')
    plt.axvline(x=best_epoch - 1, color='g', linestyle='--', alpha=0.7)
    plt.legend();
    plt.grid(True, alpha=0.3)
    plt.savefig(CONFIG["loss_plot"], dpi=150, bbox_inches='tight')


# ================= 7. 测试与错题分析 =================
def final_model(test_df, vocab):
    print("\n[Test] 正在加载最佳权重进行评估...")
    test_loader = get_dataloader(test_df, vocab, CONFIG["batch_size"], shuffle=False)

    # 构建模型框架 (需再次获取空权重用于占位)
    pretrained_weights = load_pretrained_vectors(vocab, CONFIG["pretrained_path"], CONFIG["pretrained_cache"],
                                                 CONFIG["embedding_dim"])
    model = SentimentCNNLST
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

# ================= 全局配置 =================
CONFIG = {
    # 【原始数据集路径】
    "raw_data_path": r"D:\LSTM\data_online\online_shopping_10_cats.csv",

    # 【工作目录及生成文件存储路径】
    "base_dir": r"D:\LSTM\data_online",
    "train_path": r"D:\LSTM\data_online\train_processed.csv",
    "dev_path": r"D:\LSTM\data_online\dev_processed.csv",
    "test_path": r"D:\LSTM\data_online\test_processed.csv",
    "vocab_path": r"D:\LSTM\data_online\vocab.json",  # 词表存储路径
    "model_save": r"D:\LSTM\data_online\best_model.pth",  # 最优模型权重
    "loss_plot": r"D:\LSTM\data_online\loss_curve.png",  # 训练曲线图
    "cm_plot": r"D:\LSTM\data_online\confusion_matrix.png",  # 混淆矩阵图

    # 【模型超参数】
    "vocab_size_max": 20000,  # 数据量变大，增加词表容量
    "embedding_dim": 256,
    "hidden_dim": 128,
    "n_layers": 2,
    "dropout": 0.5,
    "max_len": 64,

    # 【训练超参数】
    "batch_size": 512,
    "num_workers": 4,  # Windows环境下如果报错可以改成 0
    "pin_memory": True,
    "lr": 0.001,
    "epochs": 50,
    "patience": 8,

    # 【设备】自动检测CUDA
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True
}


# ================= 1. 数据集拆分与预处理 =================
def prepare_data():
    """
    检查是否已存在处理好的数据集。
    如果没有，读取原始数据，分词，按 8:1:1 拆分并保存。
    如果已有，直接加载。
    """
    os.makedirs(CONFIG["base_dir"], exist_ok=True)

    # 如果已经存在切分并分词好的文件，直接读取，节省大量时间
    if os.path.exists(CONFIG["train_path"]) and os.path.exists(CONFIG["test_path"]):
        print("[Prepare] 检测到已处理的数据集，正在直接加载...")
        train_df = pd.read_csv(CONFIG["train_path"])
        dev_df = pd.read_csv(CONFIG["dev_path"])
        test_df = pd.read_csv(CONFIG["test_path"])
        # 处理可能的 NaN 文本
        train_df['review_tokens'] = train_df['review_tokens'].fillna("")
        dev_df['review_tokens'] = dev_df['review_tokens'].fillna("")
        test_df['review_tokens'] = test_df['review_tokens'].fillna("")
        print(f"[Prepare] 加载完成 - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
        return train_df, dev_df, test_df

    print("[Prepare] 未检测到处理好的数据集，开始从原始文件构建...")
    # 读取原始数据
    df = pd.read_csv(CONFIG["raw_data_path"])

    # 清洗：移除空值 (online_shopping_10_cats 默认列名为 label 和 review)
    df = df.dropna(subset=['label', 'review'])

    # 分词
    print("[Prepare] 正在进行jieba分词，这可能需要1-2分钟...")
    start = time.time()
    df['review_tokens'] = df['review'].astype(str).apply(lambda x: " ".join(jieba.lcut(x)))
    print(f"[Prepare] 分词完成，耗时: {time.time() - start:.2f}s")

    # 按照 8:1:1 进行层次化划分 (stratify=df['label'] 保证正负样本比例均匀)
    print("[Prepare] 正在按 8:1:1 划分 Train/Dev/Test...")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    # 保存到本地，下次直接调用
    train_df[['label', 'review_tokens']].to_csv(CONFIG["train_path"], index=False, encoding='utf-8')
    dev_df[['label', 'review_tokens']].to_csv(CONFIG["dev_path"], index=False, encoding='utf-8')
    test_df[['label', 'review_tokens']].to_csv(CONFIG["test_path"], index=False, encoding='utf-8')

    print(f"[Prepare] 数据集划分并保存完成 - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    return train_df, dev_df, test_df


# ================= 2. 词表构建与加载 =================
def build_or_load_vocab(train_df, max_vocab_size):
    """如果词表文件存在则直接读取，否则根据训练集重新构建并保存"""
    if os.path.exists(CONFIG["vocab_path"]):
        print(f"[Vocab] 检测到本地词表，直接加载: {CONFIG['vocab_path']}")
        with open(CONFIG["vocab_path"], 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab

    print("[Vocab] 未检测到本地词表，开始根据训练集构建...")
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

    print(f"[Vocab] 词表构建完成并保存: 包含 {len(vocab)} 个词")
    return vocab


# ================= 3. 数据集与 DataLoader =================
class SentimentDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len
        self.unk_id = self.vocab.get("<UNK>", 1)
        self.pad_id = self.vocab.get("<PAD>", 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = int(self.df.iloc[idx]['label'])
        text = str(self.df.iloc[idx]['review_tokens'])
        tokens = text.split() if text else []

        ids = [self.vocab.get(token, self.unk_id) for token in tokens[:self.max_len]]
        if len(ids) < self.max_len:
            ids += [self.pad_id] * (self.max_len - len(ids))

        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def get_dataloader(df, vocab, batch_size, shuffle=True):
    dataset = SentimentDataset(df, vocab, CONFIG["max_len"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"] if CONFIG["device"] == "cuda" else False,
        persistent_workers=True if CONFIG["num_workers"] > 0 else False,
        prefetch_factor=2 if CONFIG["num_workers"] > 0 else None
    )


# ================= 4. 模型定义 (Bi-LSTM + Attention) =================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        scores = self.attention(lstm_outputs).squeeze(2)
        alpha = torch.softmax(scores, dim=1).unsqueeze(2)
        context_vector = torch.sum(lstm_outputs * alpha, dim=1)
        return context_vector


class SentimentAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        lstm_out_dim = hidden_dim * 2
        self.attention = Attention(lstm_out_dim)
        self.fc = nn.Linear(lstm_out_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, _ = self.lstm(embedded)
        context = self.attention(output)
        return self.fc(self.dropout(context))


# ================= 5. 训练与评估逻辑 =================
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

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def train_model(train_df, dev_df, vocab):
    print(f"\n[Train] 开始训练 | 设备: {CONFIG['device']} | 架构: Bi-LSTM + Attention")
    print("-" * 60)

    train_loader = get_dataloader(train_df, vocab, CONFIG["batch_size"], shuffle=True)
    dev_loader = get_dataloader(dev_df, vocab, CONFIG["batch_size"], shuffle=False)

    model = SentimentAttention(
        len(vocab),
        CONFIG["embedding_dim"],
        CONFIG["hidden_dim"],
        CONFIG["n_layers"],
        CONFIG["dropout"]
    ).to(CONFIG["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler() if CONFIG["use_amp"] and CONFIG["device"] == "cuda" else None

    history = {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}
    best_dev_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()
        model.train()
        train_losses = []
        train_preds, train_labels = [], []

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

        train_acc = accuracy_score(train_labels, train_preds)
        dev_loss, dev_acc, _, _ = evaluate(model, dev_loader, criterion, CONFIG["device"])

        scheduler.step(dev_loss)

        history['train_loss'].append(np.mean(train_losses))
        history['train_acc'].append(train_acc)
        history['dev_loss'].append(dev_loss)
        history['dev_acc'].append(dev_acc)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[Epoch {epoch + 1:2d}/{CONFIG['epochs']}] "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Dev Loss: {dev_loss:.4f} | "
              f"Dev Acc: {dev_acc:.2%} | "
              f"Time: {epoch_time:.2f}s")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["model_save"])
            print(f"  --> ✓ 准确率提升，保存最佳模型 (Acc: {dev_acc:.2%})")
        else:
            patience_counter += 1
            print(f"  --> 未改善 ({patience_counter}/{CONFIG['patience']})")

        if patience_counter >= CONFIG["patience"]:
            print(f"\n[Train] 早停触发！最优轮次: {best_epoch} (Dev Acc: {best_dev_acc:.2%})")
            break

    print(f"[Train] 训练完成，最优模型保存在: {CONFIG['model_save']}")
    plot_training_curve(history, best_epoch)
    return model, history, best_epoch


def plot_training_curve(history, best_epoch):
    """绘制并保存训练曲线，格式与你的截图一致"""
    plt.figure(figsize=(12, 4))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b.-', label='Train Loss')
    plt.plot(history['dev_loss'], 'r.-', label='Dev Loss')
    plt.axvline(x=best_epoch - 1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'b.-', label='Train Acc')
    plt.plot(history['dev_acc'], 'r.-', label='Dev Acc')
    plt.axvline(x=best_epoch - 1, color='g', linestyle='--', alpha=0.7)
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CONFIG["loss_plot"], dpi=150, bbox_inches='tight')
    print(f"[Plot] 训练曲线已保存: {CONFIG['loss_plot']}")


# ================= 6. 测试集评估 =================
def final_model(test_df, vocab):
    print("\n" + "=" * 60)
    print("[Test] 正在加载最佳权重并在测试集上进行评估...")

    test_loader = get_dataloader(test_df, vocab, CONFIG["batch_size"], shuffle=False)

    model = SentimentAttention(
        len(vocab),
        CONFIG["embedding_dim"],
        CONFIG["hidden_dim"],
        CONFIG["n_layers"],
        CONFIG["dropout"]
    ).to(CONFIG["device"])

    # 加载最佳模型权重
    model.load_state_dict(torch.load(CONFIG["model_save"], map_location=CONFIG["device"]))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, CONFIG["device"])

    print(f"\n[Test] 详细分类报告:")
    # 打印格式与你提供的截图完全一致
    print(classification_report(all_labels, all_preds, target_names=['负面', '正面'], digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
    plt.title("Confusion Matrix (Test Set)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(CONFIG["cm_plot"], dpi=150)
    print(f"[Plot] 混淆矩阵已保存: {CONFIG['cm_plot']}")

    return test_acc


# ================= 主函数 =================
def main():
    # 1. 切分数据（已加缓存判断）
    train_df, dev_df, test_df = prepare_data()

    # 2. 构建/加载词表（已加持久化判断）
    vocab = build_or_load_vocab(train_df, max_vocab_size=CONFIG["vocab_size_max"])

    # 3. 训练网络
    model, history, best_epoch = train_model(train_df, dev_df, vocab)

    # 4. 测试集检验
    test_acc = final_model(test_df, vocab)

    print("\n" + "=" * 60)
    print("[Finish] 全部流程完成！")
    print(f"  最终测试准确率: {test_acc:.2%}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    main()
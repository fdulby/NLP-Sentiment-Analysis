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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import time
import numpy as np

# ================= 全局配置（保留了你原有的全部路径） =================
CONFIG = {
    # 【数据路径】输入的3个TSV文件（train/dev/test三分法）
    "train_path": r"D:\LSTM\data-chn\train.tsv",
    "dev_path": r"D:\LSTM\data-chn\dev.tsv",
    "test_path": r"D:\LSTM\data-chn\test.tsv",

    # 【输出路径】所有结果保存到这里
    "output_dir": r"D:\LSTM\output",
    "model_save": r"D:\LSTM\output\best_model.pth",  # 最优模型权重
    "loss_plot": r"D:\LSTM\output\loss_curve.png",  # 训练曲线图
    "cm_plot": r"D:\LSTM\output\confusion_matrix.png",  # 混淆矩阵图
    "vocab_path": r"D:\LSTM\output\vocab.json",  # 词表（可复用）

    # 【模型超参数】(结合之前最优实验结果进行了微调)
    "embedding_dim": 256,  # 提升词嵌入维度
    "hidden_dim": 128,  # 降低隐藏层维度，防止过拟合
    "n_layers": 2,  # 结合Attention，2层Bi-LSTM足够
    "dropout": 0.5,
    "max_len": 64,

    # 【训练超参数】
    "batch_size": 512,
    "num_workers": 4,
    "pin_memory": True,
    "lr": 0.001,
    "epochs": 50,
    "patience": 8,  # 延长早停耐心值，给予模型跨越局部最优的时间

    # 【设备】自动检测CUDA
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True
}


# ================= 数据预处理 =================
def prepare_data():
    """读取并处理数据"""
    print("[Prepare] 开始数据预处理...")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    train_df = pd.read_csv(CONFIG["train_path"], sep='\t')
    dev_df = pd.read_csv(CONFIG["dev_path"], sep='\t')
    test_df = pd.read_csv(CONFIG["test_path"], sep='\t')

    train_df = train_df.dropna(subset=['label', 'text_a'])
    dev_df = dev_df.dropna(subset=['label', 'text_a'])
    test_df = test_df.dropna(subset=['label', 'text_a'])

    print("[Prepare] 正在进行jieba分词（可能需要30-60秒）...")
    start = time.time()

    train_df['review_tokens'] = train_df['text_a'].astype(str).apply(lambda x: " ".join(jieba.lcut(x)))
    dev_df['review_tokens'] = dev_df['text_a'].astype(str).apply(lambda x: " ".join(jieba.lcut(x)))
    test_df['review_tokens'] = test_df['text_a'].astype(str).apply(lambda x: " ".join(jieba.lcut(x)))

    print(f"[Prepare] 分词完成，耗时: {time.time() - start:.2f}s")
    return train_df, dev_df, test_df


def build_vocab(train_df, max_vocab_size=10000):
    """基于训练集构建词表"""
    all_words = []
    for text in train_df['review_tokens']:
        all_words.extend(text.split())

    word_counts = Counter(all_words)
    common_words = word_counts.most_common(max_vocab_size)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(common_words):
        vocab[word] = i + 2

    with open(CONFIG["vocab_path"], 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    return vocab


# ================= 数据集定义 =================
class SentimentDataset(Dataset):
    def __init__(self, df, vocab, max_len=64):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len
        self.unk_id = self.vocab.get("<UNK>", 1)
        self.pad_id = self.vocab.get("<PAD>", 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = int(self.df.iloc[idx]['label'])
        tokens = str(self.df.iloc[idx]['review_tokens']).split()

        ids = [self.vocab.get(token, self.unk_id) for token in tokens[:self.max_len]]
        if len(ids) < self.max_len:
            ids += [self.pad_id] * (self.max_len - len(ids))

        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def get_dataloader(df, vocab, batch_size=1024, shuffle=True):
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


# ================= 模型定义 (加入 Attention) =================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # 学习一个权重矩阵来计算每个时间步的重要性
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        # lstm_outputs 形状: (batch_size, seq_len, hidden_dim)

        # 1. 计算每个词的得分
        scores = self.attention(lstm_outputs).squeeze(2)  # (batch_size, seq_len)

        # 2. 归一化得分得到权重 (alpha)
        alpha = torch.softmax(scores, dim=1).unsqueeze(2)  # (batch_size, seq_len, 1)

        # 3. 将权重与 LSTM 隐状态加权求和，得到最终的上下文向量
        context_vector = torch.sum(lstm_outputs * alpha, dim=1)  # (batch_size, hidden_dim)
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

        # Bi-LSTM 输出维度是 hidden_dim * 2
        lstm_out_dim = hidden_dim * 2

        # 注意力层
        self.attention = Attention(lstm_out_dim)

        # 分类头
        self.fc = nn.Linear(lstm_out_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))

        # output 包含所有时间步的隐状态: (batch, seq_len, hidden_dim * 2)
        output, _ = self.lstm(embedded)

        # 通过 Attention 层获取加权特征
        context = self.attention(output)

        # 全连接层分类
        return self.fc(self.dropout(context))


# ================= 评估函数 =================
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


# ================= 训练流程 =================
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

    # 增加学习率调度器：如果验证集Loss连续3轮不降，学习率减半
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    scaler = GradScaler() if CONFIG["use_amp"] and CONFIG["device"] == "cuda" else None

    history = {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}
    best_dev_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()

        # ----- 训练阶段 -----
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

        # ----- 验证阶段 -----
        dev_loss, dev_acc, _, _ = evaluate(model, dev_loader, criterion, CONFIG["device"])

        # 步进学习率调度器
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

        # ----- 早停检查 -----
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
    """绘制并保存训练曲线"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-o', label='Train Loss', markersize=3)
    plt.plot(history['dev_loss'], 'r-o', label='Dev Loss', markersize=3)
    plt.axvline(x=best_epoch - 1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'b-o', label='Train Acc', markersize=3)
    plt.plot(history['dev_acc'], 'r-o', label='Dev Acc', markersize=3)
    plt.axvline(x=best_epoch - 1, color='g', linestyle='--', alpha=0.7)
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CONFIG["loss_plot"], dpi=150, bbox_inches='tight')
    print(f"[Train] 曲线已保存: {CONFIG['loss_plot']}")


# ================= 测试流程 =================
def final_model(test_df, vocab):
    print("\n" + "=" * 60)
    print("[Test] 在测试集上进行最终评估...")

    test_loader = get_dataloader(test_df, vocab, CONFIG["batch_size"], shuffle=False)

    model = SentimentAttention(
        len(vocab),
        CONFIG["embedding_dim"],
        CONFIG["hidden_dim"],
        CONFIG["n_layers"],
        CONFIG["dropout"]
    ).to(CONFIG["device"])

    model.load_state_dict(torch.load(CONFIG["model_save"], map_location=CONFIG["device"]))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, CONFIG["device"])

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n[Test] 测试结果:")
    print(f"  测试集大小: {len(test_df)} 条")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    print(f"\n[Test] 详细分类报告:")
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

    return test_acc


# ================= 主函数 =================
def main():
    train_df, dev_df, test_df = prepare_data()
    vocab = build_vocab(train_df)
    model, history, best_epoch = train_model(train_df, dev_df, vocab)
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
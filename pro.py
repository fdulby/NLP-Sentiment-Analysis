import pandas as pd
import jieba
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import time
import numpy as np

# ================= 全局配置（用户只需修改这里） =================
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

    # 【模型超参数】
    "embedding_dim": 128,
    "hidden_dim": 256,
    "n_layers": 2,
    "dropout": 0.5,
    "max_len": 64,

    # 【训练超参数】
    "batch_size": 512,  # Y9000P可设为512-1024
    "num_workers": 4,  # Windows建议≤4，Linux可设为8
    "pin_memory": True,
    "lr": 0.001,
    "epochs": 50,  # 早停会提前终止
    "patience": 3,  # 早停耐心值（验证集准确率不提升则停）

    # 【设备】自动检测CUDA
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True  # 混合精度加速（仅CUDA有效）
}


# ================= 数据预处理 =================
def prepare_data():
    """
    读取3个TSV文件，分词处理，保存为CSV（便于复用）
    返回处理后的DataFrame
    """
    print("[Prepare] 开始数据预处理...")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 读取TSV（sep='\t'）
    train_df = pd.read_csv(CONFIG["train_path"], sep='\t')
    dev_df = pd.read_csv(CONFIG["dev_path"], sep='\t')
    test_df = pd.read_csv(CONFIG["test_path"], sep='\t')

    print("列名：", train_df.columns.tolist())
    print(train_df.head())  # 查看前几行数据

    print(f"[Prepare] 原始数据 - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    # 清洗空值
    train_df = train_df.dropna(subset=['label', 'text_a'])
    dev_df = dev_df.dropna(subset=['label', 'text_a'])
    test_df = test_df.dropna(subset=['label', 'text_a'])

    # 分词（添加新列review_tokens）
    print("[Prepare] 正在进行jieba分词（可能需要30-60秒）...")
    start = time.time()

    train_df['review_tokens'] = train_df['text_a'].astype(str).apply(lambda x: " ".join(jieba.lcut(x)))
    dev_df['review_tokens'] = dev_df['text_a'].astype(str).apply(lambda x: " ".join(jieba.lcut(x)))
    test_df['review_tokens'] = test_df['text_a'].astype(str).apply(lambda x: " ".join(jieba.lcut(x)))

    print(f"[Prepare] 分词完成，耗时: {time.time() - start:.2f}s")

    # 保存处理后的文件（可选，方便调试）
    train_df[['label', 'review_tokens']].to_csv(
        os.path.join(CONFIG["output_dir"], "train_processed.csv"), index=False, encoding='utf-8'
    )
    dev_df[['label', 'review_tokens']].to_csv(
        os.path.join(CONFIG["output_dir"], "dev_processed.csv"), index=False, encoding='utf-8'
    )
    test_df[['label', 'review_tokens']].to_csv(
        os.path.join(CONFIG["output_dir"], "test_processed.csv"), index=False, encoding='utf-8'
    )

    print("[Prepare] 预处理完成，数据已保存至 output_dir")
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

    # 保存词表
    with open(CONFIG["vocab_path"], 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"[Prepare] 词表构建完成: {len(vocab)} 词")
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

        # 转ID并截断/填充
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


# ================= 模型定义 =================
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim * 2, 2)  # 二分类：负面/正面
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        # 拼接双向最后时刻的隐状态
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden_cat))


# ================= 评估函数 =================
def evaluate(model, dataloader, criterion, device):
    """通用评估函数：返回loss, acc, 预测列表, 真实列表"""
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


# ================= 训练流程（含早停） =================
def train_model(train_df, dev_df, vocab):
    """训练并返回最佳模型路径"""
    print(f"\n[Train] 开始训练 | 设备: {CONFIG['device']} | 早停耐心值: {CONFIG['patience']}")
    print("-" * 60)

    # 数据加载器
    train_loader = get_dataloader(train_df, vocab, CONFIG["batch_size"], shuffle=True)
    dev_loader = get_dataloader(dev_df, vocab, CONFIG["batch_size"], shuffle=False)

    # 模型初始化
    model = SentimentLSTM(
        len(vocab),
        CONFIG["embedding_dim"],
        CONFIG["hidden_dim"],
        CONFIG["n_layers"],
        CONFIG["dropout"]
    ).to(CONFIG["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scaler = GradScaler() if CONFIG["use_amp"] and CONFIG["device"] == "cuda" else None

    # 训练记录
    history = {
        'train_loss': [], 'train_acc': [],
        'dev_loss': [], 'dev_acc': []
    }

    # 早停变量（监控验证集准确率）
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

        # 记录历史
        history['train_loss'].append(np.mean(train_losses))
        history['train_acc'].append(train_acc)
        history['dev_loss'].append(dev_loss)
        history['dev_acc'].append(dev_acc)

        epoch_time = time.time() - epoch_start

        print(f"[Epoch {epoch + 1:2d}/{CONFIG['epochs']}] "
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
            print(f"  --> ✓ 验证准确率提升，保存模型 (Acc: {dev_acc:.2%})")
        else:
            patience_counter += 1
            print(f"  --> 未改善 ({patience_counter}/{CONFIG['patience']})")

        if patience_counter >= CONFIG["patience"]:
            print(f"\n[Train] 早停触发！最优轮次: {best_epoch} (Dev Acc: {best_dev_acc:.2%})")
            break

    print(f"[Train] 训练完成，最优模型保存在: {CONFIG['model_save']}")

    # 绘制Loss曲线
    plot_training_curve(history, best_epoch)

    return model, history, best_epoch


def plot_training_curve(history, best_epoch):
    """绘制并保存训练曲线"""
    plt.figure(figsize=(12, 4))

    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-o', label='Train Loss', markersize=3)
    plt.plot(history['dev_loss'], 'r-o', label='Dev Loss', markersize=3)
    plt.axvline(x=best_epoch - 1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Acc曲线
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
    print(f"[Train] Loss曲线已保存: {CONFIG['loss_plot']}")


# ================= 测试流程（最终评估） =================
def final_model(test_df, vocab):
    """在独立测试集上评估最终性能"""
    print("\n" + "=" * 60)
    print("[Test] 在测试集上进行最终评估...")

    test_loader = get_dataloader(test_df, vocab, CONFIG["batch_size"], shuffle=False)

    # 加载最佳模型
    model = SentimentLSTM(
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

    # 计算详细指标
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n[Test] 测试结果:")
    print(f"  测试集大小: {len(test_df)} 条")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # 分类报告
    print(f"\n[Test] 详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['负面', '正面'], digits=4))

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'], yticklabels=['负面', '正面'])
    plt.title("Confusion Matrix (Test Set)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(CONFIG["cm_plot"], dpi=150)
    print(f"[Test] 混淆矩阵已保存: {CONFIG['cm_plot']}")

    return test_acc


# ================= 主函数 =================
def main():
    # 1. 数据预处理（读取3个TSV，分词）
    train_df, dev_df, test_df = prepare_data()

    # 2. 构建词表（基于训练集）
    vocab = build_vocab(train_df)

    # 3. 训练（含验证和早停）
    model, history, best_epoch = train_model(train_df, dev_df, vocab)

    # 4. 最终测试（独立测试集）
    test_acc = final_model(test_df, vocab)

    print("\n" + "=" * 60)
    print("[Finish] 全部流程完成！")
    print(f"  最佳模型: {CONFIG['model_save']}")
    print(f"  训练曲线: {CONFIG['loss_plot']}")
    print(f"  混淆矩阵: {CONFIG['cm_plot']}")
    print(f"  最终测试准确率: {test_acc:.2%}")


if __name__ == "__main__":
    # 设置随机种子保证可复现
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    main()
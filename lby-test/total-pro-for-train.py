import pandas as pd
import jieba
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np

# ================= 配置（所有路径都在LSTM文件夹内） =================
CONFIG = {
    "input_raw": "LSTM/ChnSentiCorp.csv",  # 原始数据
    "output_dir": "LSTM/processed_data/",  # 处理后数据目录
    "train_data_path": "LSTM/processed_data/train.csv",
    "val_data_path": "LSTM/processed_data/val.csv",  # 9:1中的1
    "vocab_path": "LSTM/processed_data/vocab.json",
    "model_save": "LSTM/lstm_best.pth",  # 最优模型保存位置
    "loss_plot": "LSTM/training_curve.png",  # 训练曲线图

    # 模型参数
    "embedding_dim": 128,
    "hidden_dim": 256,
    "n_layers": 2,
    "dropout": 0.5,
    "max_len": 64,

    # 训练参数（Y9000P优化）
    "batch_size": 1024,
    "num_workers": 8,
    "pin_memory": True,
    "lr": 0.001,
    "epochs": 50,  # 设大点，早停会截断
    "patience": 5,  # 早停耐心值
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,

    # 划分比例 9:1
    "train_ratio": 0.9
}


# ================= 数据预处理 =================
def prepare_data():
    """9:1划分：90%训练，10%验证"""
    if all(os.path.exists(p) for p in [CONFIG["train_data_path"], CONFIG["val_data_path"], CONFIG["vocab_path"]]):
        print("[Prepare] 检测到已处理数据，跳过")
        return

    if not os.path.exists(CONFIG["input_raw"]):
        raise FileNotFoundError(f"找不到原始数据集 {CONFIG['input_raw']}，请确保ChnSentiCorp.csv放在LSTM文件夹中")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print(f"[Prepare] 开始预处理（划分比例: {CONFIG['train_ratio']:.0%}/{1 - CONFIG['train_ratio']:.0%}）")
    start = time.time()

    df = pd.read_csv(CONFIG["input_raw"]).dropna()
    print(f"[Prepare] 加载数据: {len(df)} 条")

    # 分词
    print("[Prepare] Jieba分词中...")
    df['review_tokens'] = df['review'].astype(str).apply(lambda x: " ".join(jieba.lcut(x)))

    # 9:1分层划分（保持标签分布一致）
    train_df, val_df = train_test_split(
        df[['label', 'review_tokens']],
        train_size=CONFIG['train_ratio'],
        random_state=42,
        stratify=df['label']
    )

    train_df.to_csv(CONFIG["train_data_path"], index=False, encoding='utf-8')
    val_df.to_csv(CONFIG["val_data_path"], index=False, encoding='utf-8')

    # 基于训练集构建词表
    build_vocab(train_df['review_tokens'], CONFIG["vocab_path"])

    print(f"[Prepare] 完成！耗时: {time.time() - start:.2f}s")
    print(f"[Prepare] Train: {len(train_df)} | Val: {len(val_df)}")


def build_vocab(tokenized_texts, save_path, max_vocab_size=10000):
    all_words = [word for text in tokenized_texts for word in text.split()]
    word_counts = Counter(all_words)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({word: i + 2 for i, (word, _) in enumerate(word_counts.most_common(max_vocab_size))})

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[Prepare] 词表构建完成: {len(vocab)} 词")


# ================= 模型定义 =================
class SentimentDataset(Dataset):
    def __init__(self, data_path, max_len=64):
        self.df = pd.read_csv(data_path)
        with open(CONFIG["vocab_path"], 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
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


def get_dataloader(data_path, batch_size=1024, shuffle=True):
    dataset = SentimentDataset(data_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        persistent_workers=True,
        prefetch_factor=4
    )


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden_cat))


# ================= 训练与验证（含早停） =================
def evaluate(model, data_loader, device):
    """返回验证集上的loss和accuracy"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels


def train():
    # 数据预处理
    prepare_data()

    # 加载词表
    with open(CONFIG["vocab_path"], 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    train_loader = get_dataloader(CONFIG["train_data_path"], shuffle=True)
    val_loader = get_dataloader(CONFIG["val_data_path"], shuffle=False)

    # 初始化模型
    model = SentimentLSTM(len(vocab), CONFIG["embedding_dim"], CONFIG["hidden_dim"],
                          output_dim=2, n_layers=CONFIG["n_layers"], dropout=CONFIG["dropout"]).to(CONFIG["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scaler = GradScaler() if CONFIG["use_amp"] and CONFIG["device"] == "cuda" else None

    # 早停变量
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print(f"\n[Train] 开始训练 | 设备: {CONFIG['device']} | 早停耐心值: {CONFIG['patience']}")
    print("-" * 60)

    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()

        # 训练阶段
        model.train()
        train_losses = []
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

        # 验证阶段
        val_loss, val_acc, _, _ = evaluate(model, val_loader, CONFIG["device"])

        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start

        print(f"[Epoch {epoch + 1:2d}/{CONFIG['epochs']}] "
              f"Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2%} | "
              f"Time: {epoch_time:.2f}s")

        # 早停检查：基于验证准确率保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["model_save"])
            print(f"  --> ✓ 验证准确率提升，保存最优模型 (Acc: {val_acc:.2%})")
        else:
            patience_counter += 1
            print(f"  --> 未改善 ({patience_counter}/{CONFIG['patience']})")

        if patience_counter >= CONFIG["patience"]:
            print(f"\n[Train] 早停触发！最优轮次: {best_epoch} (Val Acc: {best_val_acc:.2%})")
            break

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    plt.plot(history['val_loss'], 'r-o', label='Val Loss', markersize=4)
    plt.axvline(x=best_epoch - 1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], 'g-o', markersize=4)
    plt.axvline(x=best_epoch - 1, color='r', linestyle='--', alpha=0.7)
    plt.title(f"Validation Accuracy (Best: {best_val_acc:.2%})")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CONFIG["loss_plot"], dpi=150)
    print(f"[Train] 训练曲线已保存: {CONFIG['loss_plot']}")


if __name__ == "__main__":
    train()



#前四个文件的一个完整版
import pandas as pd
import jieba
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# 配置
CONFIG = {
    # 路径配置
    "input_raw": "ChnSentiCorp.csv",  # 原始数据集路径
    "output_dir": "./processed_data/",  # 输出目录
    "train_data_path": "./processed_data/train.csv",
    "test_data_path": "./processed_data/test.csv",
    "vocab_path": "./processed_data/vocab.json",
    "model_save": "lstm_best.pth",
    "loss_plot": "loss_curve.png",

    # 模型超参数
    "embedding_dim": 128,
    "hidden_dim": 256,
    "n_layers": 2,
    "dropout": 0.5,
    "max_len": 64,

    # 训练超参数
    "batch_size": 64,
    "lr": 0.001,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# prepare
def load_and_clean_data(path):
    df = pd.read_csv(path)
    return df.dropna()


def tokenize_text(text):
    tokens = jieba.lcut(str(text))
    return " ".join(tokens)


def build_vocab(tokenized_texts, save_path, max_vocab_size=10000):
    all_words = []
    for text in tokenized_texts:
        all_words.extend(text.split())

    word_counts = Counter(all_words)
    common_words = word_counts.most_common(max_vocab_size)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(common_words):
        vocab[word] = i + 2

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f"词表已构建，共包含 {len(vocab)} 个词汇。")
    return vocab


def prepare_data():
    """统筹数据预处理的全流程"""
    if not os.path.exists(CONFIG["input_raw"]):
        raise FileNotFoundError(f"找不到原始数据集 {CONFIG['input_raw']}，请检查路径！")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print("开始预处理数据...")

    data = load_and_clean_data(CONFIG["input_raw"])
    print("正在进行 jieba 分词...")
    data['review_tokens'] = data['review'].apply(tokenize_text)

    train_df, test_df = train_test_split(data[['label', 'review_tokens']], test_size=0.2, random_state=42)
    build_vocab(train_df['review_tokens'], CONFIG["vocab_path"])

    train_df.to_csv(CONFIG["train_data_path"], index=False, encoding='utf-8')
    test_df.to_csv(CONFIG["test_data_path"], index=False, encoding='utf-8')
    print("数据处理完成并已保存！\n" + "-" * 30)


# model
class SentimentDataset(Dataset):
    def __init__(self, data_path, vocab_path, max_len=50):
        self.df = pd.read_csv(data_path)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.max_len = max_len
        self.unk_id = self.vocab.get("<UNK>", 1)
        self.pad_id = self.vocab.get("<PAD>", 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = int(self.df.iloc[idx]['label'])
        text = str(self.df.iloc[idx]['review_tokens'])

        tokens = text.split()
        ids = [self.vocab.get(token, self.unk_id) for token in tokens]

        if len(ids) < self.max_len:
            ids += [self.pad_id] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def get_dataloader(data_path, vocab_path, batch_size=32, max_len=50, shuffle=True):
    dataset = SentimentDataset(data_path, vocab_path, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(self.dropout(hidden_cat))


# train
def train():
    with open(CONFIG["vocab_path"], 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    train_loader = get_dataloader(CONFIG["train_data_path"], CONFIG["vocab_path"],
                                  CONFIG["batch_size"], CONFIG["max_len"])
    test_loader = get_dataloader(CONFIG["test_data_path"], CONFIG["vocab_path"],
                                 CONFIG["batch_size"], CONFIG["max_len"], shuffle=False)

    model = SentimentLSTM(len(vocab), CONFIG["embedding_dim"], CONFIG["hidden_dim"],
                          output_dim=2, n_layers=CONFIG["n_layers"], dropout=CONFIG["dropout"])
    model = model.to(CONFIG["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    epoch_losses = []
    best_test_acc = 0.0

    print(f"开始训练，设备: {CONFIG['device']}")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        # 测试集评估
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = correct / total
        print(f"Epoch [{epoch + 1}/{CONFIG['epochs']}] - Loss: {avg_loss:.4f} - Test Acc: {test_acc:.4%}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), CONFIG["model_save"])
            print(f"--> 检测到更好的模型，已保存权重至 {CONFIG['model_save']} (Acc: {test_acc:.4%})")

    # 绘制并保存 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, CONFIG["epochs"] + 1), epoch_losses, marker='o', color='b', label='Train Loss')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(CONFIG["loss_plot"])
    print(f"\n训练完成！Loss 曲线已保存为 {CONFIG['loss_plot']}")


# main
if __name__ == "__main__":
    # 1. 检查是否需要生成数据
    if not os.path.exists(CONFIG["train_data_path"]) or not os.path.exists(CONFIG["vocab_path"]):
        prepare_data()
    else:
        print("检测到已处理好的数据，跳过预处理阶段。\n" + "-" * 30)

    # 2. 执行训练
    train()
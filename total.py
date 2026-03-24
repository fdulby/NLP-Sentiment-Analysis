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

# [优化点 2] 统一配置字典：涵盖路径、模型结构、训练参数
CONFIG = {
    "paths": {
        "input_raw": "ChnSentiCorp.csv",      # 原始数据集
        "output_dir": "./processed_data/",    # 输出目录
        "train_csv": "./processed_data/train.csv",
        "test_csv": "./processed_data/test.csv",
        "vocab_json": "./processed_data/vocab.json",
        "model_save": "lstm_best.pth",
        "loss_plot": "loss_curve.png"
    },
    "model": {
        "embedding_dim": 128,
        "hidden_dim": 256,
        "n_layers": 2,
        "dropout": 0.5,
        "max_len": 64
    },
    "train": {
        "batch_size": 64,
        "lr": 0.001,
        "epochs": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
}

# 配置路径，存在输入的中文数据集与输出
INPUT_DATA_PATH = ""  # 例如: "ChnSentiCorp.csv"
OUTPUT_DIR = ""  # 例如: "./processed_data/"

# 创建输出目录
if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


#加载数据
def load_and_clean_data(path):
    # 假设 CSV 包含 'label' 和 'review' 两列
    df = pd.read_csv(path)
    # 去除空值
    df = df.dropna()
    return df


# 分词处理
def tokenize_text(text):
    # 使用 jieba 精确模式分词
    # 在这里去除停用词、标点符号？
    tokens = jieba.lcut(str(text))
    return " ".join(tokens)  # 用空格拼接，方便存入 CSV


# 建立vocab
def build_vocab(tokenized_texts, save_path, max_vocab_size=10000):
    """
    训练集生成词典：{词语: ID}
    特殊 token: <PAD> (填充), <UNK> (未知词)
    """
    all_words = []
    for text in tokenized_texts:
        all_words.extend(text.split())

    # 统计词频，取前 max_vocab_size 个高频词
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(max_vocab_size)

    # 构建词表，0 给 <PAD>，1 给 <UNK>
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(common_words):
        vocab[word] = i + 2

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    print(f"词表已构建，共包含 {len(vocab)} 个词汇。")
    return vocab


#main
if __name__ == "__main__":
    if not INPUT_DATA_PATH or not OUTPUT_DIR:
        print("错误：请先在脚本中设置 INPUT_DATA_PATH 和 OUTPUT_DIR 路径！")
    else:
        print("开始预处理数据...")

        # 1. 加载
        data = load_and_clean_data(INPUT_DATA_PATH)

        # 2. 分词
        print("正在进行 jieba 分词...")
        data['review_tokens'] = data['review'].apply(tokenize_text)

        # 3. 划分 (80% 训练, 20% 测试)
        train_df, test_df = train_test_split(data[['label', 'review_tokens']], test_size=0.2, random_state=42)

        # 4. 构建词表 (仅基于训练集构建，防止测试集信息泄露)
        vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
        build_vocab(train_df['review_tokens'], vocab_path)

        # 5. 保存 CSV
        train_path = os.path.join(OUTPUT_DIR, "train.csv")
        test_path = os.path.join(OUTPUT_DIR, "test.csv")

        train_df.to_csv(train_path, index=False, encoding='utf-8')
        test_df.to_csv(test_path, index=False, encoding='utf-8')

        print(f"数据处理完成！")
        print(f"训练集已保存至: {train_path}")
        print(f"测试集已保存至: {test_path}")
        print(f"词表已保存至: {vocab_path}")



class SentimentDataset(Dataset):
    def __init__(self, data_path, vocab_path, max_len=50):
        """
        :param data_path: train.csv 或 test.csv 的路径
        :param vocab_path: vocab.json 的路径
        :param max_len: 句子长度（超过截断，不足补齐）
        """
        # 1. 加载数据
        self.df = pd.read_csv(data_path)

        # 2. 加载词表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        self.max_len = max_len
        self.unk_id = self.vocab.get("<UNK>", 1)
        self.pad_id = self.vocab.get("<PAD>", 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取标签和分词后的文本
        label = int(self.df.iloc[idx]['label'])
        text = str(self.df.iloc[idx]['review_tokens'])

        # 将词语列表转换为 ID 列表
        tokens = text.split()
        ids = []
        for token in tokens:
            # 如果词在词表里就取 ID，不在就取 <UNK> 的 ID
            ids.append(self.vocab.get(token, self.unk_id))

        # 3. Padding & Truncating (填充与截断)
        if len(ids) < self.max_len:
            # 不足部分补 0
            ids += [self.pad_id] * (self.max_len - len(ids))
        else:
            # 超过部分直接截断
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# 简单测一下
def get_dataloader(data_path, vocab_path, batch_size=32, max_len=50, shuffle=True):
    dataset = SentimentDataset(data_path, vocab_path, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    # 连接路径
    TRAIN_CSV = "path/to/train.csv"
    VOCAB_JSON = "path/to/vocab.json"

    # 简单测试一下输出是否正确
    try:
        test_loader = get_dataloader(TRAIN_CSV, VOCAB_JSON, batch_size=2)
        first_batch = next(iter(test_loader))
        inputs, labels = first_batch
        print("输入 Tensor 形状:", inputs.shape)  # 应该是 [2, 50]
        print("标签 Tensor 形状:", labels.shape)  # 应该是 [2]
        print("第一条数据的 ID 序列:", inputs[0])
    except Exception as e:
        print("请检查路径是否正确后再测试:", e)



class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        """
        :param vocab_size: 词表大小 (vocab.json 中的长度)
        :param embedding_dim: 每个词向量的维度 (通常设为 128, 256 )
        :param hidden_dim: LSTM 隐藏层的神经元数量
        :param output_dim: 分类数量 (本项目为 2：正向和负向)
        :param n_layers: LSTM 的层数
        :param dropout: 丢弃率，用于防止过拟合
        """
        super(SentimentLSTM, self).__init__()

        # Embedding 层：将单词 ID 映射为稠密向量
        # padding_idx=0 位置是填充的，不需要更新梯度
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM ：
        # batch_first=True 使得输入形状为 [batch_size, seq_len, embedding_dim]
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout)

        # 全连接层：
        # 双向 LSTM，所以输入特征维度是 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Dropout 层：
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        """
        :param text: [batch_size, seq_len]
        :return: logits (未经过 softmax 的分类得分)
        """
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(text))

        # output: 每个时间步的输出 [batch_size, seq_len, hidden_dim * 2]
        # hidden: 最后一步的状态 [num_layers * 2, batch_size, hidden_dim]
        output, (hidden, cell) = self.lstm(embedded)

        # 重要：拼接双向 LSTM 的隐藏状态
        # hidden[-2,:,:] 是最后一层正向 LSTM 的隐藏状态
        # hidden[-1,:,:] 是最后一层反向 LSTM 的隐藏状态
        # 将它们在维度 1 (特征维度) 上进行拼接
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # 将拼接后的向量送入全连接层得到分类结果
        # [batch_size, hidden_dim * 2] -> [batch_size, output_dim]
        return self.fc(self.dropout(hidden_cat))


# test
if __name__ == "__main__":
    # 模拟超参数
    V_SIZE = 5000
    E_DIM = 128
    H_DIM = 256
    O_DIM = 2  # 正/负
    LAYERS = 2
    DROP = 0.5

    model = SentimentLSTM(V_SIZE, E_DIM, H_DIM, O_DIM, LAYERS, DROP)
    print(model)

    # 模拟一条输入数据: Batch_size=4, Seq_len=50
    test_input = torch.randint(0, V_SIZE, (4, 50))
    test_output = model(test_input)
    print("模型输出形状:", test_output.shape)  # 应该是 [# 4, 2]


def train():
    # 数据与模型
    with open(config["vocab_path"], 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    train_loader = get_dataloader(config["train_data_path"], config["vocab_path"],
                                  config["batch_size"], config["max_len"])
    test_loader = get_dataloader(config["test_data_path"], config["vocab_path"],
                                 config["batch_size"], config["max_len"], shuffle=False)

    model = SentimentLSTM(len(vocab), config["embedding_dim"], config["hidden_dim"],
                          output_dim=2, n_layers=config["n_layers"], dropout=config["dropout"])
    model = model.to(config["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # loss记录
    epoch_losses = []
    best_test_acc = 0.0

    print(f"开始训练，设备: {config['device']}")

    #循环
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        # 用测试集合评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = correct / total
        print(f"Epoch [{epoch + 1}/{config['epochs']}] - Loss: {avg_loss:.4f} - Test Acc: {test_acc:.4%}")

        #保存最好的参数
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "lstm_best.pth")
            print(f"--> 检测到更好的模型，已保存权重至 lstm_best.pth (Acc: {test_acc:.4%})")

    # 绘制loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config["epochs"] + 1), epoch_losses, marker='o', color='b', label='Train Loss')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()
    print("训练完成！Loss 曲线已保存为 loss_curve.png")


if __name__ == "__main__":
    train()
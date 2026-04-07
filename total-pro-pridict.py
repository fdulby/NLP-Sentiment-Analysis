import jieba
import json
import torch
import torch.nn as nn
import os

# ================= 配置 =================
CONFIG = {
    "vocab_path": "LSTM/processed_data/vocab.json",
    "model_save": "LSTM/lstm_best.pth",
    "embedding_dim": 128,
    "hidden_dim": 256,
    "n_layers": 2,
    "dropout": 0.5,
    "max_len": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ================= 模型定义（与训练一致） =================
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


# ================= 预测器 =================
class Predictor:
    def __init__(self):
        self.device = CONFIG["device"]

        # 加载词表和模型
        with open(CONFIG["vocab_path"], 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        self.model = SentimentLSTM(len(self.vocab), CONFIG["embedding_dim"], CONFIG["hidden_dim"],
                                   output_dim=2, n_layers=CONFIG["n_layers"], dropout=CONFIG["dropout"]).to(self.device)
        self.model.load_state_dict(torch.load(CONFIG["model_save"], map_location=self.device))
        self.model.eval()

    def predict(self, text):
        # 分词 -> 转ID -> 填充
        tokens = jieba.lcut(str(text))
        ids = [self.vocab.get(t, 1) for t in tokens[:CONFIG["max_len"]]]
        if len(ids) < CONFIG["max_len"]:
            ids += [0] * (CONFIG["max_len"] - len(ids))

        inputs = torch.tensor([ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)[0]

        return {
            "负面": f"{probs[0].item():.4f}",
            "正面": f"{probs[1].item():.4f}"
        }


# ================= 主程序：直接输入文本 =================
if __name__ == "__main__":
    # 在这里直接修改要预测的文本
    text = "这家餐厅的服务态度非常差，等了一个小时还没上菜"

    # 执行预测
    predictor = Predictor()
    result = predictor.predict(text)

    # 输出结果
    print(f"文本: {text}")
    print(f"负面概率: {result['负面']}")
    print(f"正面概率: {result['正面']}")

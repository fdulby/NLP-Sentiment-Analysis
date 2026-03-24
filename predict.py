import torch
import torch.nn.functional as F
import jieba
import json
import os
from model import SentimentLSTM

#对于环境的配置
config = {
    "model_path": "lstm_best.pth",
    "vocab_path": "processed_data/vocab.json",
    "embedding_dim": 128,
    "hidden_dim": 256,
    "n_layers": 2,
    "max_len": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


class Predictor:
    def __init__(self, config):
        self.config = config

        # 加载vocab
        with open(config["vocab_path"], 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        # init
        self.model = SentimentLSTM(
            len(self.vocab),
            config["embedding_dim"],
            config["hidden_dim"],
            output_dim=2,
            n_layers=config["n_layers"],
            dropout=0.0
        )

        # 加载权重
        if os.path.exists(config["model_path"]):
            self.model.load_state_dict(torch.load(config["model_path"], map_location=config["device"]))
            self.model.to(config["device"])
            self.model.eval()  # 切换到评估模式
            print(f"成功加载模型权重: {config['model_path']}")
        else:
            raise FileNotFoundError(f"未找到权重文件: {config['model_path']}，请先运行 train.py")

    def predict(self, text):
        # 1. 分词
        tokens = jieba.lcut(text)

        # 2. Token 转 ID (处理未知词 <UNK>)
        unk_id = self.vocab.get("<UNK>", 1)
        pad_id = self.vocab.get("<PAD>", 0)
        ids = [self.vocab.get(t, unk_id) for t in tokens]

        # 3. Padding/Truncating (对齐长度)
        if len(ids) < self.config["max_len"]:
            ids += [pad_id] * (self.config["max_len"] - len(ids))
        else:
            ids = ids[:self.config["max_len"]]

        # 4. 转为 Tensor 并推理
        input_tensor = torch.tensor([ids]).to(self.config["device"])  # 增加 batch 维度

        with torch.no_grad():
            outputs = self.model(input_tensor)  # 得到 logits
            # 使用 Softmax 将得分转化为概率
            probs = F.softmax(outputs, dim=1)

        # 获取概率最大的类别及其概率
        prob, pred = torch.max(probs, dim=1)

        label = "正向 (Positive)" if pred.item() == 1 else "负向 (Negative)"
        return label, prob.item()


# 交互
if __name__ == "__main__":
    try:
        predictor = Predictor(config)
        print("\n--- 情感分析预测系统已启动 (输入 'quit' 退出) ---")

        while True:
            user_input = input("\n请输入一段评价: ").strip()
            if user_input.lower() == 'quit':
                break
            if not user_input:
                continue

            label, score = predictor.predict(user_input)
            print(f"分析结果: {label}")
            print(f"置信度: {score:.4f}")

    except Exception as e:
        print(f"运行出错: {e}")
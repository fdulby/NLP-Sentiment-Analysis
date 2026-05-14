import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import os
from dataset_lby import get_dataloader  # 导入逻辑
from model_lby import SentimentLSTM  # 导入模型

#超参数
config = {
    "train_data_path": "processed_data/train.csv",
    "test_data_path": "processed_data/test.csv",
    "vocab_path": "processed_data/vocab.json",
    "embedding_dim": 128,
    "hidden_dim": 256,
    "n_layers": 2,
    "dropout": 0.5,
    "batch_size": 64,
    "lr": 0.001,
    "epochs": 10,
    "max_len": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


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
import torch
import torch.nn as nn


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
    print("模型输出形状:", test_output.shape)  # 应该是 [4, 2]
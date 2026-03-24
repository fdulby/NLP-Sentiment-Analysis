import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json


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
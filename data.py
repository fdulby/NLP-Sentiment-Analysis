import pandas as pd
import jieba
import json
import os
from sklearn.model_selection import train_test_split
from collections import Counter

# ================= 配置路径（请根据你的实际情况填写） =================
INPUT_DATA_PATH = ""  # 例如: "ChnSentiCorp.csv" 或 "waimai_10k.csv"
OUTPUT_DIR = ""  # 例如: "./processed_data/"

# 创建输出目录
if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ================= 1. 加载数据 =================
def load_and_clean_data(path):
    # 假设 CSV 包含 'label' 和 'review' 两列
    df = pd.read_csv(path)
    # 去除空值
    df = df.dropna()
    return df


# ================= 2. 分词处理 =================
def tokenize_text(text):
    # 使用 jieba 精确模式分词
    # 也可以根据需要在这里去除停用词、标点符号等
    tokens = jieba.lcut(str(text))
    return " ".join(tokens)  # 用空格拼接，方便存入 CSV


# ================= 3. 构建词表 (Vocab) =================
def build_vocab(tokenized_texts, save_path, max_vocab_size=10000):
    """
    根据训练集生成词典：{词语: ID}
    包含特殊 token: <PAD> (填充), <UNK> (未知词)
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


# ================= 主流程 =================
if __name__ == "__main__":
    if not INPUT_DATA_PATH or not OUTPUT_DIR:
        print("错误：请先在脚本中设置 INPUT_DATA_PATH 和 OUTPUT_DIR 路径！")
    else:
        print("开始预处理数据...")

        # 1. 加载
        data = load_and_clean_data(INPUT_DATA_PATH)

        # 2. 分词 (这一步在数据量大时可能较慢)
        print("正在进行 jieba 分词...")
        data['review_tokens'] = data['review'].apply(tokenize_text)

        # 3. 划分数据集 (80% 训练, 20% 测试)
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
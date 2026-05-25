## 基于 Bi-LSTM 的中文情感分析
一个面向课程实验的中文情感二分类项目，基于手写 Bi-LSTM（不依赖 nn.LSTM）对中文外卖评价进行正向 / 负向判别。数据集来自 HuggingFace 的 waimai_10k（约 1 万条中文餐饮评论），使用 jieba 分词，支持从数据准备、训练到交互式预测的完整流程。
### 项目结构
.
├── PJ8_LSTM/                  # 主线实验版（推荐使用）
│   ├── train.py               # 训练脚本，含手写 BiLSTMClassifier
│   ├── prepare_data.py        # 数据下载、清洗、划分、构建词表
│   ├── predict.py             # 加载模型，交互式预测
│   ├── requirements.txt       # 依赖列表
│   ├── 实验指导.md             # 课程实验说明
│   ├── processed_data/        # 预处理后的数据（train/val/test.csv, vocab.json）
│   ├── runs/                  # 批量实验结果
│   ├── attention/             # Attention 扩展实验
│   │   └── train-attention.py # Bi-LSTM + Attention 实现
│   ├── maxpool/               # MaxPool 扩展实验
│   │   └── train-maxpool.py   # Bi-LSTM + MaxPool 实现
│   └── other/                 # 其他对比模型
│       ├── GRU.py             # Bi-GRU 实现
│       ├── TextCNN.py         # TextCNN 实现
│       └── contrast.md        # 各模型对比分析
│
├── lby-test/                  # 根目录版本（使用 nn.LSTM 调包实现）
    ├── prepare_data_lby.py    # 数据预处理
    ├── dataset_lby.py         # 自定义 Dataset
    ├── model_lby.py           # PyTorch 内置 nn.LSTM 模型
    ├── train_lby.py           # 训练脚本
    ├── predict_lby.py         # 预测脚本
    └── pro*.py / total*.py    # 多个扩展优化版本
### 环境要求
*Python 3.8+
*PyTorch 1.9+
*jieba、pandas、scikit-learn、matplotlib、datasets


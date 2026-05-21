# LSTM 与 Attention 的连接

## 原始 BiLSTM

原来的模型只使用两个向量：

```text
fw_last + bw_last
```

也就是前向最后状态和后向最后状态。

## 加入 Attention 后

BiLSTM 不再只取最后状态，而是保留所有时间步输出：

```text
H = [h1, h2, ..., ht]
```

其中每个 `h_t` 都是前向和后向隐藏状态的拼接。

## Attention 计算

先对每个时间步的输出做线性变换：

```text
U = tanh(H W + b)
```

再计算每个词的重要性分数：

```text
score = U v
```

对所有时间步做 softmax：

```text
alpha = softmax(score)
```

`alpha` 就是每个词的注意力权重。

## 得到句向量

用注意力权重对所有 LSTM 输出加权求和：

```text
context = sum(alpha_t * h_t)
```

这个 `context` 是整句话的表示。

## 最终分类

```text
context -> Dropout -> Linear -> logits
```

## 核心区别

- 原 BiLSTM：只看最后的前向/后向状态。
- Attention-BiLSTM：看所有时间步，并自动学习哪些词更重要。
- Attention 更适合处理关键词明显或正负混合的评论。

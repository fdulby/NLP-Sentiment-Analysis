# LSTM 与 MaxPool 的连接

## 原始 BiLSTM

原来的模型只取：

```text
fw_last + bw_last
```

也就是最后的前向状态和后向状态。

## 加入 MaxPool 后

BiLSTM 保留所有时间步输出：

```text
H = [h1, h2, ..., ht]
```

每个 `h_t` 都是前向和后向隐藏状态的拼接。

## MaxPool 计算

对所有时间步做最大池化：

```text
pooled = max(H, dim=time)
```

也就是每个特征维度只保留整句话中最强的响应。

## Padding 处理

`<PAD>` 位置不参与池化。

做法是先把 PAD 对应的输出设成很小的值：

```text
PAD hidden -> -inf
```

这样 max pooling 不会选到填充位置。

## 最终分类

```text
pooled -> Dropout -> Linear -> logits
```

## 核心区别

- 原 BiLSTM：只使用最后状态。
- MaxPool-BiLSTM：使用所有时间步，并选出最强特征。
- MaxPool 更擅长捕捉“难吃”“太慢”“很好吃”等强关键词。

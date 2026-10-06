原始 Transformer 采用正余弦绝对位置编码，其主要特点有：

1. 无可学习参数，不需要额外训练位置嵌入。
2. 可以为任意位置计算编码，形式上支持超出训练长度；但不保证模型在更长序列上的效果。
3. **多频率结构，提供不同位置尺度的变化模式。**
4. **隐含相对位置线性性质：固定偏移对应固定的线性变换。**

==正余弦位置编码的关键设计：位置移动一个固定距离后，新的位置向量能够由原向量经过一个固定的线性变换得到。== 原论文希望这一性质有助于模型学习相对位置关系。

## 数学形式

$$
\begin{aligned}
PE(pos,2i) &= \sin\left( \frac{pos}{10000^{\frac{2i}{d_{model}}}} \right) \\
PE(pos,2i+1) &= \cos\left( \frac{pos}{10000^{\frac{2i}{d_{model}}}} \right)
\end{aligned}
$$

位置 $pos$ 和特征维度均从 $0$ 开始编号。下方示例假设 $d_{model}$ 为正偶数，每两个维度共享同一频率，分别使用 sin 和 cos。

![[{52AD81BD-000F-4C84-8521-7AB508EEC60A}.png]]

## Code
```python
import math
import torch

def sinusoidal_position_encoding(maxlen, d_model):
    if d_model <= 0 or d_model % 2 != 0:
        raise ValueError("d_model 必须为正偶数")
    pe = torch.zeros(maxlen, d_model)

    position = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
```

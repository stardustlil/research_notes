---
aliases:
  - 可学习绝对位置编码
  - 可学习位置嵌入
tags:
  - Transformer/位置编码
---

# Learned Positional Embedding（可学习绝对位置编码）

上级：[[Position Encode|位置编码]] ｜ 前置：[[Self-Attention#置换等变性|置换等变性]]

> [!abstract] 总结
> 为每个绝对位置学习一个位置向量，直接加到对应 token 的向量上：$\tilde X=X+P_{1:n}$。

## 动机

无 mask 的自注意力满足 $f(\Pi X)=\Pi f(X)$，无法区分 token 的位置（见 [[Self-Attention#置换等变性|置换等变性]]）。最直接的办法：给每个位置准备一个可学习的向量，让模型自己学会「位置 $t$ 应该长什么样」。

## 符号

| 符号 | 形状 | 含义 |
| --- | --- | --- |
| $P$ | $n_{\max}\times d$ | 可学习位置矩阵 |
| $p_i$ | $1\times d$ | $P$ 的第 $i$ 行，即位置编号 $t_i=i-1$ 的位置向量 |
| $P_{1:n}$ | $n\times d$ | $P$ 的前 $n$ 行 |
| $G$ | $d\times d$ | $W_QW_K^\top$，仅用于展开分数 |

## 定义

对长度 $n\le n_{\max}$ 的输入 $X\in\mathbb{R}^{n\times d}$，取 $P$ 的前 $n$ 行：

$$
P_{1:n}=\begin{bmatrix}p_1\\ \vdots\\ p_n\end{bmatrix}\in\mathbb{R}^{n\times d},
\qquad
\tilde X=X+P_{1:n}\in\mathbb{R}^{n\times d}.
$$

之后用 $\tilde X$ 代替 $X$ 进入[[Self-Attention|自注意力]]，即 $\tilde x_i=x_i+p_i$。

## 性质与推导

### 打破置换等变性

把输入行打乱为 $\Pi X$ 后，位置矩阵**不随之打乱**：

$$
\Pi X+P_{1:n}\neq\Pi\left(X+P_{1:n}\right)\quad(\text{一般情况}).
$$

所以同一个 token 出现在不同位置时，进入注意力的向量不同，注意力可以区分位置。

### 位置如何进入注意力分数

令 $G=W_QW_K^\top\in\mathbb{R}^{d\times d}$。加入位置后 $\tilde q_i=\tilde x_iW_Q$、$\tilde k_j=\tilde x_jW_K$，展开二者的点积：

$$
\tilde q_i\tilde k_j^\top
=(x_i+p_i)\,G\,(x_j+p_j)^\top
=x_iGx_j^\top+x_iGp_j^\top+p_iGx_j^\top+p_iGp_j^\top.
$$

四项中位置都以**绝对**形式 $p_i,p_j$ 出现；分数是否只依赖位置差 $i-j$，完全取决于学到的参数。这正是 [[RoPE#动机|RoPE 的出发点]]。

## 代码实现

```python
import torch
import torch.nn as nn

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.P = nn.Parameter(torch.empty(max_len, d_model))    # P: [n_max, d]
        nn.init.normal_(self.P, std=0.02)

    def forward(self, x):
        # x: [B, n, d]
        n = x.shape[-2]
        if n > self.P.shape[0]:
            raise ValueError("序列长度超过 n_max，没有对应的位置向量")
        return x + self.P[:n]                                   # X̃ = X + P_{1:n}: [B, n, d]
```

## 对比与易错点

- **不能超出 $n_{\max}$**：没有训练过的位置就没有对应的行，这是它与[[Sinusoidal positional encoding|正余弦位置编码]]的主要区别。
- 常见实现写成 `nn.Embedding(max_len, d)` 再用 `torch.arange(n)` 查表，与本实现等价：查表的下标正是 $t_i$。

## 相关笔记

- [[Sinusoidal positional encoding|正余弦位置编码]]：把可学习的 $P$ 换成固定的多频率正余弦函数。
- [[RoPE|旋转位置编码]]：不再把位置加到输入上，而是旋转 $q_i,k_j$，使分数显式依赖 $i-j$。

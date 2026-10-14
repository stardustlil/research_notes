---
aliases:
  - 可学习绝对位置编码
  - Learned Positional Embedding
---

# Learned-PE（可学习绝对位置编码）

前置：[[Self-Attention#置换等变性]]

$$
\tilde X=X+P_{1:n},\qquad P\in\mathbb{R}^{n_{\max}\times d}\ \text{可学习}.
$$

## 动机

无 mask 的注意力满足 $f(\Pi X)=\Pi f(X)$。最直接的打破方式：给每个位置一个可学习向量，加到输入上。

## 定义

$p_i$ 为 $P$ 的第 $i$ 行（位置 $t_i=i-1$），$n\le n_{\max}$：

$$
P_{1:n}=\begin{bmatrix}p_1\\ \vdots\\ p_n\end{bmatrix}\in\mathbb{R}^{n\times d},
\qquad
\tilde x_i=x_i+p_i.
$$

$\tilde X$ 代替 $X$ 进入注意力。

## 推导

### 打破置换等变性

$P$ 不随输入行一起置换：一般地

$$
\Pi X+P_{1:n}\neq\Pi\left(X+P_{1:n}\right).
$$

### 位置以绝对形式进入分数

令 $G=W_QW_K^\top\in\mathbb{R}^{d\times d}$：

$$
\tilde q_i\tilde k_j^\top=(x_i+p_i)\,G\,(x_j+p_j)^\top
=x_iGx_j^\top+x_iGp_j^\top+p_iGx_j^\top+p_iGp_j^\top.
$$

位置只以 $p_i,p_j$ 出现；分数是否只依赖 $i-j$ 完全取决于学到的参数。[[RoPE]] 从结构上保证这一点。

## 代码

```python
import torch

def learned_pe(x, P):
    # x: [B, n, d]；P: [n_max, d]，可学习
    return x + P[: x.shape[-2]]                                     # X̃: [B, n, d]

# 验证：打破置换等变
x, P = torch.randn(1, 5, 8), torch.randn(16, 8)
perm = torch.tensor([1, 0, 2, 3, 4])
assert not torch.allclose(learned_pe(x[:, perm], P), learned_pe(x, P)[:, perm])
```

## 辨析

- $n>n_{\max}$ 时没有对应的行，无法计算。

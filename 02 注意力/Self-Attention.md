---
aliases:
  - 自注意力
---

# Self-Attention（自注意力）

$$
O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,\qquad Q=XW_Q,\ K=XW_K,\ V=XW_V.
$$

## 动机

让每个 token 的新表示按内容相关性从整个序列汇聚信息：权重由输入算出，而不是固定的。

## 定义

单头，省略 batch。$X\in\mathbb{R}^{n\times d}$，$W_Q,W_K\in\mathbb{R}^{d\times d_k}$，$W_V\in\mathbb{R}^{d\times d_v}$。

$$
\begin{aligned}
Q&=XW_Q\in\mathbb{R}^{n\times d_k},\quad K=XW_K\in\mathbb{R}^{n\times d_k},\quad V=XW_V\in\mathbb{R}^{n\times d_v},\\
S&=\frac{QK^\top}{\sqrt{d_k}}\in\mathbb{R}^{n\times n},\qquad S_{ij}=\frac{q_ik_j^\top}{\sqrt{d_k}},\\
A&=\operatorname{softmax}(S+M)\in\mathbb{R}^{n\times n},\\
O&=AV\in\mathbb{R}^{n\times d_v},\qquad o_i=\sum_{j=1}^{n}A_{ij}v_j.
\end{aligned}
$$

$M$ 为加性 mask（允许取 $0$，屏蔽取 $-\infty$），每行至少保留一个允许位置，否则 softmax 分母为 $0$。$A$ 每行非负且和为 $1$，$o_i$ 是可见 value 的加权平均。

## 推导

### 为什么要缩放

设 $q_i,k_j$ 各分量独立、均值 $0$、方差 $1$。每项 $q_{i,c}k_{j,c}$ 均值 $0$，方差 $\mathbb{E}[q_{i,c}^2]\,\mathbb{E}[k_{j,c}^2]=1$；$d_k$ 项独立相加：

$$
\operatorname{Var}(q_ik_j^\top)=d_k
\;\Longrightarrow\;
\operatorname{Var}\left(\frac{q_ik_j^\top}{\sqrt{d_k}}\right)=1.
$$

不缩放时分数标准差随 $\sqrt{d_k}$ 增长，softmax 趋于 one-hot，梯度趋于 $0$。

### 置换等变性

$\Pi$ 为置换矩阵，$\Pi^\top\Pi=I$。$M=0$ 时把输入换成 $\Pi X$：

$$
\begin{aligned}
S_\Pi&=\frac{(\Pi Q)(\Pi K)^\top}{\sqrt{d_k}}=\Pi S\Pi^\top,\\
A_\Pi&=\operatorname{softmax}(\Pi S\Pi^\top)=\Pi A\Pi^\top,\\
O_\Pi&=A_\Pi\,\Pi V=\Pi A\Pi^\top\Pi V=\Pi O.
\end{aligned}
$$

第二行成立，因为行 softmax 只在行内归一化：左乘 $\Pi$ 交换行，右乘 $\Pi^\top$ 交换行内元素，都与 softmax 可交换。

所以 $f(\Pi X)=\Pi f(X)$：注意力无法区分位置，需要[[位置编码]]。因果 mask 下 $\Pi M\Pi^\top\neq M$，等变性不成立。

## 代码

```python
import math
import torch

def self_attention(x, W_q, W_k, W_v, causal=False):
    # x: [B, n, d]；W_q, W_k: [d, d_k]；W_v: [d, d_v]
    q, k, v = x @ W_q, x @ W_k, x @ W_v                             # Q, K: [B, n, d_k]；V: [B, n, d_v]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])      # S: [B, n, n]
    if causal:
        n = x.shape[-2]
        mask = torch.ones(n, n, dtype=torch.bool).triu(1)          # True = 屏蔽 j > i
        scores = scores.masked_fill(mask, float("-inf"))           # S + M
    attn = torch.softmax(scores, dim=-1)                           # A: [B, n, n]
    return attn @ v                                                # O: [B, n, d_v]

# 验证
x = torch.randn(2, 5, 16)
W_q, W_k, W_v = torch.randn(16, 8), torch.randn(16, 8), torch.randn(16, 4)
for causal in (False, True):
    ref = torch.nn.functional.scaled_dot_product_attention(x @ W_q, x @ W_k, x @ W_v, is_causal=causal)
    assert torch.allclose(self_attention(x, W_q, W_k, W_v, causal), ref, atol=1e-5)
perm = torch.randperm(5)                                           # 置换等变
assert torch.allclose(self_attention(x[:, perm], W_q, W_k, W_v), self_attention(x, W_q, W_k, W_v)[:, perm], atol=1e-5)
```

## 辨析

- softmax 沿 key 方向（`dim=-1`）。
- 缩放用 $\sqrt{d_k}$，不是 $\sqrt d$。
- 本库 bool mask 的 `True` 表示屏蔽；`F.scaled_dot_product_attention` 的 `attn_mask` 中 `True` 表示允许。

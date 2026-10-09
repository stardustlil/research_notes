---
aliases:
  - 多头注意力
  - Multi-Head Attention
tags:
  - Transformer/注意力
---

# MHA（多头注意力）

上级：[[Attention|注意力]] ｜ 前置：[[Self-Attention]]

> [!abstract] 总结
> 用 $H$ 组独立的 $W_Q,W_K,W_V$ 并行做 $H$ 次[[Self-Attention|自注意力]]，把 $H$ 个输出拼接后，用输出投影矩阵 $W_O$ 融合回 $d$ 维。

## 动机

单个头只有一个权重矩阵 $A$，每个 token 只能形成一种「关注模式」。多个头在不同的投影子空间里各算一套 $A^{(h)}$，可以同时关注不同类型的关系。取 $d_k=d_v=d/H$ 时，总参数量和计算量与一个 $d_k=d$ 的单头大致相同。

## 符号

省略 batch 维度，输入 $X\in\mathbb{R}^{n\times d}$。局部符号：

| 符号 | 形状 | 含义 |
| --- | --- | --- |
| $W_Q^{(h)},W_K^{(h)}$ | $d\times d_k$ | 第 $h$ 个头的 query、key 投影 |
| $W_V^{(h)}$ | $d\times d_v$ | 第 $h$ 个头的 value 投影 |
| $W_O$ | $Hd_v\times d$ | 输出投影 |
| $O_{\text{cat}}$ | $n\times Hd_v$ | 所有头输出的拼接 |
| $W_Q^{\text{all}},W_K^{\text{all}},W_V^{\text{all}}$ | $d\times Hd_k$、$d\times Hd_k$、$d\times Hd_v$ | 各头投影按列拼成的大矩阵 |
| $W_O^{(h)}$ | $d_v\times d$ | $W_O$ 中第 $h$ 个头对应的行块 |

常用设置为 $d_k=d_v=d/H$（要求 $d$ 能被 $H$ 整除）。

## 定义

对第 $h$ 个头，$h=1,\dots,H$，按 [[Self-Attention#定义|Self-Attention 的定义]]计算：

$$
\begin{aligned}
Q^{(h)}&=XW_Q^{(h)}\in\mathbb{R}^{n\times d_k},\\
K^{(h)}&=XW_K^{(h)}\in\mathbb{R}^{n\times d_k},\\
V^{(h)}&=XW_V^{(h)}\in\mathbb{R}^{n\times d_v},\\
A^{(h)}&=\operatorname{softmax}\left(\frac{Q^{(h)}K^{(h)\top}}{\sqrt{d_k}}+M\right)\in\mathbb{R}^{n\times n},\\
O^{(h)}&=A^{(h)}V^{(h)}\in\mathbb{R}^{n\times d_v}.
\end{aligned}
$$

所有头共用同一个 mask $M$。拼接后做输出投影：

$$
\begin{aligned}
O_{\text{cat}}&=\operatorname{Concat}\left(O^{(1)},\dots,O^{(H)}\right)\in\mathbb{R}^{n\times Hd_v},\\
Y&=O_{\text{cat}}W_O\in\mathbb{R}^{n\times d}.
\end{aligned}
$$

## 性质与推导

### H 个小投影 = 一个大投影再切分

把各头的投影矩阵按列拼成一个大矩阵：

$$
W_Q^{\text{all}}=\left[W_Q^{(1)},\dots,W_Q^{(H)}\right]\in\mathbb{R}^{d\times Hd_k}.
$$

由分块矩阵乘法：

$$
XW_Q^{\text{all}}=\left[XW_Q^{(1)},\dots,XW_Q^{(H)}\right]=\left[Q^{(1)},\dots,Q^{(H)}\right].
$$

所以代码里只需一个 `nn.Linear(d, H*d_k)`，再把最后一维切成 $H$ 段，就得到全部 $Q^{(h)}$。这正是代码中 `split_heads` 做的事。$W_K^{\text{all}},W_V^{\text{all}}$ 同理。

### 输出投影 = 各头贡献之和

把 $W_O$ 按行切成 $H$ 块，$W_O^{(h)}\in\mathbb{R}^{d_v\times d}$：

$$
W_O=\begin{bmatrix}W_O^{(1)}\\ \vdots\\ W_O^{(H)}\end{bmatrix}
\quad\Longrightarrow\quad
Y=O_{\text{cat}}W_O=\sum_{h=1}^{H}O^{(h)}W_O^{(h)}.
$$

「融合各头」的含义就是：每个头的输出先各自投影回 $d$ 维，再相加。

### 参数量

取 $d_k=d_v=d/H$ 时，$W_Q^{\text{all}},W_K^{\text{all}},W_V^{\text{all}}$ 各为 $d\times d$，$W_O$ 为 $d\times d$，共 $4d^2$ 个参数（不含偏置），与头数 $H$ 无关。

## 代码实现

代码中 `head_dim` 对应 $d_k=d_v=d/H$；`q_proj.weight.T` 对应拼接后的 $W_Q^{\text{all}}$。

```python
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads                  # H
        self.head_dim = d_model // num_heads        # d_k = d_v = d / H

        self.q_proj = nn.Linear(d_model, d_model, bias=False)    # W_Q^all = [W_Q^(1), ..., W_Q^(H)]
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)  # W_O

    def forward(self, x, causal=False):
        B, n, d_model = x.shape

        def split_heads(proj):
            # [B, n, d] -> [B, n, H, d_k] -> [B, H, n, d_k]
            return proj.reshape(B, n, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(x))                          # Q^(h): [B, H, n, d_k]
        k = split_heads(self.k_proj(x))                          # K^(h): [B, H, n, d_k]
        v = split_heads(self.v_proj(x))                          # V^(h): [B, H, n, d_v]

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)   # S^(h): [B, H, n, n]

        if causal:
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))     # 所有头共用 M

        attn = torch.softmax(scores, dim=-1)                     # A^(h): [B, H, n, n]
        out = attn @ v                                           # O^(h): [B, H, n, d_v]

        out = out.transpose(1, 2).reshape(B, n, d_model)         # O_cat: [B, n, H*d_v]
        return self.out_proj(out)                                # Y: [B, n, d]
```

## 对比与易错点

- **切分顺序**：必须先 `reshape` 成 `[B, n, H, d_k]` 再 `transpose` 成 `[B, H, n, d_k]`；合并时先 `transpose` 回来再 `reshape`。直接 `reshape(B, H, n, d_k)` 会把不同 token 的特征混到一起。
- **不要漏掉输出投影 $W_O$**：没有它，各头的输出只是拼在一起，没有相互融合。
- **与 `nn.MultiheadAttention` 的区别**：官方实现默认带偏置，且默认输入形状为 `[n, B, d]`（`batch_first=False`）。

## 相关笔记

- [[Self-Attention|自注意力]]：单个头的完整计算与性质。
- [[RoPE|旋转位置编码]]：在每个头的 $Q^{(h)},K^{(h)}$ 上分别施加旋转。

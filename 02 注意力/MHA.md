---
aliases:
  - 多头注意力
  - Multi-Head Attention
---

# MHA（多头注意力）

前置：[[Self-Attention]]

$$
Y=\operatorname{Concat}\left(O^{(1)},\dots,O^{(H)}\right)W_O,\qquad
O^{(h)}=\operatorname{softmax}\left(\frac{Q^{(h)}K^{(h)\top}}{\sqrt{d_k}}+M\right)V^{(h)}.
$$

## 动机

单头只有一个 $A$，每个 token 只形成一种关注分布。$H$ 个头在不同投影子空间各算一个 $A^{(h)}$；取 $d_k=d_v=d/H$ 时参数量与单个 $d_k=d$ 的头相同。

## 定义

$X\in\mathbb{R}^{n\times d}$。对 $h=1,\dots,H$，$W_Q^{(h)},W_K^{(h)}\in\mathbb{R}^{d\times d_k}$，$W_V^{(h)}\in\mathbb{R}^{d\times d_v}$：

$$
\begin{aligned}
Q^{(h)}&=XW_Q^{(h)},\quad K^{(h)}=XW_K^{(h)},\quad V^{(h)}=XW_V^{(h)},\\
O^{(h)}&=\operatorname{softmax}\left(\frac{Q^{(h)}K^{(h)\top}}{\sqrt{d_k}}+M\right)V^{(h)}\in\mathbb{R}^{n\times d_v},\\
O_{\text{cat}}&=\operatorname{Concat}\left(O^{(1)},\dots,O^{(H)}\right)\in\mathbb{R}^{n\times Hd_v},\\
Y&=O_{\text{cat}}W_O\in\mathbb{R}^{n\times d},\qquad W_O\in\mathbb{R}^{Hd_v\times d}.
\end{aligned}
$$

所有头共用 $M$。

## 推导

### 多个小投影等于一个大投影再切分

记 $W_Q^{\text{all}}=\left[W_Q^{(1)},\dots,W_Q^{(H)}\right]\in\mathbb{R}^{d\times Hd_k}$，由分块乘法：

$$
XW_Q^{\text{all}}=\left[XW_Q^{(1)},\dots,XW_Q^{(H)}\right]=\left[Q^{(1)},\dots,Q^{(H)}\right].
$$

所以一次矩阵乘法后把最后一维切成 $H$ 段即得各头的 $Q^{(h)}$；$K,V$ 同理。

### 输出投影等于各头贡献之和

把 $W_O$ 按行切成 $W_O^{(h)}\in\mathbb{R}^{d_v\times d}$：

$$
Y=\left[O^{(1)},\dots,O^{(H)}\right]\begin{bmatrix}W_O^{(1)}\\ \vdots\\ W_O^{(H)}\end{bmatrix}=\sum_{h=1}^{H}O^{(h)}W_O^{(h)}.
$$

### 参数量与头数无关

$d_k=d_v=d/H$ 时 $W_Q^{\text{all}},W_K^{\text{all}},W_V^{\text{all}},W_O$ 均为 $d\times d$，共 $4d^2$。

## 代码

```python
import math
import torch

def mha(x, W_q, W_k, W_v, W_o, H, causal=False):
    # x: [B, n, d]；W_q, W_k, W_v, W_o: [d, d]（W^all 与 W_O），d_k = d_v = d / H
    B, n, d = x.shape
    split = lambda z: z.view(B, n, H, d // H).transpose(1, 2)      # [B, n, d] -> [B, H, n, d_k]
    q, k, v = split(x @ W_q), split(x @ W_k), split(x @ W_v)       # Q^(h), K^(h), V^(h)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d // H)           # [B, H, n, n]
    if causal:
        scores = scores.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), float("-inf"))
    out = torch.softmax(scores, dim=-1) @ v                        # O^(h): [B, H, n, d_v]
    return out.transpose(1, 2).reshape(B, n, d) @ W_o              # Y = O_cat W_O: [B, n, d]

# 验证：等于逐头做 Self-Attention 再按 W_O 行块求和
# 依赖：[[Self-Attention]] 中的 self_attention
B, n, d, H = 2, 5, 16, 4
x = torch.randn(B, n, d)
W_q, W_k, W_v, W_o = (torch.randn(d, d) for _ in range(4))
c = d // H
ref = sum(self_attention(x, W_q[:, h*c:(h+1)*c], W_k[:, h*c:(h+1)*c], W_v[:, h*c:(h+1)*c], causal=True)
          @ W_o[h*c:(h+1)*c] for h in range(H))
assert torch.allclose(mha(x, W_q, W_k, W_v, W_o, H, causal=True), ref, atol=1e-4)
```

## 辨析

- 切分必须先 `view(B, n, H, d_k)` 再 `transpose`；直接 `view(B, H, n, d_k)` 会把不同 token 的特征混在一起。合并时顺序相反。
- `nn.Linear(d_in, d_out)` 的 `weight` 形状是 $d_{\text{out}}\times d_{\text{in}}$，对应本库的 $W^\top$。

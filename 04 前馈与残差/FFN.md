---
aliases:
  - 前馈网络
  - Feed-Forward Network
  - MLP
---

# FFN（前馈网络）

前置：[[Self-Attention]]

$$
\operatorname{FFN}(X)=\phi(XW_1)W_2,\qquad W_1\in\mathbb{R}^{d\times d_{\text{ff}}},\quad W_2\in\mathbb{R}^{d_{\text{ff}}\times d}.
$$

## 动机

[[Self-Attention]] 的输出 $O=AV$ 对 $V$ 是线性的：在 $A$ 固定时，每个 token 只得到 $v_j$ 的凸组合，特征维上没有非线性变换。FFN 对每个 token 单独做一次「升维、逐元素非线性、降维」。

## 定义

$X\in\mathbb{R}^{n\times d}$，$\phi$ 为逐元素激活，通常 $d_{\text{ff}}=4d$：

$$
\begin{aligned}
Z&=\phi(XW_1)\in\mathbb{R}^{n\times d_{\text{ff}}},\\
\operatorname{FFN}(X)&=ZW_2\in\mathbb{R}^{n\times d}.
\end{aligned}
$$

$\phi$ 的两种取法：

$$
\operatorname{ReLU}(z)=\max(0,z),\qquad
\operatorname{GELU}(z)=z\,\Phi(z).
$$

## 推导

### 逐 token 独立

$XW_1$ 的第 $i$ 行是 $x_iW_1$，$\phi$ 逐元素作用，所以

$$
\operatorname{FFN}(X)_{i,:}=\phi(x_iW_1)W_2 .
$$

第 $i$ 行输出只依赖 $x_i$。对置换矩阵 $\Pi$，$\phi(\Pi XW_1)=\Pi\phi(XW_1)$，故 $\operatorname{FFN}(\Pi X)=\Pi\operatorname{FFN}(X)$。token 之间的信息交换只发生在注意力里。

### 没有激活就退化为一个线性层

$\phi(z)=z$ 时 $\operatorname{FFN}(X)=X(W_1W_2)$，$W_1W_2\in\mathbb{R}^{d\times d}$ 且 $\operatorname{rank}(W_1W_2)\le d$，升维到 $d_{\text{ff}}$ 没有增加表达能力。

### 隐藏单元之和

记 $u_l\in\mathbb{R}^{d\times1}$ 为 $W_1$ 第 $l$ 列，$w_l\in\mathbb{R}^{1\times d}$ 为 $W_2$ 第 $l$ 行，按分块乘法：

$$
\operatorname{FFN}(x_i)=\sum_{l=1}^{d_{\text{ff}}}\phi(x_iu_l)\,w_l .
$$

每个隐藏单元用 $x_iu_l$ 打分，再把 $w_l$ 按分数加到输出上：$u_l$ 是键，$w_l$ 是值，与注意力的区别是键值是参数而非由输入投影得到，且权重是 $\phi$ 而非 softmax。

### ReLU 时分段线性且正齐次

记 $D(x_i)=\operatorname{diag}\left([x_iu_1>0],\dots,[x_iu_{d_{\text{ff}}}>0]\right)$，则

$$
\operatorname{FFN}(x_i)=x_iW_1D(x_i)W_2 .
$$

在 $D$ 不变的区域内 FFN 是线性映射。对 $a>0$，$D(ax_i)=D(x_i)$，所以 $\operatorname{FFN}(ax_i)=a\operatorname{FFN}(x_i)$。

### 参数量是注意力的两倍

$W_1,W_2$ 共 $2dd_{\text{ff}}$ 个参数，$d_{\text{ff}}=4d$ 时为 $8d^2$，是 [[MHA#参数量与头数无关]] 中 $4d^2$ 的两倍。每个 token 的乘加次数同样为 $2dd_{\text{ff}}$。

## 代码

```python
import torch

def ffn(x, W_1, W_2, act=torch.relu):
    # x: [B, n, d]；W_1: [d, d_ff]，W_2: [d_ff, d]
    z = act(x @ W_1)                                                # Z: [B, n, d_ff]
    return z @ W_2                                                  # [B, n, d]

# 验证
B, n, d, d_ff = 2, 5, 8, 32
x = torch.randn(B, n, d)
W_1, W_2 = torch.randn(d, d_ff), torch.randn(d_ff, d)
y = ffn(x, W_1, W_2)
perm = torch.randperm(n)
assert torch.allclose(ffn(x[:, perm], W_1, W_2), y[:, perm])                          # 逐 token
assert torch.allclose(ffn(x, W_1, W_2, act=lambda z: z), x @ (W_1 @ W_2), atol=1e-4)   # 无激活即线性层
assert torch.allclose(sum(torch.relu(x @ W_1[:, l:l+1]) * W_2[l] for l in range(d_ff)),
                      y, atol=1e-4)                                                    # 隐藏单元之和
assert torch.allclose(ffn(3 * x, W_1, W_2), 3 * y, atol=1e-4)                          # 正齐次
```

## 辨析

- 原论文每个线性层带偏置：$\phi(XW_1+\cdots)W_2+\cdots$。偏置不影响以上推导，LLaMA 等模型已去掉，本库省略。
- `torch.nn.functional.gelu` 默认用 $\Phi$ 的精确形式（误差函数）；GPT-2 用 `approximate="tanh"` 的近似，两者数值不同。

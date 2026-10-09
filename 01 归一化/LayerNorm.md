---
aliases:
  - 层归一化
tags:
  - Transformer/归一化
---

# LayerNorm（层归一化）

上级：[[Normalization|归一化]]

> [!abstract] 总结
> 对每个 token 的特征向量单独做标准化（减均值、除标准差），再做逐维仿射变换；统计量只来自这个 token 自身，训练与推理完全一致。

## 动机

[[BatchNorm]] 的统计量依赖 batch：batch 小时噪声大，且训练与推理行为不同。在 Transformer 中，序列长度可变、自回归推理时一次只处理一个新 token，更适合**每个 token 自己归一化自己**。LayerNorm 正是如此：统计集合是单个 token 的 $d$ 个特征。

## 符号

Transformer 中的输入为 $X\in\mathbb{R}^{B\times n\times d}$。LayerNorm 对其中 $B\cdot n$ 个 token 各自独立计算，因此只需讨论第 $i$ 个 token 的行向量：

$$
x_i=[x_{i,1},\dots,x_{i,d}]\in\mathbb{R}^{1\times d}.
$$

此时统计集合 $\mathcal{S}=\{x_{i,1},\dots,x_{i,d}\}$，$m=d$。可学习参数 $\gamma,\beta\in\mathbb{R}^{1\times d}$，所有 token 共享。$\mu_i,\sigma_i^2$ 是第 $i$ 个 token 的均值与方差。

## 定义

$$
\begin{aligned}
\mu_i&=\frac{1}{d}\sum_{c=1}^{d}x_{i,c},\\
\sigma_i^2&=\frac{1}{d}\sum_{c=1}^{d}(x_{i,c}-\mu_i)^2,\\
\hat x_{i,c}&=\frac{x_{i,c}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}},\\
y_{i,c}&=\gamma_c\hat x_{i,c}+\beta_c.
\end{aligned}
$$

写成向量形式：

$$
y_i=\operatorname{LayerNorm}(x_i)=\gamma\odot\hat x_i+\beta\in\mathbb{R}^{1\times d}.
$$

对整个输入，均值和方差的形状为 `[B, n, 1]`：每个 token 一个。

## 性质与推导

### 平移与缩放不变性

忽略 $\epsilon$ 时，对任意 $a>0$ 和标量 $e$，令 $x'_{i,c}=ax_{i,c}+e$，则

$$
\begin{aligned}
\mu'_i&=a\mu_i+e,\qquad \sigma'^2_i=a^2\sigma_i^2,\\
\hat x'_{i,c}&=\frac{ax_{i,c}+e-(a\mu_i+e)}{a\sigma_i}=\frac{x_{i,c}-\mu_i}{\sigma_i}=\hat x_{i,c}.
\end{aligned}
$$

所以 LayerNorm 的输出不受输入整体平移和整体缩放的影响。

### 几何解释

忽略 $\epsilon$ 时，$\hat x_i$ 满足两个约束：

$$
\sum_{c=1}^{d}\hat x_{i,c}=0,
\qquad
\lVert\hat x_i\rVert^2=\sum_{c=1}^{d}\frac{(x_{i,c}-\mu_i)^2}{\sigma_i^2}=\frac{d\sigma_i^2}{\sigma_i^2}=d.
$$

即：先把 $x_i$ 投影到「各分量之和为 $0$」的超平面上，再缩放到半径为 $\sqrt d$ 的球面上。[[RMSNorm]] 只保留后一步。

## 代码实现

`normalized_shape` 可以是整数（取 $d$）或元组；统计集合是最后 `len(normalized_shape)` 维的全部元素。

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)

        self.gamma = nn.Parameter(torch.ones(*self.normalized_shape))   # γ: [d]
        self.beta = nn.Parameter(torch.zeros(*self.normalized_shape))   # β: [d]

    def forward(self, x):
        # x: [B, n, d]
        dims = tuple(range(-len(self.normalized_shape), 0))             # 统计集合所在的维度
        mu = x.mean(dim=dims, keepdim=True)                             # μ_i: [B, n, 1]（normalized_shape = d 时）
        var = x.var(dim=dims, unbiased=False, keepdim=True)             # σ_i²: [B, n, 1]

        x_hat = (x - mu) / torch.sqrt(var + self.eps)                   # x̂: [B, n, d]
        return self.gamma * x_hat + self.beta                           # y:  [B, n, d]
```

## 对比与易错点

- **方差是有偏的**（除以 $d$），与 `nn.LayerNorm` 一致；`torch.var` 默认是无偏的，必须写 `unbiased=False`。
- **$\epsilon$ 在根号内**。
- **训练与推理相同**：没有滑动统计量，也不需要区分 `train()` / `eval()`。
- 本实现与 `nn.LayerNorm(normalized_shape, eps)` 一致，`weight`、`bias` 对应 $\gamma$、$\beta$。

## 相关笔记

- [[BatchNorm|批归一化]]：对比「按通道跨 batch 统计」与「按 token 统计」。
- [[RMSNorm|均方根归一化]]：去掉减均值和 $\beta$，只保留缩放到球面这一步。

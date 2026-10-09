---
aliases:
  - 均方根归一化
tags:
  - Transformer/归一化
---

# RMSNorm（均方根归一化）

上级：[[Normalization|归一化]] ｜ 前置：[[LayerNorm]]

> [!abstract] 总结
> 在 LayerNorm 的基础上去掉「减均值」和偏置 $\beta$，只用均方根（RMS）缩放每个 token 的特征向量。

## 动机

[[LayerNorm]] 做两件事：去均值（平移不变）和除以标准差（缩放不变）。RMSNorm 认为起主要作用的是缩放这一步，于是只保留它：不计算均值，也不学习 $\beta$。这样计算步骤更少、参数更少；实际速度取决于实现和硬件。

## 符号

与 [[LayerNorm#符号|LayerNorm]] 相同：只讨论第 $i$ 个 token 的行向量 $x_i\in\mathbb{R}^{1\times d}$，统计集合为它的 $d$ 个分量 $x_{i,1},\dots,x_{i,d}$。可学习参数只有 $\gamma\in\mathbb{R}^{1\times d}$。

## 定义

$$
\begin{aligned}
\operatorname{RMS}(x_i)&=\sqrt{\frac{1}{d}\sum_{c=1}^{d}x_{i,c}^2+\epsilon},\\
y_{i,c}&=\gamma_c\,\frac{x_{i,c}}{\operatorname{RMS}(x_i)}.
\end{aligned}
$$

写成向量形式：

$$
y_i=\operatorname{RMSNorm}(x_i)=\gamma\odot\frac{x_i}{\operatorname{RMS}(x_i)}\in\mathbb{R}^{1\times d}.
$$

## 性质与推导

### 与 LayerNorm 的关系

若 $\mu_i=0$，则 $\sigma_i^2=\dfrac1d\sum_c(x_{i,c}-0)^2=\dfrac1d\sum_c x_{i,c}^2$，于是 $\sqrt{\sigma_i^2+\epsilon}=\operatorname{RMS}(x_i)$。再令 $\beta=0$，LayerNorm 就退化为 RMSNorm。

### 只有缩放不变性

忽略 $\epsilon$ 时，对 $a>0$：

$$
\operatorname{RMS}(ax_i)=a\operatorname{RMS}(x_i)
\quad\Longrightarrow\quad
\frac{ax_i}{\operatorname{RMS}(ax_i)}=\frac{x_i}{\operatorname{RMS}(x_i)}.
$$

但对整体平移 $x_{i,c}\mapsto x_{i,c}+e$ 不具有不变性，因为没有减去均值。

### 几何解释

忽略 $\epsilon$ 时：

$$
\left\lVert\frac{x_i}{\operatorname{RMS}(x_i)}\right\rVert^2
=\frac{\sum_c x_{i,c}^2}{\frac1d\sum_c x_{i,c}^2}=d.
$$

即把 $x_i$ 沿原方向缩放到半径为 $\sqrt d$ 的球面上；与 [[LayerNorm#几何解释|LayerNorm]] 相比，少了投影到超平面这一步。

## 代码实现

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))                         # γ: [d]

    def norm(self, x):
        # x / RMS(x)；rsqrt(a) = 1 / sqrt(a)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # x: [B, n, d]；先转 float32 计算，再转回原 dtype
        return (self.gamma * self.norm(x.float())).type_as(x)           # y: [B, n, d]
```

## 对比与易错点

- **$\epsilon$ 的位置因实现而异**：本库与 `nn.RMSNorm` 都把 $\epsilon$ 放在根号内；也有实现写成 $x/(\operatorname{RMS}+\epsilon)$，数值上略有差别。
- **低精度训练**：平方求均值在 fp16 / bf16 下容易溢出或损失精度，所以代码先转成 `float32` 计算。
- 输入为 float32 时，本实现与 `nn.RMSNorm(d, eps=1e-5)`（PyTorch ≥ 2.4）一致，`weight` 对应 $\gamma$；低精度输入时本实现先升到 float32，结果可能略有差别。

## 相关笔记

- [[LayerNorm#定义|LayerNorm 的定义]]：对比是否减均值、是否有 $\beta$。
- [[Normalization#快速对比|归一化对比表]]：三种归一化的横向对比。

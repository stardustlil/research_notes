---
aliases:
  - 批归一化
---

# BatchNorm（批归一化）

$$
y=\gamma_c\frac{x-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}+\beta_c,\qquad x\in\mathcal{S}_c.
$$

## 动机

把每个通道的值标准化到均值 $0$、方差约 $1$，再用 $\gamma,\beta$ 恢复所需的尺度与偏移，使各层输入的数值尺度不随前层参数漂移。

## 定义

输入 $X\in\mathbb{R}^{B\times C\times H_{\text{img}}\times W_{\text{img}}}$。通道 $c$ 的统计集合 $\mathcal{S}_c$ 是该通道在所有样本、所有空间位置上的元素，$m=\lvert\mathcal{S}_c\rvert=BH_{\text{img}}W_{\text{img}}$。

训练时：

$$
\begin{aligned}
\mu_c&=\frac{1}{m}\sum_{x\in\mathcal{S}_c}x,\\
\sigma_c^2&=\frac{1}{m}\sum_{x\in\mathcal{S}_c}(x-\mu_c)^2,\\
\hat x&=\frac{x-\mu_c}{\sqrt{\sigma_c^2+\epsilon}},\\
y&=\gamma_c\hat x+\beta_c,
\end{aligned}
$$

$\gamma,\beta\in\mathbb{R}^{C}$。同时以动量 $\kappa$ 更新滑动统计量：

$$
\begin{aligned}
\mu_c^{\text{run}}&\leftarrow(1-\kappa)\,\mu_c^{\text{run}}+\kappa\,\mu_c,\\
(\sigma_c^2)^{\text{run}}&\leftarrow(1-\kappa)\,(\sigma_c^2)^{\text{run}}+\kappa\,\frac{m}{m-1}\sigma_c^2.
\end{aligned}
$$

推理时用 $\mu_c^{\text{run}},(\sigma_c^2)^{\text{run}}$ 代替 $\mu_c,\sigma_c^2$。

## 推导

### 标准化后均值为零、方差约为一

$$
\frac{1}{m}\sum_{x\in\mathcal{S}_c}\hat x
=\frac{1}{\sqrt{\sigma_c^2+\epsilon}}\left(\frac{1}{m}\sum_{x\in\mathcal{S}_c}x-\mu_c\right)=0,
\qquad
\frac{1}{m}\sum_{x\in\mathcal{S}_c}\hat x^2=\frac{\sigma_c^2}{\sigma_c^2+\epsilon}\approx1.
$$

### 仿射变换可以恢复恒等映射

取 $\gamma_c=\sqrt{\sigma_c^2+\epsilon}$、$\beta_c=\mu_c$：

$$
y=\sqrt{\sigma_c^2+\epsilon}\cdot\frac{x-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}+\mu_c=x.
$$

所以加入 BatchNorm 不缩小可表示的函数集合。

### 滑动方差用无偏估计

$\mathbb{E}[\sigma_c^2]=\frac{m-1}{m}\operatorname{Var}(x)$，乘以 $\frac{m}{m-1}$ 得到总体方差的无偏估计。当前 batch 用有偏方差标准化，累积给推理用的是无偏方差；这要求 $m>1$。

## 代码

```python
import torch

def batch_norm(x, gamma, beta, run_mu, run_var, training, kappa=0.1, eps=1e-5):
    # x: [B, C, H_img, W_img]
    if training:
        m = x.numel() // x.shape[1]                                     # |S_c|
        mu = x.mean(dim=(0, 2, 3))                                      # μ_c: [C]
        var = x.var(dim=(0, 2, 3), unbiased=False)                      # σ_c²: [C]
        run_mu.mul_(1 - kappa).add_(kappa * mu)
        run_var.mul_(1 - kappa).add_(kappa * var * m / (m - 1))
    else:
        mu, var = run_mu, run_var
    x_hat = (x - mu[:, None, None]) / torch.sqrt(var[:, None, None] + eps)
    return gamma[:, None, None] * x_hat + beta[:, None, None]

# 验证
x = torch.randn(4, 3, 5, 5)
gamma, beta = torch.randn(3), torch.randn(3)
rm, rv = torch.zeros(3), torch.ones(3)
y = batch_norm(x, gamma, beta, rm, rv, training=True)
rm2, rv2 = torch.zeros(3), torch.ones(3)
y2 = torch.nn.functional.batch_norm(x, rm2, rv2, gamma, beta, training=True)
assert torch.allclose(y, y2, atol=1e-5) and torch.allclose(rv, rv2)
```

## 辨析

- $\gamma,\beta$ 由梯度更新；$\mu^{\text{run}},(\sigma^2)^{\text{run}}$ 由滑动平均更新，不是参数。
- 统计量依赖 batch：$m$ 小则噪声大，训练与推理的计算不同。[[LayerNorm]] 把 $\mathcal{S}$ 换成单个 token 的特征，消除这一依赖。

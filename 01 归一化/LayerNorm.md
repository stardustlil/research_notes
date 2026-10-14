---
aliases:
  - 层归一化
---

# LayerNorm（层归一化）

前置：[[BatchNorm]]

$$
y_i=\gamma\odot\frac{x_i-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}+\beta.
$$

## 动机

[[BatchNorm]] 的统计量依赖 batch，训练与推理计算不同。把统计集合换成单个 token 的 $d$ 个特征，每个 token 只用自己归一化，训练与推理一致。

## 定义

对第 $i$ 个 token $x_i\in\mathbb{R}^{1\times d}$，$\mathcal{S}=\{x_{i,1},\dots,x_{i,d}\}$，$m=d$；$\gamma,\beta\in\mathbb{R}^{1\times d}$ 由所有 token 共享：

$$
\begin{aligned}
\mu_i&=\frac{1}{d}\sum_{c=1}^{d}x_{i,c},\\
\sigma_i^2&=\frac{1}{d}\sum_{c=1}^{d}(x_{i,c}-\mu_i)^2,\\
\hat x_{i,c}&=\frac{x_{i,c}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}},\\
y_{i,c}&=\gamma_c\hat x_{i,c}+\beta_c.
\end{aligned}
$$

## 推导

### 平移与缩放不变

忽略 $\epsilon$，对 $a>0$ 和标量 $e$，令 $x'_{i,c}=ax_{i,c}+e$：

$$
\mu'_i=a\mu_i+e,\qquad \sigma'_i=a\sigma_i,\qquad
\hat x'_{i,c}=\frac{ax_{i,c}+e-(a\mu_i+e)}{a\sigma_i}=\hat x_{i,c}.
$$

### 几何意义

忽略 $\epsilon$：

$$
\sum_{c=1}^{d}\hat x_{i,c}=0,
\qquad
\lVert\hat x_i\rVert^2=\sum_{c=1}^{d}\frac{(x_{i,c}-\mu_i)^2}{\sigma_i^2}=\frac{d\sigma_i^2}{\sigma_i^2}=d.
$$

即先投影到超平面 $\sum_c x_c=0$，再缩放到半径 $\sqrt d$ 的球面。[[RMSNorm]] 只保留第二步。

## 代码

```python
import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    # x: [B, n, d]
    mu = x.mean(-1, keepdim=True)                                   # μ_i: [B, n, 1]
    var = x.var(-1, unbiased=False, keepdim=True)                   # σ_i²: [B, n, 1]
    return gamma * (x - mu) / torch.sqrt(var + eps) + beta          # y: [B, n, d]

# 验证
x, gamma, beta = torch.randn(2, 4, 8), torch.randn(8), torch.randn(8)
assert torch.allclose(layer_norm(x, gamma, beta),
                      torch.nn.functional.layer_norm(x, (8,), gamma, beta), atol=1e-5)
assert torch.allclose(layer_norm(3 * x + 2, gamma, beta), layer_norm(x, gamma, beta), atol=1e-4)  # 平移缩放不变
```

## 辨析

- 方差除以 $d$（有偏）；`torch.var` 默认无偏，需 `unbiased=False`。

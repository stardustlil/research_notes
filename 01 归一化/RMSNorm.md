---
aliases:
  - 均方根归一化
---

# RMSNorm（均方根归一化）

前置：[[LayerNorm]]

$$
y_i=\gamma\odot\frac{x_i}{\operatorname{RMS}(x_i)},\qquad
\operatorname{RMS}(x_i)=\sqrt{\frac{1}{d}\sum_{c=1}^{d}x_{i,c}^2+\epsilon}.
$$

## 动机

[[LayerNorm]] 做两步：减均值（平移不变）和除标准差（缩放不变）。RMSNorm 只保留缩放这一步，不算均值，不学 $\beta$。

## 定义

符号同 [[LayerNorm#定义]]，可学习参数只有 $\gamma\in\mathbb{R}^{1\times d}$：

$$
y_{i,c}=\gamma_c\,\frac{x_{i,c}}{\operatorname{RMS}(x_i)}.
$$

## 推导

### LayerNorm 在零均值时退化为 RMSNorm

$\mu_i=0$ 时 $\sigma_i^2=\frac1d\sum_c x_{i,c}^2$，于是 $\sqrt{\sigma_i^2+\epsilon}=\operatorname{RMS}(x_i)$；再令 $\beta=0$ 即得 RMSNorm。

### 只有缩放不变

忽略 $\epsilon$，对 $a>0$：

$$
\operatorname{RMS}(ax_i)=a\operatorname{RMS}(x_i)
\;\Longrightarrow\;
\frac{ax_i}{\operatorname{RMS}(ax_i)}=\frac{x_i}{\operatorname{RMS}(x_i)}.
$$

平移 $x_{i,c}\mapsto x_{i,c}+e$ 会改变 $\operatorname{RMS}$ 与方向，不具有不变性。

### 几何意义

忽略 $\epsilon$：

$$
\left\lVert\frac{x_i}{\operatorname{RMS}(x_i)}\right\rVert^2
=\frac{\sum_c x_{i,c}^2}{\frac1d\sum_c x_{i,c}^2}=d.
$$

沿原方向缩放到半径 $\sqrt d$ 的球面，比 [[LayerNorm#几何意义]] 少了投影到超平面。

## 代码

```python
import torch

def rms_norm(x, gamma, eps=1e-5):
    # x: [B, n, d]
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)        # RMS(x_i): [B, n, 1]
    return gamma * x / rms                                          # y: [B, n, d]

# 验证
x, gamma = torch.randn(2, 4, 8), torch.randn(8)
assert torch.allclose(rms_norm(x, gamma),
                      torch.nn.functional.rms_norm(x, (8,), gamma, eps=1e-5), atol=1e-5)
assert torch.allclose(rms_norm(3 * x, gamma), rms_norm(x, gamma), atol=1e-4)   # 缩放不变
```

## 辨析

- $\epsilon$ 的位置因实现而异：本库与 PyTorch 放在根号内；也有实现写成 $x/(\operatorname{RMS}+\epsilon)$。

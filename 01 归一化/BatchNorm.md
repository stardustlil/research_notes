---
aliases:
  - 批归一化
tags:
  - Transformer/归一化
---

# BatchNorm（批归一化）

上级：[[Normalization|归一化]]

> [!abstract] 总结
> 对每个通道，用当前 batch 中该通道的所有元素算均值和方差做标准化，再做可学习的仿射变换；推理时改用训练中累积的滑动统计量。

## 动机

深层网络中，每一层输入的数值尺度会随着前面层参数的更新而变化，训练容易不稳定。BatchNorm 把每个通道的值标准化到均值 $0$、方差约为 $1$，再用可学习参数 $\gamma,\beta$ 恢复必要的尺度和偏移，从而稳定训练。

## 符号

输入为图像特征张量：

$$
X\in\mathbb{R}^{B\times C\times H_{\text{img}}\times W_{\text{img}}}.
$$

局部符号：

| 符号 | 含义 |
| --- | --- |
| $\mathcal{S}_c$ | 统计集合：通道 $c$ 在所有样本、所有空间位置上的元素 |
| $m$ | $\lvert\mathcal{S}_c\rvert=B\,H_{\text{img}}W_{\text{img}}$ |
| $x$ | $\mathcal{S}_c$ 中的任意一个元素 |
| $\mu_c^{\text{run}},(\sigma_c^2)^{\text{run}}$ | 通道 $c$ 的滑动均值、滑动方差 |

## 定义

**训练时**，对每个通道 $c=1,\dots,C$，在统计集合上计算均值和有偏方差：

$$
\begin{aligned}
\mu_c&=\frac{1}{m}\sum_{x\in\mathcal{S}_c}x,\\
\sigma_c^2&=\frac{1}{m}\sum_{x\in\mathcal{S}_c}(x-\mu_c)^2.
\end{aligned}
$$

对通道 $c$ 中的每个元素 $x$ 标准化，再做仿射变换：

$$
\begin{aligned}
\hat x&=\frac{x-\mu_c}{\sqrt{\sigma_c^2+\epsilon}},\\
y&=\gamma_c\hat x+\beta_c.
\end{aligned}
$$

其中 $\gamma,\beta\in\mathbb{R}^{C}$ 是可学习参数，由梯度下降更新。

同时用动量 $\kappa$ 更新滑动统计量（PyTorch 默认 $\kappa=0.1$）：

$$
\begin{aligned}
\mu_c^{\text{run}}&\leftarrow(1-\kappa)\,\mu_c^{\text{run}}+\kappa\,\mu_c,\\
(\sigma_c^2)^{\text{run}}&\leftarrow(1-\kappa)\,(\sigma_c^2)^{\text{run}}+\kappa\,\frac{m}{m-1}\sigma_c^2.
\end{aligned}
$$

**推理时**，把上面的 $\mu_c,\sigma_c^2$ 换成 $\mu_c^{\text{run}},(\sigma_c^2)^{\text{run}}$，其余公式不变。

## 性质与推导

### 标准化后的统计量

在训练时，对同一个通道，$\hat x$ 的均值为 $0$：

$$
\frac{1}{m}\sum_{x\in\mathcal{S}_c}\hat x
=\frac{1}{\sqrt{\sigma_c^2+\epsilon}}\left(\frac{1}{m}\sum_{x\in\mathcal{S}_c}x-\mu_c\right)
=0.
$$

方差为 $\dfrac{\sigma_c^2}{\sigma_c^2+\epsilon}$，当 $\epsilon\ll\sigma_c^2$ 时约等于 $1$。

### 仿射变换不会损失表达能力

如果标准化限制了网络的表达，网络可以学到 $\gamma_c=\sqrt{\sigma_c^2+\epsilon}$、$\beta_c=\mu_c$，此时

$$
y=\sqrt{\sigma_c^2+\epsilon}\cdot\frac{x-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}+\mu_c=x,
$$

即恢复恒等映射。因此标准化 + 仿射变换至少不比不做归一化差。

### 为什么滑动方差用无偏估计

$\sigma_c^2$ 除以 $m$，是对总体方差的有偏估计；乘以 $\dfrac{m}{m-1}$ 后等于除以 $m-1$ 的无偏估计。归一化当前 batch 时使用有偏方差，为推理积累总体方差时使用无偏方差，这是 PyTorch 的约定。这也要求训练时 $m>1$。

## 代码实现

`num_features` 对应 $C$，`momentum` 对应 $\kappa$。

```python
import torch
import torch.nn as nn

class MyBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()                 # self.training 默认为 True
        self.eps = eps
        self.momentum = momentum           # κ

        self.gamma = nn.Parameter(torch.ones(num_features))   # γ: [C]
        self.beta = nn.Parameter(torch.zeros(num_features))   # β: [C]

        # 滑动统计量：不是参数，不参与梯度下降
        self.register_buffer("running_mean", torch.zeros(num_features))  # μ^run: [C]
        self.register_buffer("running_var", torch.ones(num_features))    # (σ²)^run: [C]

    def forward(self, x):
        # x: [B, C, H_img, W_img]
        B, C, H_img, W_img = x.shape
        if self.training:
            m = B * H_img * W_img          # |S_c|
            if m <= 1:
                raise ValueError("训练时每个通道至少需要两个统计元素")
            mu = x.mean(dim=(0, 2, 3), keepdim=True)                  # μ_c: [1, C, 1, 1]
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)  # σ_c²: [1, C, 1, 1]

            with torch.no_grad():
                unbiased_var = var * m / (m - 1)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mu.view(-1))
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased_var.view(-1))
        else:
            mu = self.running_mean.view(1, C, 1, 1)
            var = self.running_var.view(1, C, 1, 1)

        x_hat = (x - mu) / torch.sqrt(var + self.eps)                  # x̂
        y = self.gamma.view(1, C, 1, 1) * x_hat + self.beta.view(1, C, 1, 1)
        return y
```

## 对比与易错点

- **训练与推理行为不同**：训练用当前 batch 的统计量，推理用滑动统计量。忘记调用 `model.eval()` 会让推理结果依赖 batch 的组成。
- **依赖 batch 大小**：batch 很小时 $\mu_c,\sigma_c^2$ 噪声大；$m=1$ 时方差无意义。这是 Transformer 通常改用 [[LayerNorm]] 的原因之一。
- **$\gamma,\beta$ 与滑动统计量的区别**：前者是参数（梯度下降更新），后者是 buffer（滑动平均更新）。
- 本实现与 `nn.BatchNorm2d` 的默认行为一致（`affine=True`、`track_running_stats=True`），其中 `weight`、`bias` 对应 $\gamma$、$\beta$。

## 相关笔记

- [[LayerNorm|层归一化]]：统计集合从「一个通道的整个 batch」换成「一个 token 的所有特征」，不再依赖 batch。
- [[Normalization#快速对比|归一化对比表]]：三种归一化的统计集合与参数对比。

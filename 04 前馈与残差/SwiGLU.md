---
aliases:
  - 门控前馈网络
  - GLU
  - GEGLU
---

# SwiGLU（门控前馈网络）

前置：[[FFN]]

$$
\operatorname{SwiGLU}(X)=\left(\operatorname{SiLU}(XW_1)\odot XW_3\right)W_2,\qquad
\operatorname{SiLU}(z)=z\operatorname{sigmoid}(z).
$$

## 动机

[[FFN#隐藏单元之和]] 中第 $l$ 个单元的输出 $\phi(x_iu_l)$ 只是一个线性形式过激活。GLU 用两个独立投影，一路过激活作门，一路保持线性，逐元素相乘，使隐藏单元成为两个线性形式的乘积。

## 定义

$X\in\mathbb{R}^{n\times d}$，$W_1,W_3\in\mathbb{R}^{d\times d_{\text{ff}}}$，$W_2\in\mathbb{R}^{d_{\text{ff}}\times d}$，$g$ 为门激活：

$$
\begin{aligned}
Z&=g(XW_1)\odot XW_3\in\mathbb{R}^{n\times d_{\text{ff}}},\\
\operatorname{GLU}_g(X)&=ZW_2\in\mathbb{R}^{n\times d}.
\end{aligned}
$$

| $g$ | 名称 |
| --- | --- |
| $\operatorname{sigmoid}$ | GLU |
| $\operatorname{GELU}$ | GEGLU |
| $\operatorname{SiLU}$ | SwiGLU |
| 恒等 | Bilinear |

## 推导

### 参数量对齐

三个矩阵共 $3dd_{\text{ff}}$ 个参数。要与 $d_{\text{ff}}=4d$ 的 [[FFN]]（$8d^2$）相同：

$$
3dd_{\text{ff}}=8d^2\;\Longrightarrow\;d_{\text{ff}}=\frac{8d}{3}=\frac23\cdot4d .
$$

### 隐藏单元是二次形式

记 $u_l,f_l\in\mathbb{R}^{d\times1}$ 为 $W_1,W_3$ 第 $l$ 列。Bilinear（$g$ 为恒等）时

$$
Z_{il}=(x_iu_l)(x_if_l)=x_i\,u_lf_l^\top\,x_i^\top ,
$$

是 $x_i$ 的二次形式，所以 $\operatorname{GLU}(ax_i)=a^2\operatorname{GLU}(x_i)$；而 ReLU 的 FFN 是一次正齐次的（[[FFN#ReLU 时分段线性且正齐次]]）。SwiGLU 在 $x_iu_l\gg0$ 时 $\operatorname{SiLU}(x_iu_l)\approx x_iu_l$，接近这一二次形式；$x_iu_l\ll0$ 时门关闭。

### SiLU 的导数与形状

$\operatorname{sigmoid}'(z)=\operatorname{sigmoid}(z)\left(1-\operatorname{sigmoid}(z)\right)$，于是

$$
\operatorname{SiLU}'(z)=\operatorname{sigmoid}(z)\left(1+z\left(1-\operatorname{sigmoid}(z)\right)\right).
$$

- $z\to+\infty$：$\operatorname{SiLU}(z)-z=-z\operatorname{sigmoid}(-z)\to0$，趋于恒等。
- $z\to-\infty$：$\operatorname{SiLU}(z)\to0$。
- $\operatorname{SiLU}'(z)=0\iff1+z(1-\operatorname{sigmoid}(z))=0$，唯一解 $z^*\approx-1.278$，最小值 $\operatorname{SiLU}(z^*)\approx-0.278$：非单调，负半轴有界。

### SiLU 是 ReLU 的光滑化

$z\operatorname{sigmoid}(az)$ 中令 $a\to\infty$：$z>0$ 时 $\operatorname{sigmoid}(az)\to1$，$z<0$ 时 $\to0$，极限为 $\operatorname{ReLU}(z)$。SiLU 是 $a=1$ 的情形。

## 代码

```python
import torch

def silu(z):
    return z * torch.sigmoid(z)

def glu_ffn(x, W_1, W_3, W_2, g=silu):
    # x: [B, n, d]；W_1, W_3: [d, d_ff]，W_2: [d_ff, d]；g=silu 即 SwiGLU
    z = g(x @ W_1) * (x @ W_3)                                      # Z: [B, n, d_ff]
    return z @ W_2                                                  # [B, n, d]

# 验证
B, n, d = 2, 5, 48
d_ff = 8 * d // 3                                                   # 128
assert 3 * d * d_ff == 2 * d * (4 * d)                              # 与 d_ff=4d 的 FFN 参数量相同

x = torch.randn(B, n, d)
W_1, W_3, W_2 = torch.randn(d, d_ff), torch.randn(d, d_ff), torch.randn(d_ff, d)
y = glu_ffn(x, W_1, W_3, W_2, g=lambda u: u)                        # Bilinear
assert torch.allclose(glu_ffn(3 * x, W_1, W_3, W_2, g=lambda u: u), 9 * y, rtol=1e-4, atol=1e-2)  # 二次齐次

z = torch.linspace(-6, 6, 1201, requires_grad=True)
s = silu(z)
assert torch.allclose(s, torch.nn.functional.silu(z))
grad, = torch.autograd.grad(s.sum(), z)
sig = torch.sigmoid(z)
assert torch.allclose(grad, sig * (1 + z * (1 - sig)), atol=1e-6)  # 导数公式
assert abs(z[s.argmin()].item() + 1.278) < 0.01 and abs(s.min().item() + 0.278) < 1e-3   # 最小值
assert torch.allclose(z * torch.sigmoid(1000 * z), torch.relu(z), atol=1e-3)             # a→∞ 趋于 ReLU
```

## 辨析

- SwiGLU 论文写作 $(\operatorname{Swish}_1(xW)\otimes xV)W_2$，本库对应 $W\to W_1$、$V\to W_3$；LLaMA 代码中 `w1`、`w3`、`w2` 与本库同号。
- $\operatorname{Swish}_\beta(z)=z\operatorname{sigmoid}(\beta z)$，$\beta=1$ 即 PyTorch 的 `silu`。
- LLaMA 把 $\frac{8d}{3}$ 向上取到 256 的倍数，参数量略多于 $8d^2$。

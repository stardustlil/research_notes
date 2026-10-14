---
aliases:
  - 旋转位置编码
---

# RoPE（旋转位置编码）

前置：[[Self-Attention]]、[[Sinusoidal-PE]]

$$
\tilde q_i=q_iR(t_i),\qquad \tilde k_j=k_jR(t_j),\qquad
\tilde q_i\tilde k_j^\top=q_iR(i-j)k_j^\top.
$$

## 动机

位置加在输入上时，分数中出现 $x_iGp_j^\top$ 等绝对位置项（[[Learned-PE#位置以绝对形式进入分数]]）。希望对 $q,k$ 各自施加只依赖自身位置的变换，使点积中的位置只以 $i-j$ 出现。旋转满足这一点：两个旋转在点积中相减。

## 定义

单头，$Q=XW_Q$、$K=XW_K\in\mathbb{R}^{n\times d_k}$，$d_k$ 为偶数。$\omega_r=b^{-2r/d_k}$，$r=0,\dots,d_k/2-1$，$b=10000$；$\theta_{i,r}=t_i\omega_r$。

二维旋转（右乘，逆时针转 $\omega t$）：

$$
R_\omega(t)=
\begin{bmatrix}
\cos(\omega t)&\sin(\omega t)\\
-\sin(\omega t)&\cos(\omega t)
\end{bmatrix},
\qquad
R(t)=\operatorname{diag}\left(R_{\omega_0}(t),\dots,R_{\omega_{d_k/2-1}}(t)\right)\in\mathbb{R}^{d_k\times d_k}.
$$

第 $r$ 组分量的旋转（$k$ 同理）：

$$
\begin{aligned}
\tilde q_{i,2r+1}&=q_{i,2r+1}\cos\theta_{i,r}-q_{i,2r+2}\sin\theta_{i,r},\\
\tilde q_{i,2r+2}&=q_{i,2r+1}\sin\theta_{i,r}+q_{i,2r+2}\cos\theta_{i,r}.
\end{aligned}
$$

之后按 [[Self-Attention#定义]] 计算，$S=\tilde Q\tilde K^\top/\sqrt{d_k}$。$V$ 不旋转。

## 推导

### 相对位置性质

$R_\omega(t)^\top=R_\omega(-t)$，$R_\omega(t)R_\omega(\Delta)=R_\omega(t+\Delta)$；块对角矩阵逐块相乘，$R(t)$ 同样满足。于是

$$
\tilde q_i\tilde k_j^\top
=q_iR(t_i)R(t_j)^\top k_j^\top
=q_iR(t_i-t_j)k_j^\top
=q_iR(i-j)k_j^\top.
$$

代入 $q_i=x_iW_Q$、$k_j=x_jW_K$ 得 $x_iW_QR(i-j)W_K^\top x_j^\top$：无论 $W_Q,W_K$ 取何值，位置只通过 $R(i-j)$ 进入。

### 二维情形

$d_k=2$，$q_i=\lVert q_i\rVert[\cos\phi_q,\sin\phi_q]$，$k_j=\lVert k_j\rVert[\cos\phi_k,\sin\phi_k]$：

$$
\tilde q_i\tilde k_j^\top=\lVert q_i\rVert\,\lVert k_j\rVert\cos\big[(\phi_q-\phi_k)+\omega(i-j)\big].
$$

内容决定长度与原夹角，相对位置决定附加角度。

### 保持长度

$R(t)R(t)^\top=I$，故 $\lVert\tilde q_i\rVert=\lVert q_i\rVert$。

## 代码

```python
import torch

def rope_freqs(d_k, b=10000.0):
    return b ** (-2 * torch.arange(d_k // 2) / d_k)                 # ω_r: [d_k/2]

def apply_rope(x, omega):
    # x: [..., n, d_k]；omega: [d_k/2]
    t = torch.arange(x.shape[-2], dtype=torch.float32)              # t: [n]
    theta = t[:, None] * omega                                      # θ_{i,r}: [n, d_k/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]                             # 分量 2r+1, 2r+2
    y1 = x1 * theta.cos() - x2 * theta.sin()
    y2 = x1 * theta.sin() + x2 * theta.cos()
    return torch.stack((y1, y2), -1).flatten(-2)                    # 交错放回: [..., n, d_k]

# 验证：分数只依赖 i - j；长度不变
d_k, n = 8, 8
omega = rope_freqs(d_k)
u, w = torch.randn(d_k), torch.randn(d_k)
def score(i, j):
    q, k = torch.zeros(n, d_k), torch.zeros(n, d_k)
    q[i], k[j] = u, w
    return apply_rope(q, omega)[i] @ apply_rope(k, omega)[j]
assert torch.allclose(score(1, 3), score(4, 6), atol=1e-5)
x = torch.randn(n, d_k)
assert torch.allclose(apply_rope(x, omega).norm(dim=-1), x.norm(dim=-1), atol=1e-5)
```

## 辨析

- 配对方式：本库与 LLaMA 原始实现取相邻分量 $(2r+1,2r+2)$；Hugging Face 的 `rotate_half` 取 $(c,c+d_k/2)$。二者差一个固定重排，必须与权重一致。
- 在 [[MHA]] 中作用于每个头的 $Q^{(h)},K^{(h)}$，$d_k$ 是单头维度。

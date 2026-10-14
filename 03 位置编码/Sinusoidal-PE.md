---
aliases:
  - 正余弦位置编码
---

# Sinusoidal-PE（正余弦位置编码）

前置：[[Learned-PE]]

$$
\operatorname{PE}(t)_{2r+1}=\sin(\omega_rt),\qquad
\operatorname{PE}(t)_{2r+2}=\cos(\omega_rt),\qquad
\omega_r=b^{-2r/d}.
$$

## 动机

[[Learned-PE]] 需要训练位置表且只定义到 $n_{\max}$。改用固定函数：任意 $t$ 可计算，并且位置平移 $\Delta$ 对应一个只依赖 $\Delta$ 的线性变换。

## 定义

$d$ 为偶数，$r=0,\dots,d/2-1$，$b=10000$。$p_i=\operatorname{PE}(t_i)$，与 [[Learned-PE]] 相同地相加：

$$
\tilde X=X+P_{1:n},\qquad P_{1:n}=\begin{bmatrix}\operatorname{PE}(t_1)\\ \vdots\\ \operatorname{PE}(t_n)\end{bmatrix}\in\mathbb{R}^{n\times d}.
$$

原论文 $PE(pos,2i)$ 中 $pos=t$、$i=r$，第 $2i$ 维（0 起）即本库第 $2r+1$ 个分量（1 起）。

下图为 $n=d=100$ 的 $P$（横轴分量，纵轴 $t$）：小 $r$ 高频，大 $r$ 低频。

![[attachments/sinusoidal-pe.png]]

## 推导

### 平移对应固定旋转

取第 $r$ 组，$\omega=\omega_r$，由和角公式：

$$
\big[\sin(\omega(t+\Delta)),\ \cos(\omega(t+\Delta))\big]
=\big[\sin(\omega t),\ \cos(\omega t)\big]
\begin{bmatrix}
\cos(\omega\Delta)&-\sin(\omega\Delta)\\
\sin(\omega\Delta)&\cos(\omega\Delta)
\end{bmatrix}.
$$

右侧即 [[RoPE#定义]] 中的 $R_\omega(-\Delta)$，与 $t$ 无关。拼接各组：

$$
\operatorname{PE}(t+\Delta)=\operatorname{PE}(t)\,R(-\Delta).
$$

### 内积只依赖距离

$$
\operatorname{PE}(t)\operatorname{PE}(t+\Delta)^\top
=\sum_{r}\big[\sin(\omega_rt)\sin(\omega_r(t+\Delta))+\cos(\omega_rt)\cos(\omega_r(t+\Delta))\big]
=\sum_{r=0}^{d/2-1}\cos(\omega_r\Delta).
$$

### 波长成等比数列

$\lambda_r=2\pi/\omega_r=2\pi\,b^{2r/d}$，从 $2\pi$ 增长到接近 $2\pi b$，公比 $b^{2/d}$。

## 代码

```python
import torch

def sinusoidal_pe(max_len, d, b=10000.0):
    t = torch.arange(max_len, dtype=torch.float32)[:, None]        # t: [n_max, 1]
    omega = b ** (-2 * torch.arange(d // 2) / d)                    # ω_r: [d/2]
    P = torch.zeros(max_len, d)
    P[:, 0::2] = torch.sin(t * omega)                               # 分量 2r+1
    P[:, 1::2] = torch.cos(t * omega)                               # 分量 2r+2
    return P                                                        # P: [n_max, d]

# 验证：内积只依赖 Δ
P, d = sinusoidal_pe(64, 16), 16
omega = 10000.0 ** (-2 * torch.arange(d // 2) / d)
for t, delta in [(0, 5), (20, 5), (7, 30)]:
    assert torch.allclose(P[t] @ P[t + delta], torch.cos(omega * delta).sum(), atol=1e-4)
```

## 辨析

- 编码加在 $X$ 上：经过 $W_QW_K^\top$ 后，分数一般不再只依赖 $i-j$（同 [[Learned-PE#位置以绝对形式进入分数]]）。

---
aliases:
  - YaRN 长上下文扩展
---

# YaRN（长上下文扩展）

前置：[[RoPE]]

$$
\omega'_r=(1-\xi_r)\frac{\omega_r}{s}+\xi_r\,\omega_r,\qquad
A=\operatorname{softmax}\left(\frac{S}{\tau}+M\right),\quad \tau=\frac{1}{\alpha^2}.
$$

## 动机

把训练窗口 $n_{\text{train}}$ 的 [[RoPE]] 模型扩展到 $n_{\text{ext}}=s\,n_{\text{train}}$。相对旋转角 $\Delta\theta_r=(t_i-t_j)\omega_r$：

- 直接外推：低频组出现训练中没见过的角度。
- 统一插值 $\omega_r\to\omega_r/s$：角度回到训练范围，但相邻 token 的角度差缩小为 $\omega_r/s$，高频组失去局部分辨率。

YaRN 按频率分别处理，并调整 softmax 温度。

## 定义

### 分频插值

第 $r$ 组的波长与训练窗口内的圈数：

$$
\lambda_r=\frac{2\pi}{\omega_r},
\qquad
\rho_r=\frac{n_{\text{train}}}{\lambda_r}.
$$

圈数少（低频）完全插值，圈数多（高频）保持，中间线性过渡：

$$
\xi_r=
\begin{cases}
0, & \rho_r\le\rho_{\text{lo}},\\
\dfrac{\rho_r-\rho_{\text{lo}}}{\rho_{\text{hi}}-\rho_{\text{lo}}}, & \rho_{\text{lo}}<\rho_r<\rho_{\text{hi}},\\
1, & \rho_r\ge\rho_{\text{hi}}.
\end{cases}
$$

LLaMA 系列取 $\rho_{\text{lo}}=1$、$\rho_{\text{hi}}=32$（[论文 §3.2](https://arxiv.org/html/2309.00071v3#S3.SS2)）。

### 温度

$\tilde Q,\tilde K$ 按 $\omega'_r$ 旋转，$S=\tilde Q\tilde K^\top/\sqrt{d_k}$。把 $\tilde Q,\tilde K$ 各乘 $\alpha$：

$$
\frac{(\alpha\tilde Q)(\alpha\tilde K)^\top}{\sqrt{d_k}}=\alpha^2S=\frac{S}{\tau}.
$$

$M/\tau=M$（元素只有 $0,-\infty$），所以与 $\operatorname{softmax}(S/\tau+M)$ 一致。拟合值 $\alpha=1+0.1\ln s$，$s>1$ 时 $\tau<1$（[论文 §3.3](https://arxiv.org/html/2309.00071v3#S3.SS3)）。

## 推导

### 两端的含义

$\xi_r=0\Rightarrow\omega'_r=\omega_r/s$（完全插值）；$\xi_r=1\Rightarrow\omega'_r=\omega_r$（不变）。中间段是过 $(\rho_{\text{lo}},0)$、$(\rho_{\text{hi}},1)$ 的直线，保证连续；直线形式与阈值是实验选择，不是推导结果。

### 降低温度使权重更集中

固定第 $i$ 行，$\mathcal{V}_i=\{j:M_{ij}=0\}$，$n_i=\lvert\mathcal{V}_i\rvert$，以下求和都在 $\mathcal{V}_i$ 上。定义

$$
\sigma_{A,i}^2=\frac{1}{n_i}\sum_{j}A_{ij}^2-\frac{1}{n_i^2},
\qquad
\bar S_i^{A}=\sum_{j}A_{ij}S_{ij}.
$$

softmax 对温度求导：

$$
A_{ij}=\frac{e^{S_{ij}/\tau}}{\sum_{j'}e^{S_{ij'}/\tau}}
\;\Longrightarrow\;
\frac{dA_{ij}}{d\tau}=-\frac{A_{ij}}{\tau^2}\left(S_{ij}-\bar S_i^{A}\right).
$$

代入并利用 $\sum_{j'}A_{ij'}=1$ 对称化：

$$
\begin{aligned}
\frac{d\sigma_{A,i}^2}{d\tau}
&=-\frac{2}{n_i\tau^2}\left(\sum_{j}A_{ij}^2S_{ij}-\bar S_i^{A}\sum_{j}A_{ij}^2\right)\\
&=-\frac{1}{n_i\tau^2}\sum_{j,j'}A_{ij}A_{ij'}\left(A_{ij}-A_{ij'}\right)\left(S_{ij}-S_{ij'}\right)\le0,
\end{aligned}
$$

最后一步因 $A_{ij}$ 与 $S_{ij}$ 同序。所以 $\tau$ 减小时权重方差不减。

## 代码

```python
import math
import torch
# 依赖：[[RoPE]] 中的 rope_freqs、apply_rope

def yarn_freqs(d_k, n_train, s, b=10000.0, rho_lo=1.0, rho_hi=32.0):
    omega = rope_freqs(d_k, b)                                      # ω_r
    rho = n_train * omega / (2 * math.pi)                           # ρ_r = n_train / λ_r
    xi = ((rho - rho_lo) / (rho_hi - rho_lo)).clamp(0, 1)           # ξ_r
    return (1 - xi) * omega / s + xi * omega                        # ω'_r: [d_k/2]

def yarn_rope(q, k, n_train, s):
    # q, k: [..., n, d_k]
    omega_p = yarn_freqs(q.shape[-1], n_train, s)
    alpha = 1 + 0.1 * math.log(s)                                   # τ = 1 / α²
    return alpha * apply_rope(q, omega_p), alpha * apply_rope(k, omega_p)

# 验证：高频组不变，低频组除以 s；权重方差随 τ 减小而增大
d_k, n_train, s = 64, 4096, 8
omega, omega_p = rope_freqs(d_k), yarn_freqs(d_k, n_train, s)
rho = n_train * omega / (2 * math.pi)
assert torch.allclose(omega_p[rho >= 32], omega[rho >= 32])
assert torch.allclose(omega_p[rho <= 1], omega[rho <= 1] / s)
S = torch.randn(10)
var = [torch.softmax(S / tau, -1).var(unbiased=False) for tau in (2.0, 1.0, 0.5)]
assert var[0] <= var[1] <= var[2]
```

## 辨析

- 作者实现先把圈数阈值换算成组编号 $r$，在 $r$ 上线性过渡，中间段权重与本文不完全相同；$\alpha$ 乘进 cos、sin 表，等价于 $\tilde q,\tilde k$ 各乘 $\alpha$（[作者实现](https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py)）。
- 论文设置：4K 的 Llama 2，$s=16$ 用 64K 数据训练 400 步，再 $s=32$ 继续 200 步，评估到 128K（[论文 §4.1](https://arxiv.org/html/2309.00071v3#S4.SS1)）。

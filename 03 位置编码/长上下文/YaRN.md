---
aliases:
  - YaRN 长上下文扩展
tags:
  - Transformer/位置编码/长上下文
---

# YaRN（长上下文扩展）

上级：[[Position Encode|位置编码]] ｜ 前置：[[RoPE]]

> [!abstract] 总结
> 把训练窗口为 $n_{\text{train}}$ 的 RoPE 模型扩展到 $n_{\text{ext}}$：低频组插值、高频组保持、中间线性过渡（分频插值），同时把 $q,k$ 乘以 $\alpha$ 来降低 softmax 温度。

## 动机

希望以较少的继续训练成本，把使用 [[RoPE]] 的模型从训练窗口 $n_{\text{train}}$ 扩展到 $n_{\text{ext}}>n_{\text{train}}$，改善长文本能力，同时尽量保持短文本能力。

回顾 RoPE：第 $r$ 组的旋转角为 $\theta_{i,r}=t_i\omega_r$，两个位置的相对旋转角为

$$
\Delta\theta_r=(t_i-t_j)\,\omega_r.
$$

两种直接做法都有问题：

- **直接外推**：位置超过 $n_{\text{train}}$ 后，低频组会出现训练中从未见过的角度。
- **统一插值** $\omega_r\mapsto\omega_r/s$：所有角度都落回训练范围内，但相邻 token 的角度差从 $\omega_r$ 缩小为 $\omega_r/s$，高频组区分局部位置的能力被削弱。

YaRN 的做法是**分频插值 + attention 温度校准**。

## 符号

| 符号 | 含义 |
| --- | --- |
| $s=n_{\text{ext}}/n_{\text{train}}$ | 扩展倍数 |
| $\lambda_r=2\pi/\omega_r$ | 第 $r$ 组的波长 |
| $\rho_r=n_{\text{train}}/\lambda_r$ | 训练窗口内第 $r$ 组转过的圈数 |
| $\rho_{\text{lo}},\rho_{\text{hi}}$ | 圈数阈值（论文记作 $\alpha,\beta$，与下面的 $\alpha$ 无关） |
| $\xi_r$ | 第 $r$ 组保留原频率的比例（论文记作 $\gamma(r)$） |
| $\omega'_r$ | 插值后的角频率 |
| $\tau$ | softmax 温度（论文记作 $t$） |
| $\alpha$ | 对 $q,k$ 的幅度倍率 |
| $\mathcal{V}_i$、$n_i$ | 第 $i$ 行未被屏蔽的 key 下标集合 $\{j:M_{ij}=0\}$ 及其大小（因果 mask 下 $n_i=i$） |

论文符号与本库的对照见 [[数学符号约定#4.6 与论文 / 框架写法的对照|符号约定]]。

## 定义

### 分频插值

按原训练窗口与原频率计算每组的圈数：

$$
\lambda_r=\frac{2\pi}{\omega_r},
\qquad
\rho_r=\frac{n_{\text{train}}}{\lambda_r}=\frac{n_{\text{train}}\,\omega_r}{2\pi}.
$$

转得少的（低频）组完整插值，转得多的（高频）组保持原频率，中间线性过渡：

$$
\omega'_r=(1-\xi_r)\frac{\omega_r}{s}+\xi_r\,\omega_r,
\qquad
\xi_r=
\begin{cases}
0, & \rho_r\le\rho_{\text{lo}},\\
\dfrac{\rho_r-\rho_{\text{lo}}}{\rho_{\text{hi}}-\rho_{\text{lo}}}, & \rho_{\text{lo}}<\rho_r<\rho_{\text{hi}},\\
1, & \rho_r\ge\rho_{\text{hi}}.
\end{cases}
$$

论文针对 LLaMA 系列取 $\rho_{\text{lo}}=1$、$\rho_{\text{hi}}=32$。[论文 §3.2](https://arxiv.org/html/2309.00071v3#S3.SS2)

### 校准 attention 分数

设 $\tilde Q,\tilde K$ 已按新频率 $\omega'_r$ 旋转、尚未做幅度缩放。引入温度 $\tau>0$：

$$
S=\frac{\tilde Q\tilde K^\top}{\sqrt{d_k}},
\qquad
A=\operatorname{softmax}\left(\frac{S}{\tau}+M\right).
$$

因为 $M$ 的元素只取 $0$ 或 $-\infty$，有 $M/\tau=M$，所以先加 mask 再除以 $\tau$ 与上式等价。

YaRN 不直接改 softmax，而是把 $\tilde Q,\tilde K$ 同乘幅度倍率 $\alpha$，分数随之乘以 $\alpha^2$：

$$
\frac{(\alpha\tilde Q)(\alpha\tilde K)^\top}{\sqrt{d_k}}=\alpha^2S=\frac{S}{\tau},
\qquad
\tau=\frac{1}{\alpha^2}.
$$

针对 LLaMA 系列，作者通过实验拟合：

$$
\alpha=1+0.1\ln s,
\qquad
\tau=(1+0.1\ln s)^{-2}.
$$

$s>1$ 时 $\tau<1$，即**降低温度**。[论文 §3.3](https://arxiv.org/html/2309.00071v3#S3.SS3)

## 性质与推导

### 插值比例为什么是线性的

中间段取经过 $(\rho_{\text{lo}},0)$ 与 $(\rho_{\text{hi}},1)$ 两点的直线，即 $\xi(\rho)=(\rho-\rho_{\text{lo}})/(\rho_{\text{hi}}-\rho_{\text{lo}})$，使两端与常数段连续。采用直线、以及阈值取多少，都属于方法设计与实验选择，不是推导出来的。

两端的含义：$\xi_r=0$ 时 $\omega'_r=\omega_r/s$（完整插值）；$\xi_r=1$ 时 $\omega'_r=\omega_r$（保持不变）。

### 温度与分数方差

固定一个 query（第 $i$ 行），只看未屏蔽的位置 $j\in\mathcal{V}_i$，改变 $\tau$ 时保持分数 $S_{ij}$ 不变。令

$$
\bar S_i=\frac{1}{n_i}\sum_{j\in\mathcal{V}_i}S_{ij},
\qquad
\sigma_{S,i}^2=\frac{1}{n_i}\sum_{j\in\mathcal{V}_i}\left(S_{ij}-\bar S_i\right)^2.
$$

缩放后的均值为 $\bar S_i/\tau$，方差为

$$
\frac{1}{n_i}\sum_{j\in\mathcal{V}_i}\left(\frac{S_{ij}}{\tau}-\frac{\bar S_i}{\tau}\right)^2=\frac{\sigma_{S,i}^2}{\tau^2}.
$$

所以 **$0<\tau<1$ 放大分数方差，$\tau>1$ 缩小分数方差**。

### 温度与注意力权重方差

可见位置的权重为

$$
A_{ij}(\tau)=\frac{e^{S_{ij}/\tau}}{\sum_{j'\in\mathcal{V}_i}e^{S_{ij'}/\tau}},
\qquad
\frac{A_{ij}}{A_{ij'}}=e^{(S_{ij}-S_{ij'})/\tau}.
$$

对 $S_{ij}>S_{ij'}$，降低 $\tau$ 会增大权重比。定义第 $i$ 行权重在可见位置上的方差，以及按权重加权的平均分数：

$$
\sigma_{A,i}^2(\tau)=\frac{1}{n_i}\sum_{j\in\mathcal{V}_i}\left(A_{ij}-\frac{1}{n_i}\right)^2=\frac{1}{n_i}\sum_{j\in\mathcal{V}_i}A_{ij}^2-\frac{1}{n_i^2},
\qquad
\bar S_i^{A}=\sum_{j\in\mathcal{V}_i}A_{ij}S_{ij}.
$$

对 softmax 求导：

$$
\frac{dA_{ij}}{d\tau}=-\frac{A_{ij}}{\tau^2}\left(S_{ij}-\bar S_i^{A}\right).
$$

代入 $\sigma_{A,i}^2(\tau)$，利用 $\sum_{j'}A_{ij'}=1$ 并对称化（以下求和均在 $\mathcal{V}_i$ 上）：

$$
\begin{aligned}
\frac{d\sigma_{A,i}^2}{d\tau}
&=-\frac{2}{n_i\tau^2}\left(\sum_{j}A_{ij}^2S_{ij}-\bar S_i^{A}\sum_{j}A_{ij}^2\right)\\
&=-\frac{1}{n_i\tau^2}\sum_{j,j'}A_{ij}A_{ij'}\left(A_{ij}-A_{ij'}\right)\left(S_{ij}-S_{ij'}\right)\le0.
\end{aligned}
$$

最后一步利用 $A_{ij}$ 与 $S_{ij}$ 同序（分数越大权重越大），因此 $(A_{ij}-A_{ij'})(S_{ij}-S_{ij'})\ge0$。

结论：分数固定时，**降低温度使权重方差不减、分布更集中**；所有分数相等时方差恒为 $0$。该结论不要求每个非最高分位置的权重都下降。

上述方差关系是数学结论；温度的拟合式及其对困惑度的改善来自实验。

## 代码实现

复用 [[RoPE#代码实现|RoPE 笔记中的 `apply_rope`]]，只替换频率并乘上 $\alpha$。

```python
import math
import torch
# 依赖：RoPE 笔记中的 apply_rope(x, omega)

def yarn_frequencies(d_k, n_train, s, base=10000.0, rho_lo=1.0, rho_hi=32.0):
    r = torch.arange(d_k // 2, dtype=torch.float32)
    omega = base ** (-2 * r / d_k)                                   # ω_r: [d_k/2]
    rho = n_train * omega / (2 * math.pi)                            # ρ_r = n_train / λ_r
    xi = ((rho - rho_lo) / (rho_hi - rho_lo)).clamp(0.0, 1.0)        # ξ_r，clamp 实现分段函数
    return (1 - xi) * omega / s + xi * omega                         # ω'_r: [d_k/2]

def yarn_alpha(s):
    return 1.0 + 0.1 * math.log(s) if s > 1 else 1.0                 # α，τ = 1 / α²

def yarn_rope(q, k, n_train, s, base=10000.0):
    # q, k: [..., n, d_k]
    omega_prime = yarn_frequencies(q.shape[-1], n_train, s, base)
    alpha = yarn_alpha(s)
    return alpha * apply_rope(q, omega_prime), alpha * apply_rope(k, omega_prime)   # αQ̃, αK̃
```

## 对比与易错点

- **与作者实现的差别**：上文在圈数 $\rho_r$ 上线性过渡；作者代码先把圈数阈值换算成分组编号 $r$，再在编号区间内线性过渡，中间段的权重不完全相同。作者代码把 $\alpha$ 乘进 cos、sin 表，等价于把 $\tilde q,\tilde k$ 各乘 $\alpha$。[作者实现](https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py)
- **温度方向**：应表述为「降低温度增大方差」，而不是笼统地说「温度增大方差」。
- **设置 $s$ 不等于扩展成功**：有效上下文需要同时通过长文本困惑度、检索与短文本能力评估；YaRN 也不减少长上下文的计算与 KV cache 成本。
- **论文的训练设置**：从原窗口 4K 的 Llama 2 出发，$s=16$，用 64K 长度数据训练 400 步；再设 $s=32$，从前一模型继续训练 200 步，评估到 128K。[论文 §4.1](https://arxiv.org/html/2309.00071v3#S4.SS1)

## 相关笔记

- [[RoPE#定义|RoPE 的分组与频率]]：$\omega_r$、$\theta_{i,r}$ 和旋转矩阵。
- [[Self-Attention#为什么要缩放|Self-Attention 的缩放]]：温度 $\tau$ 是在 $\sqrt{d_k}$ 之上的又一次缩放。
- [[Self-Attention#Softmax 归一化|Self-Attention 的 softmax]]：分数如何变为权重。

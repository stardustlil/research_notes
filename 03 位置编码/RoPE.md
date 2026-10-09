---
aliases:
  - 旋转位置编码
tags:
  - Transformer/位置编码
---

# RoPE（旋转位置编码）

上级：[[Position Encode|位置编码]] ｜ 前置：[[Self-Attention]]、[[Sinusoidal positional encoding]]

> [!abstract] 总结
> 按位置把 $q_i$、$k_j$ 分组旋转 $t_i\omega_r$、$t_j\omega_r$ 角度，使注意力分数中显式出现相对位置：$\tilde q_i\tilde k_j^\top=q_iR(i-j)k_j^\top$。

## 动机

[[Sinusoidal positional encoding|正余弦位置编码]]本身有相对位置结构，但它是加到输入上的。代入投影后：

$$
\tilde q_i\tilde k_j^\top=(x_i+p_i)\,W_QW_K^\top\,(x_j+p_j)^\top,
$$

经过可学习的 $W_QW_K^\top$，位置对分数的作用一般不再只依赖位置差（见 [[Learned Positional Embedding#位置如何进入注意力分数|四项展开]]）。

RoPE 希望对 query、key 施加只依赖各自位置的变换，使点积具有形式

$$
\tilde q_i\tilde k_j^\top=g(q_i,k_j,i-j),
$$

即在内容向量不变时，位置对分数的作用只依赖相对位移。做法是旋转 $q_i,k_j$：两个旋转在点积中相减，自然只剩相对角度。

## 符号

单个注意力头，省略 batch 维度。$Q=XW_Q$、$K=XW_K\in\mathbb{R}^{n\times d_k}$ 为不含位置信息的 query、key，$d_k$ 为偶数。

| 符号 | 含义 |
| --- | --- |
| $b$ | 底数，原始 RoPE 取 $10000$ |
| $\omega_r=b^{-2r/d_k}$ | 第 $r$ 组的角频率，$r=0,\dots,d_k/2-1$ |
| $\theta_{i,r}=t_i\omega_r$ | 第 $i$ 个 token 在第 $r$ 组上的旋转角 |
| $\phi_{q_i},\phi_{k_j}$ | 二维情形下 $q_i,k_j$ 与横轴的夹角（仅用于几何解释） |

## 定义

### 旋转矩阵

采用行向量右乘，二维旋转矩阵定义为

$$
R_\omega(t)=
\begin{bmatrix}
\cos(\omega t)&\sin(\omega t)\\
-\sin(\omega t)&\cos(\omega t)
\end{bmatrix}.
$$

对行向量 $[u_1,u_2]$：

$$
[u_1,u_2]\,R_\omega(t)=\big[u_1\cos(\omega t)-u_2\sin(\omega t),\ u_1\sin(\omega t)+u_2\cos(\omega t)\big],
$$

即把向量**逆时针**旋转 $\omega t$。

把 $d_k$ 个分量两两分组，第 $r$ 组为第 $2r+1$、$2r+2$ 个分量，每组用自己的频率 $\omega_r$。整体旋转矩阵为块对角矩阵：

$$
R(t)=\operatorname{diag}\left(R_{\omega_0}(t),R_{\omega_1}(t),\dots,R_{\omega_{d_k/2-1}}(t)\right)\in\mathbb{R}^{d_k\times d_k}.
$$

### 对 query 和 key 旋转

$$
\tilde q_i=q_iR(t_i),\qquad\tilde k_j=k_jR(t_j).
$$

具体到第 $r$ 组，query 的分量变为（key 同理）：

$$
\begin{aligned}
\tilde q_{i,2r+1}&=q_{i,2r+1}\cos\theta_{i,r}-q_{i,2r+2}\sin\theta_{i,r},\\
\tilde q_{i,2r+2}&=q_{i,2r+1}\sin\theta_{i,r}+q_{i,2r+2}\cos\theta_{i,r}.
\end{aligned}
$$

### 完整注意力计算

把旋转后的行向量堆成矩阵：

$$
\tilde Q=\begin{bmatrix}q_1R(t_1)\\ \vdots\\ q_nR(t_n)\end{bmatrix}\in\mathbb{R}^{n\times d_k},
\qquad
\tilde K=\begin{bmatrix}k_1R(t_1)\\ \vdots\\ k_nR(t_n)\end{bmatrix}\in\mathbb{R}^{n\times d_k}.
$$

其余步骤与 [[Self-Attention#定义|Self-Attention]] 相同：

$$
\begin{aligned}
S&=\frac{\tilde Q\tilde K^\top}{\sqrt{d_k}},\qquad S_{ij}=\frac{q_iR(i-j)k_j^\top}{\sqrt{d_k}},\\
A&=\operatorname{softmax}(S+M),\\
O&=AV\in\mathbb{R}^{n\times d_v}.
\end{aligned}
$$

$S_{ij}$ 的第二种写法见下方[[#相对位置性质]]。**value 不旋转**。

## 性质与推导

### 二维几何解释

取 $d_k=2$，此时只有一组，记 $\omega=\omega_0$。对非零向量写成极坐标：

$$
q_i=\lVert q_i\rVert\,[\cos\phi_{q_i},\sin\phi_{q_i}],
\qquad
k_j=\lVert k_j\rVert\,[\cos\phi_{k_j},\sin\phi_{k_j}].
$$

原点积为

$$
q_ik_j^\top=\lVert q_i\rVert\,\lVert k_j\rVert\cos(\phi_{q_i}-\phi_{k_j}).
$$

旋转不改变长度，只把角度分别加上 $\omega t_i$、$\omega t_j$：

$$
\begin{aligned}
\tilde q_i\tilde k_j^\top
&=\lVert q_i\rVert\,\lVert k_j\rVert\cos\big[(\phi_{q_i}+\omega t_i)-(\phi_{k_j}+\omega t_j)\big]\\
&=\lVert q_i\rVert\,\lVert k_j\rVert\cos\big[(\phi_{q_i}-\phi_{k_j})+\omega(i-j)\big].
\end{aligned}
$$

内容决定长度和原始夹角，相对位置决定额外的角度差。

### 相对位置性质

二维旋转矩阵满足

$$
R_\omega(t)^\top=R_\omega(-t),\qquad R_\omega(t)R_\omega(\Delta)=R_\omega(t+\Delta).
$$

块对角矩阵逐块相乘，所以 $R(t)$ 也满足同样的性质，从而

$$
R(t_i)R(t_j)^\top=R(t_i-t_j)=R(i-j).
$$

代入点积：

$$
\begin{aligned}
\tilde q_i\tilde k_j^\top
&=q_iR(t_i)\big(k_jR(t_j)\big)^\top\\
&=q_iR(t_i)R(t_j)^\top k_j^\top\\
&=q_iR(i-j)k_j^\top.
\end{aligned}
$$

再代入 $q_i=x_iW_Q$、$k_j=x_jW_K$：

$$
\tilde q_i\tilde k_j^\top=x_iW_Q\,R(i-j)\,W_K^\top x_j^\top.
$$

无论 $W_Q,W_K$ 学成什么样，显式的位置作用都只通过 $R(i-j)$ 进入分数。

### 旋转保持长度

$R(t)$ 是正交矩阵，$R(t)R(t)^\top=I$，所以 $\lVert\tilde q_i\rVert=\lVert q_i\rVert$。RoPE 只改变方向，不改变 $q,k$ 的大小。

## 代码实现

`apply_rope` 适用于任意前导维度（如多头的 `[B, H, n, d_k]`），频率 `omega` 作为参数传入，便于 [[YaRN#代码实现|YaRN]] 复用。分组方式为相邻分量配对，与上文定义一致。

```python
import torch

def rope_frequencies(d_k, base=10000.0):
    r = torch.arange(d_k // 2, dtype=torch.float32)                  # r = 0, ..., d_k/2 - 1
    return base ** (-2 * r / d_k)                                    # ω_r: [d_k/2]

def apply_rope(x, omega):
    # x: Q 或 K，[..., n, d_k]；omega: [d_k/2]
    n, d_k = x.shape[-2], x.shape[-1]
    assert d_k % 2 == 0

    t = torch.arange(n, dtype=torch.float32, device=x.device)        # t_i = i - 1: [n]
    theta = t[:, None] * omega.to(x.device)[None, :]                 # θ_{i,r}: [n, d_k/2]
    cos, sin = theta.cos().to(x.dtype), theta.sin().to(x.dtype)

    x1, x2 = x[..., 0::2], x[..., 1::2]                              # 第 2r+1、2r+2 个分量: [..., n, d_k/2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)                 # 交错放回: [..., n, d_k]

def rope(q, k, base=10000.0):
    omega = rope_frequencies(q.shape[-1], base)
    return apply_rope(q, omega), apply_rope(k, omega)                # Q̃, K̃
```

## 对比与易错点

- **配对方式**：本库与 Meta 的 LLaMA 参考实现一样按相邻分量 $(2r+1,2r+2)$ 配对；Hugging Face 的 `rotate_half` 实现把第 $c$ 个与第 $c+d_k/2$ 个分量配对（转换权重时已对 $W_Q,W_K$ 的列做了相应重排）。两者只差一个固定的分量重排，数学性质相同，但必须与预训练权重使用的方式一致。
- **位置编号**：推理时使用 KV cache，新 token 的位置是已有长度而不是 $0$，`t` 需要加上偏移。
- **作用在每个头上**：在 [[MHA]] 中，RoPE 施加在切分后的每个 $Q^{(h)},K^{(h)}$ 上，$d_k$ 是单头维度。
- **只旋转 $q,k$**，不旋转 $v$。

## 相关笔记

- [[Self-Attention#缩放点积分数|Self-Attention 的分数计算]]：对比旋转前后的点积。
- [[Sinusoidal positional encoding#平移对应固定旋转|正余弦编码的平移性质]]：同一个旋转矩阵 $R_\omega$ 的另一种用法。
- [[MHA|多头注意力]]：RoPE 作用在每个头的 $Q^{(h)},K^{(h)}$ 上。
- [[Learned Positional Embedding#位置如何进入注意力分数|可学习位置编码的四项展开]]：对比绝对位置如何进入分数。
- [[YaRN|长上下文扩展（YaRN）]]：修改 $\omega_r$ 并校准温度，把 RoPE 模型扩展到更长上下文。

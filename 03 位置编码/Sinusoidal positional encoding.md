---
aliases:
  - 正余弦位置编码
  - 正弦位置编码
tags:
  - Transformer/位置编码
---

# Sinusoidal Positional Encoding（正余弦位置编码）

上级：[[Position Encode|位置编码]] ｜ 前置：[[Learned Positional Embedding]]

> [!abstract] 总结
> 用 $d/2$ 组不同频率的 $\sin,\cos$ 函数生成固定的位置向量 $\operatorname{PE}(t)$，加到输入上；位置平移 $\Delta$ 等价于对位置向量做一个只依赖 $\Delta$ 的旋转。

## 动机

原始 Transformer 希望位置编码满足：

1. 没有可学习参数，不需要训练位置表；
2. 任意位置 $t$ 都能算出编码，形式上不受 $n_{\max}$ 限制（但不保证模型在更长序列上的效果）；
3. 多频率结构：高频分量区分相邻位置，低频分量区分远距离位置；
4. **固定偏移对应固定的线性变换**，原论文希望这有助于模型学习相对位置关系。

## 符号

要求 $d$ 为正偶数。局部符号：

| 符号 | 含义 |
| --- | --- |
| $\operatorname{PE}(t)\in\mathbb{R}^{1\times d}$ | 位置编号 $t$ 的位置向量 |
| $\omega_r=b^{-2r/d}$ | 第 $r$ 组的角频率，$r=0,\dots,d/2-1$，$b=10000$ |

这里的频率与 [[RoPE]] 的 $\omega_r$ 形式相同，只是指数分母为 $d$（RoPE 中为 $d_k$）。

## 定义

第 $r$ 组占据第 $2r+1$、$2r+2$ 个分量，分别取 $\sin$ 和 $\cos$：

$$
\begin{aligned}
\operatorname{PE}(t)_{2r+1}&=\sin(\omega_rt),\\
\operatorname{PE}(t)_{2r+2}&=\cos(\omega_rt).
\end{aligned}
$$

位置矩阵的第 $i$ 行取 $p_i=\operatorname{PE}(t_i)$，与[[Learned Positional Embedding|可学习位置编码]]相同地加到输入上：

$$
\tilde X=X+P_{1:n},\qquad P_{1:n}=\begin{bmatrix}\operatorname{PE}(t_1)\\ \vdots\\ \operatorname{PE}(t_n)\end{bmatrix}\in\mathbb{R}^{n\times d}.
$$

> [!note] 与原论文写法的对应
> 原论文写作 $PE(pos,2i)=\sin\left(pos/10000^{2i/d_{\text{model}}}\right)$，其中特征维从 $0$ 编号。对应关系为 $pos=t$、$i=r$，原论文的第 $2i$ 维（0 起）就是本库的第 $2r+1$ 个分量（1 起）。

下图是 $n=100$、$d=100$ 的 $P$：横轴为特征分量，纵轴为位置 $t$。左侧（小 $r$）频率高、变化快；右侧（大 $r$）频率低、几乎不变。

![[attachments/{52AD81BD-000F-4C84-8521-7AB508EEC60A}.png]]

## 性质与推导

### 平移对应固定旋转

只看第 $r$ 组，记 $\omega=\omega_r$。由和角公式：

$$
\begin{aligned}
\sin(\omega(t+\Delta))&=\sin(\omega t)\cos(\omega\Delta)+\cos(\omega t)\sin(\omega\Delta),\\
\cos(\omega(t+\Delta))&=\cos(\omega t)\cos(\omega\Delta)-\sin(\omega t)\sin(\omega\Delta).
\end{aligned}
$$

写成行向量右乘矩阵：

$$
\big[\sin(\omega(t+\Delta)),\ \cos(\omega(t+\Delta))\big]
=\big[\sin(\omega t),\ \cos(\omega t)\big]
\begin{bmatrix}
\cos(\omega\Delta)&-\sin(\omega\Delta)\\
\sin(\omega\Delta)&\cos(\omega\Delta)
\end{bmatrix}.
$$

右侧矩阵就是 [[RoPE#定义|RoPE 中的旋转矩阵]] $R_\omega(-\Delta)$，**只依赖偏移 $\Delta$，与 $t$ 无关**。把 $d/2$ 组拼起来：

$$
\operatorname{PE}(t+\Delta)=\operatorname{PE}(t)\,R(-\Delta),
$$

其中 $R(\cdot)$ 是 RoPE 笔记中的块对角旋转矩阵（这里维度为 $d$）。这就是「固定偏移对应固定线性变换」。

### 位置向量的内积只依赖距离

$$
\operatorname{PE}(t)\operatorname{PE}(t+\Delta)^\top
=\sum_{r=0}^{d/2-1}\big[\sin(\omega_rt)\sin(\omega_r(t+\Delta))+\cos(\omega_rt)\cos(\omega_r(t+\Delta))\big]
=\sum_{r=0}^{d/2-1}\cos(\omega_r\Delta).
$$

结果只与 $\Delta$ 有关，且关于 $\Delta$ 对称。

### 波长呈几何级数

第 $r$ 组的波长 $\lambda_r=2\pi/\omega_r=2\pi\,b^{2r/d}$，从 $2\pi$（$r=0$）增长到接近 $2\pi b$，相邻两组之比固定为 $b^{2/d}$。

## 代码实现

`t` 对应 $t$，`omega` 对应 $\omega_r$；`0::2` 取出 0 起编号的偶数列，即本库 1 起编号的第 $2r+1$ 个分量。

```python
import torch

def sinusoidal_position_encoding(max_len, d_model, base=10000.0):
    if d_model <= 0 or d_model % 2 != 0:
        raise ValueError("d_model 必须为正偶数")

    t = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)          # t: [n_max, 1]
    r = torch.arange(d_model // 2, dtype=torch.float32)                   # r: [d/2]
    omega = base ** (-2 * r / d_model)                                    # ω_r: [d/2]

    P = torch.zeros(max_len, d_model)                                     # P: [n_max, d]
    P[:, 0::2] = torch.sin(t * omega)                                     # 分量 2r+1
    P[:, 1::2] = torch.cos(t * omega)                                     # 分量 2r+2
    return P
```

## 对比与易错点

- 许多实现写成 `torch.exp(2r * (-ln b / d))`，它与 $b^{-2r/d}$ 数学上相同，只是计算方式不同。
- **加在输入上，而不是 $q,k$ 上**：经过投影 $W_Q,W_K$ 后，位置对分数的作用一般不再只依赖位置差，见 [[RoPE#动机|RoPE 的动机]]。
- 也有实现把 $\sin$ 放前一半、$\cos$ 放后一半（不交错），只是分量顺序不同。

## 相关笔记

- [[Learned Positional Embedding|可学习绝对位置编码]]：同样加在输入上，但位置向量是学出来的。
- [[RoPE|旋转位置编码]]：把这里的「平移 = 旋转」性质直接用到 $q_i,k_j$ 上。

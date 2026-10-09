---
aliases:
  - 自注意力
tags:
  - Transformer/注意力
---

# Self-Attention（自注意力）

上级：[[Attention|注意力]]

> [!abstract] 总结
> 每个 token 用自己的 query 与所有 key 做缩放点积，经 mask 和 softmax 得到权重，再对所有 value 加权平均：
> $$O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.$$

## 动机

希望每个 token 的新表示能根据**内容**从整个序列中汇聚信息：与自己越相关的 token，贡献越大。权重不是固定的，而是由输入本身算出来的。

## 符号

本篇讨论单个注意力头，省略 batch 维度。输入与参数：

$$
X\in\mathbb{R}^{n\times d},\qquad
W_Q,W_K\in\mathbb{R}^{d\times d_k},\qquad
W_V\in\mathbb{R}^{d\times d_v}.
$$

$X$ 的第 $i$ 行 $x_i\in\mathbb{R}^{1\times d}$ 是第 $i$ 个 token 的向量。其余符号（$Q,K,V,S,M,S',A,O$）见 [[数学符号约定#4.3 注意力|符号约定 · 注意力]]。

## 定义

### 计算 Q、K、V

$$
\begin{aligned}
Q&=XW_Q\in\mathbb{R}^{n\times d_k},\\
K&=XW_K\in\mathbb{R}^{n\times d_k},\\
V&=XW_V\in\mathbb{R}^{n\times d_v}.
\end{aligned}
$$

第 $i$ 行分别为 $q_i=x_iW_Q$、$k_i=x_iW_K$、$v_i=x_iW_V$。

### 缩放点积分数

$$
S=\frac{QK^\top}{\sqrt{d_k}}\in\mathbb{R}^{n\times n},
\qquad
S_{ij}=\frac{q_ik_j^\top}{\sqrt{d_k}}.
$$

$S_{ij}$ 是第 $i$ 个 token（query）对第 $j$ 个 token（key）的未归一化相关性分数。

### 加入 mask

$$
S'=S+M.
$$

$M\in\mathbb{R}^{n\times n}$ 为加性 mask：允许关注的位置取 $0$，屏蔽的位置取 $-\infty$。因果 mask 在 $j>i$ 时取 $-\infty$；无 mask 时 $M=0$。要求每行至少保留一个允许位置。

### Softmax 归一化

对 $S'$ 按行做 softmax，得到注意力权重：

$$
A=\operatorname{softmax}(S')\in\mathbb{R}^{n\times n},
\qquad
A_{ij}=\frac{\exp(S'_{ij})}{\sum_{j'=1}^{n}\exp(S'_{ij'})}.
$$

每行满足 $A_{ij}\ge0$、$\sum_{j}A_{ij}=1$；被屏蔽位置的权重为 $0$。

### 加权求和

$$
O=AV\in\mathbb{R}^{n\times d_v},
\qquad
o_i=\sum_{j=1}^{n}A_{ij}v_j\in\mathbb{R}^{1\times d_v}.
$$

第 $i$ 个 token 的输出 $o_i$ 是允许关注的 value 向量的加权平均。

## 性质与推导

### 为什么要缩放

设 $q_i,k_j$ 的各分量相互独立、均值为 $0$、方差为 $1$。每一项 $q_{i,c}k_{j,c}$ 的均值为 $0$，方差为

$$
\operatorname{Var}(q_{i,c}k_{j,c})=\mathbb{E}[q_{i,c}^2]\,\mathbb{E}[k_{j,c}^2]=1.
$$

$d_k$ 项独立相加，得到 $\operatorname{Var}(q_ik_j^\top)=d_k$。因此

$$
\operatorname{Var}\left(\frac{q_ik_j^\top}{\sqrt{d_k}}\right)=1.
$$

不缩放时，分数的标准差随 $\sqrt{d_k}$ 增长，softmax 容易接近 one-hot，梯度变小。除以 $\sqrt{d_k}$ 让分数尺度与 $d_k$ 无关。[[YaRN#校准 attention 分数|YaRN 的温度]]就是在此基础上再乘一个系数。

### 置换等变性

局部符号：$\Pi\in\{0,1\}^{n\times n}$ 为置换矩阵，满足 $\Pi^\top\Pi=I$；$\Pi X$ 表示把 $X$ 的行（token）重新排列。

无 mask（$M=0$）时，把注意力看成函数 $f(X)=\operatorname{softmax}\left(XW_QW_K^\top X^\top/\sqrt{d_k}\right)XW_V$。输入换成 $\Pi X$ 后：

$$
\begin{aligned}
S_\Pi&=\frac{(\Pi Q)(\Pi K)^\top}{\sqrt{d_k}}=\Pi S\Pi^\top,\\
A_\Pi&=\operatorname{softmax}(\Pi S\Pi^\top)=\Pi A\Pi^\top,\\
O_\Pi&=A_\Pi(\Pi V)=\Pi A\Pi^\top\Pi V=\Pi AV=\Pi O.
\end{aligned}
$$

第二行成立，是因为行 softmax 只在每一行内部归一化：左乘 $\Pi$ 只交换行，右乘 $\Pi^\top$ 只在每行内部交换元素顺序，都不改变 softmax 的计算结果，只改变位置。

结论：$f(\Pi X)=\Pi f(X)$。**打乱输入顺序，输出只按同样方式被打乱**，注意力本身无法区分 token 的位置。这就是[[Position Encode|位置编码]]的动机。

带因果 mask 时，一般 $\Pi M\Pi^\top\neq M$，上述推导不成立：因果 mask 本身就包含了顺序信息。

## 代码实现

为了与公式逐行对应，这里直接用矩阵 `W_q, W_k, W_v` 做右乘，而不是 `nn.Linear`。

```python
import math
import torch

def self_attention(x, W_q, W_k, W_v, causal=False):
    # x: [B, n, d]；W_q, W_k: [d, d_k]；W_v: [d, d_v]
    n = x.shape[-2]
    d_k = W_q.shape[-1]

    q = x @ W_q                                            # Q: [B, n, d_k]
    k = x @ W_k                                            # K: [B, n, d_k]
    v = x @ W_v                                            # V: [B, n, d_v]

    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)      # S: [B, n, n]

    if causal:
        # True 表示屏蔽：j > i 的上三角部分
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))   # S' = S + M

    attn = torch.softmax(scores, dim=-1)                   # A: [B, n, n]，按行
    out = attn @ v                                         # O: [B, n, d_v]
    return out
```

## 对比与易错点

- **softmax 的维度**：必须是 `dim=-1`（对 key 归一化），对应公式中的「按行」。
- **整行被屏蔽会得到 NaN**：分母变成 $0$。所以要求每行至少有一个允许位置（因果 mask 下对角线总是允许的）。
- **缩放用 $\sqrt{d_k}$**，不是 $\sqrt d$。
- **mask 的布尔语义**：本库 `True` 表示屏蔽；`F.scaled_dot_product_attention` 的 bool `attn_mask` 中 `True` 表示允许，见 [[数学符号约定#6. 数学 ↔ 代码对照|符号约定]]。
- 本实现与 `F.scaled_dot_product_attention(q, k, v, is_causal=causal)` 的结果一致。

## 相关笔记

- [[MHA|多头注意力]]：把本篇的计算并行做 $H$ 次，再拼接融合。
- [[Position Encode|位置编码]]：置换等变性说明需要额外注入位置信息。
- [[Learned Positional Embedding|可学习绝对位置编码]]：打破置换等变性的最直接做法。
- [[RoPE#完整注意力计算|带 RoPE 的注意力]]：对 $q_i,k_j$ 旋转后，分数 $S_{ij}$ 显式依赖 $i-j$。
- [[YaRN#校准 attention 分数|YaRN 的温度校准]]：在 $S$ 上再除以温度 $\tau$。

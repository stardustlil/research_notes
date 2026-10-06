## 为什么需要位置编码

### Self-Attention 的置换等变性

把输入 token 的顺序重新排列，输出也会按照同样的顺序重新排列。

**没有位置编码时，无 mask 的 self-attention 具有置换等变性。**

因此，仅靠这样的注意力计算无法区分 token 的绝对位置。这里讨论的是无 mask 的情况；固定的因果 mask 本身就包含顺序约束，不能直接套用同样的结论。

## 可学习绝对位置编码

**定义可学习位置矩阵**
设最大序列长度为 N ， 模型维度为 d。定义可学习嵌入矩阵
$$
P \in \mathbb{R}^{N \times d}
$$
其中第 i 行为
$$
p_{i} \in \mathbb{R}^{1\times d}
$$
表示绝对位置 i 的位置向量，P 是可学习矩阵。

对于长度为 ${}n \leq N{}$ 的输入
$$
X \in \mathbb{R}^{n\times d}, \quad x_{i} \in \mathbb{R}^{1\times d}
$$
取${}P{}$ 的前${}n{}$ 行：
$$
P_{1:n} = \begin{bmatrix}
p_{1} \\
p_{2} \\
\vdots \\
p_{n}
\end{bmatrix} \in\mathbb{R}^{n\times d}
$$

进入注意力的输入变为
$$
\tilde{X} = X + P_{1:n} \in \mathbb{R}^{n\times d}.
$$
## 正余弦位置编码
[[Sinusoidal positional encoding]]

## RoPE位置编码
[[RoPE]]

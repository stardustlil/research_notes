## 公式

以下讨论单个注意力头，省略 batch 维度。设序列长度为 $n$，模型维度为 $d$，输入矩阵为
$$
X \in \mathbb{R}^{n \times d}
$$
其中第 $i$ 行为 $x_i \in \mathbb{R}^{1 \times d}$ 表示第 $i$ 个 $\text{token}$ 的向量。

设投影维度为 $d_k$ 和 $d_v$。定义可学习参数矩阵：

$$
W_Q \in \mathbb{R}^{d \times d_k}, \quad W_K \in \mathbb{R}^{d \times d_k}, \quad W_V \in \mathbb{R}^{d \times d_v}.

$$
**计算 query , key , value 矩阵：**

$$
\begin{aligned}
Q &= XW_Q \in \mathbb{R}^{n \times d_k} \\
K &= XW_K \in \mathbb{R}^{n \times d_k} \\
V &= XW_V \in \mathbb{R}^{n \times d_v}
\end{aligned}
$$
**计算注意力分数**

定义分数矩阵 $S \in \mathbb{R}^{n\times n}$ ：
$$
S = \frac{QK^T}{\sqrt{d_k}}
$$
逐元素为：
$$
S_{ij} = \frac{q_i k_j^T}{\sqrt{d_k}}
$$
$S_{ij}$ 表示第 $i$ 个 token 对第 $j$ 个 token 的未归一化相关性分数，其中 $q_i$、$k_j$ 分别是 $Q$、$K$ 的行向量。

**加入 mask**
$$
S^\prime = S + M
$$

这里 $M\in\mathbb{R}^{n\times n}$ 为加性 mask：允许关注的位置取 $0$，屏蔽位置取 $-\infty$。因果 mask 在 $j>i$ 时取 $-\infty$，其余位置取 $0$；无 mask 时取 $M=0$。每行至少保留一个允许关注的位置。

**Softmax 归一化**
对 $S^{\prime}$ 的每一行应用 softmax，得到注意力权重矩阵 $A \in \mathbb{R}^{n \times n}$

$$
A = \operatorname{softmax}(S^\prime)
$$
其中
$$
A_{ij} = \frac{\exp(S_{ij}^{\prime})}{\sum_{l=1}^{n}\exp(S_{il}^{\prime})}
$$

**加权求和**
输出矩阵 ${}O \in \mathbb{R}^{n \times d_{v}}{}$ 为：
$$
O = AV
$$
逐行：
$$
o_{i} = \sum_{j=1}^{n}A_{ij}v_{j} \in \mathbb{R}^{1 \times d_{v}}
$$
第 $i$ 个 token 的输出 $o_i$ 是允许关注的 value 向量 $v_j$ 的加权平均，权重由 $A_{ij}$ 给出；被屏蔽位置的权重为 $0$。

# 旋转位置编码（RoPE）

## 1. 设计动机

正余弦位置编码具有相对位置结构，但将位置向量加到输入后，实际注意力点积为

$$
q_i k_j^T
=
(x_i+e_i)W_QW_K^T(x_j+e_j)^T.
$$

经过可学习投影后，位置对分数的作用一般不再仅依赖位置差。

RoPE 希望通过位置相关的变换，使注意力点积具有以下形式：

$$
\widetilde q_i\widetilde k_j^T
=
g(q_i,k_j,i-j).
$$

即：在保持内容向量不变时，位置对点积的作用只依赖相对位移。

RoPE 通过分别旋转 query 和 key，使它们在点积中自动产生相对旋转。

## 2. 符号定义

设序列长度为 $n$，模型维度为 $d$，输入矩阵为

$$
X\in\mathbb R^{n\times d}.
$$

第 $i$ 行为

$$
x_i\in\mathbb R^{1\times d}.
$$

定义投影矩阵

$$
W_Q,W_K\in\mathbb R^{d\times d_k},
\qquad
W_V\in\mathbb R^{d\times d_v}.
$$

首先计算不含显式位置变换的 query、key 和 value：

$$
\begin{aligned}
Q&=XW_Q\in\mathbb R^{n\times d_k},\\
K&=XW_K\in\mathbb R^{n\times d_k},\\
V&=XW_V\in\mathbb R^{n\times d_v}.
\end{aligned}
$$

其第 $i$ 行分别记为 $q_i,k_i,v_i$。

token 编号从 $1$ 开始，对应的位置编号定义为

$$
t_i=i-1.
$$

## 3. 二维情况：几何解释

暂时设 $d_k=2$。

对于非零二维向量，设 $\phi_{q_i}$ 和 $\phi_{k_j}$ 分别为
$q_i$ 和 $k_j$ 相对于横轴的方向角，则

$$
\begin{aligned}
q_i
&=\|q_i\|
[\cos\phi_{q_i},\sin\phi_{q_i}],\\
k_j
&=\|k_j\|
[\cos\phi_{k_j},\sin\phi_{k_j}].
\end{aligned}
$$

原点积为

$$
q_i k_j^T
=
\|q_i\|\,\|k_j\|
\cos(\phi_{q_i}-\phi_{k_j}).
$$

将 $q_i$ 和 $k_j$ 分别逆时针旋转 $\omega t_i$ 和
$\omega t_j$，得到 $\widetilde q_i$ 和 $\widetilde k_j$。

旋转不改变向量长度，因此

$$
\begin{aligned}
\widetilde q_i\widetilde k_j^T
&=
\|q_i\|\,\|k_j\|
\cos\left[
(\phi_{q_i}+\omega t_i)
-
(\phi_{k_j}+\omega t_j)
\right]\\
&=
\|q_i\|\,\|k_j\|
\cos\left[
(\phi_{q_i}-\phi_{k_j})
+\omega(i-j)
\right].
\end{aligned}
$$

内容决定原来的长度和方向，相对位置决定额外的角度差。

## 4. 二维情况：矩阵推导

由于采用行向量，定义右乘的旋转矩阵

$$
R_\omega(t)
=
\begin{bmatrix}
\cos(\omega t)&\sin(\omega t)\\
-\sin(\omega t)&\cos(\omega t)
\end{bmatrix}.
$$

对行向量 $[a,b]$，有

$$
[a,b]R_\omega(t)
=
[
a\cos(\omega t)-b\sin(\omega t),
a\sin(\omega t)+b\cos(\omega t)
].
$$

该操作将向量逆时针旋转 $\omega t$。

定义

$$
\widetilde q_i=q_iR_\omega(t_i),
\qquad
\widetilde k_j=k_jR_\omega(t_j).
$$

旋转矩阵满足

$$
R_\omega(t)^T=R_\omega(-t),
$$

以及

$$
R_\omega(a)R_\omega(b)=R_\omega(a+b).
$$

因此

$$
R_\omega(t_i)R_\omega(t_j)^T
=
R_\omega(t_i-t_j)
=
R_\omega(i-j).
$$

代入点积：

$$
\begin{aligned}
\widetilde q_i\widetilde k_j^T
&=
q_iR_\omega(t_i)
\left(k_jR_\omega(t_j)\right)^T\\
&=
q_iR_\omega(t_i)R_\omega(t_j)^Tk_j^T\\
&=
q_iR_\omega(i-j)k_j^T.
\end{aligned}
$$

相对位移 $i-j$ 显式出现在注意力点积中。

## 5. 推广到高维

假设 $d_k$ 为偶数。

将 $q_i,k_i$ 的分量两两分组。采用从 $1$ 开始的分量编号，
第 $r$ 组对应第 $2r+1$ 和第 $2r+2$ 个分量，其中

$$
r=0,1,\ldots,\frac{d_k}{2}-1.
$$

每组使用一个角频率：

$$
\omega_r=B^{-2r/d_k}.
$$

原始 RoPE 取 $B=10000$。

第 $r$ 组在位置 $i$ 上的旋转角度为

$$
\theta_{i,r}=t_i\omega_r.
$$

定义整体旋转矩阵

$$
R(t)
=
\operatorname{diag}
\left(
R_{\omega_0}(t),
R_{\omega_1}(t),
\ldots,
R_{\omega_{d_k/2-1}}(t)
\right)
\in\mathbb R^{d_k\times d_k}.
$$

对 query 和 key 分别进行旋转：

$$
\widetilde q_i=q_iR(t_i),
\qquad
\widetilde k_i=k_iR(t_i).
$$

具体到第 $r$ 组，query 的分量变为

$$
\begin{aligned}
\widetilde q_{i,2r+1}
&=
q_{i,2r+1}\cos\theta_{i,r}
-
q_{i,2r+2}\sin\theta_{i,r},\\
\widetilde q_{i,2r+2}
&=
q_{i,2r+1}\sin\theta_{i,r}
+
q_{i,2r+2}\cos\theta_{i,r}.
\end{aligned}
$$

key 使用相同的旋转规则。

## 6. 高维相对位置性质

每个二维旋转块都满足

$$
R_{\omega_r}(t_i)R_{\omega_r}(t_j)^T
=
R_{\omega_r}(t_i-t_j).
$$

因此，整体旋转矩阵满足

$$
R(t_i)R(t_j)^T
=
R(t_i-t_j)
=
R(i-j).
$$

于是

$$
\begin{aligned}
\widetilde q_i\widetilde k_j^T
&=
q_iR(t_i)R(t_j)^Tk_j^T\\
&=
q_iR(i-j)k_j^T.
\end{aligned}
$$

代入 $q_i=x_iW_Q$、$k_j=x_jW_K$，得到

$$
\widetilde q_i\widetilde k_j^T
=
x_iW_QR(i-j)W_K^Tx_j^T.
$$

无论投影参数如何学习，这里的显式位置作用都通过
$R(i-j)$ 表达。

## 7. 完整注意力计算

将旋转后的行向量组成矩阵：

$$
\widetilde Q
=
\begin{bmatrix}
q_1R(t_1)\\
\vdots\\
q_nR(t_n)
\end{bmatrix}
\in\mathbb R^{n\times d_k},
$$

$$
\widetilde K
=
\begin{bmatrix}
k_1R(t_1)\\
\vdots\\
k_nR(t_n)
\end{bmatrix}
\in\mathbb R^{n\times d_k}.
$$

计算注意力分数：

$$
S=\frac{\widetilde Q\widetilde K^T}{\sqrt{d_k}}.
$$

逐元素为

$$
S_{ij}
=
\frac{q_iR(i-j)k_j^T}{\sqrt{d_k}}.
$$

加入 mask：

$$
S'=S+M.
$$

按行进行 softmax：

$$
A=\operatorname{softmax}(S'),
$$

$$
A_{ij}
=
\frac{\exp(S'_{ij})}
{\sum_{l=1}^{n}\exp(S'_{il})}.
$$

标准 RoPE 不旋转 value，因此最终输出为

$$
O=AV\in\mathbb R^{n\times d_v},
$$

逐行为

$$
o_i=\sum_{j=1}^{n}A_{ij}v_j.
$$

## 8. 代码实现

以下为单个注意力头的二维输入示例，$q,k$ 的形状均为 `[n, d_k]`，且设备和数据类型相同；位置从 $0$ 开始。代码使用相邻分量配对，与上文推导一致。

```python
import torch

def rope(q , k , base = 10000.0):
    n , d_k = q.shape
    assert k.shape == q.shape and d_k % 2 == 0

    position = torch.arange(n , device=q.device)  # (n , )
    omega    = base**(
        -torch.arange(0 , d_k , 2 , device=q.device) / d_k
    )
    angles   = position[:, None] * omega[None, :] # (n , d_k / 2)

    cos, sin = angles.cos().to(q.dtype) , angles.sin().to(q.dtype)

    def rotate(x):
        a, b = x[:, 0::2] , x[: , 1::2]
        return torch.stack((a*cos - b*sin, a*sin + b*cos) , dim = -1).flatten(1)

    return rotate(q) , rotate(k)

```

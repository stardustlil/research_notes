---
aliases:
  - 残差连接
  - Pre-LN
  - Post-LN
---

# Residual（残差连接）

前置：[[LayerNorm]]、[[FFN]]

$$
X^{[\ell]}=X^{[\ell-1]}+F^{[\ell]}\left(\operatorname{Norm}\left(X^{[\ell-1]}\right)\right),\qquad \ell=1,\dots,L.
$$

## 动机

直接堆叠 $X^{[\ell]}=F^{[\ell]}(X^{[\ell-1]})$ 时，输出对输入的雅可比是 $L$ 个子层雅可比的连乘，范数随 $L$ 指数衰减或增长。残差在每一步加上恒等映射，使连乘展开后保留一项 $I$。

## 定义

$X^{[0]}\in\mathbb{R}^{n\times d}$ 为输入，$X^{[\ell]}\in\mathbb{R}^{n\times d}$ 为第 $\ell$ 个子层后的残差流。子层 $F^{[\ell]}$ 交替取 [[MHA]] 与 [[FFN]]（或 [[SwiGLU]]），一个 Transformer 层是两个子层，$L$ 为子层总数。$\operatorname{Norm}$ 取 [[LayerNorm]] 或 [[RMSNorm]]。

Pre-LN（归一化在支路上，最后再归一化一次）：

$$
\begin{aligned}
X^{[\ell]}&=X^{[\ell-1]}+F^{[\ell]}\left(\operatorname{Norm}\left(X^{[\ell-1]}\right)\right),\\
\text{输出}&=\operatorname{Norm}\left(X^{[L]}\right).
\end{aligned}
$$

Post-LN（原论文，归一化在主路径上）：

$$
X^{[\ell]}=\operatorname{Norm}\left(X^{[\ell-1]}+F^{[\ell]}\left(X^{[\ell-1]}\right)\right).
$$

## 推导

### Pre-LN 的输出是各子层增量之和

逐步代入：

$$
X^{[L]}=X^{[0]}+\sum_{\ell=1}^{L}F^{[\ell]}\left(\operatorname{Norm}\left(X^{[\ell-1]}\right)\right).
$$

每个子层读同一条残差流，把增量写回去。

### Pre-LN 的雅可比含恒等项

把 $X$ 按行展平，记支路 $F^{[\ell]}\circ\operatorname{Norm}$ 的雅可比为 $J^{[\ell]}\in\mathbb{R}^{nd\times nd}$（$\mathrm{d}F=\mathrm{d}X\,J^{[\ell]}$）。由 $\mathrm{d}X^{[\ell]}=\mathrm{d}X^{[\ell-1]}\left(I+J^{[\ell]}\right)$：

$$
\mathrm{d}X^{[L]}
=\mathrm{d}X^{[\ell]}\prod_{\ell'=\ell+1}^{L}\left(I+J^{[\ell']}\right)
=\mathrm{d}X^{[\ell]}\left(I+\sum_{\ell'=\ell+1}^{L}J^{[\ell']}+\cdots\right).
$$

对损失 $\mathcal{L}$，$\dfrac{\partial\mathcal{L}}{\partial X^{[\ell]}}$ 中含有 $\dfrac{\partial\mathcal{L}}{\partial X^{[L]}}$ 本身，与 $J^{[\ell']}$ 的大小无关。去掉残差时只剩 $\prod_{\ell'}J^{[\ell']}$，范数上界为 $\prod_{\ell'}\lVert J^{[\ell']}\rVert$。

### LayerNorm 的雅可比

取 $\gamma=\mathbb{1},\beta=0,\epsilon=0$，对单个 token $x\in\mathbb{R}^{1\times d}$。记 $P=I-\frac1d\mathbb{1}^\top\mathbb{1}$（对称、幂等），则 $y=xP/\sigma$，$\sigma^2=xPx^\top/d$：

$$
\begin{aligned}
\mathrm{d}\sigma&=\frac{\mathrm{d}x\,Px^\top}{d\,\sigma},\\
\mathrm{d}y&=\frac{\mathrm{d}x\,P}{\sigma}-\frac{xP\,\mathrm{d}\sigma}{\sigma^2}
=\frac{\mathrm{d}x\,P}{\sigma}-\frac{\mathrm{d}x\,(Px^\top)(xP)}{d\,\sigma^3}
=\mathrm{d}x\,\frac{1}{\sigma}\left(P-\frac{y^\top y}{d}\right).
\end{aligned}
$$

最后一步用 $xP=\sigma y$。因为 $y\mathbb{1}^\top=0$、$\lVert y\rVert^2=d$，$P-y^\top y/d$ 是到 $\{\mathbb{1},y\}$ 正交补的正交投影：秩 $d-2$，谱范数 $1$。所以 $J_{\text{LN}}=\frac1\sigma\left(P-\frac{y^\top y}{d}\right)$，谱范数 $1/\sigma$。

### Post-LN 的主路径不是恒等

记 $J^{[\ell]}_{\text{LN}}$ 为第 $\ell$ 步 Norm 的雅可比（按 token 分块对角），则

$$
\mathrm{d}X^{[L]}=\mathrm{d}X^{[\ell]}\prod_{\ell'=\ell+1}^{L}\left(I+J^{[\ell']}\right)J^{[\ell']}_{\text{LN}} .
$$

展开后每一项都乘了 $L-\ell$ 个 $J_{\text{LN}}$：每个都投影掉两个方向并缩放 $1/\sigma$，没有一项是 $I$。浅层梯度依赖各层 $\sigma$ 的连乘。

### Pre-LN 的残差流方差线性增长

设各增量均值为 $0$、彼此及与 $X^{[0]}$ 不相关。支路输入已归一化，增量的逐元素方差 $\delta^2$ 不随 $\ell$ 变：

$$
\operatorname{Var}\left(X^{[\ell]}\right)=\operatorname{Var}\left(X^{[0]}\right)+\ell\delta^2 .
$$

第 $\ell$ 个子层的增量占残差流方差的比例 $\delta^2/(\operatorname{Var}(X^{[0]})+\ell\delta^2)\sim1/\ell$，残差流尺度随 $\sqrt L$ 增长，所以输出前需要最后的 $\operatorname{Norm}$。

## 代码

```python
import torch
# 依赖：[[LayerNorm]] 的 layer_norm，[[MHA]] 的 mha，[[FFN]] 的 ffn

def pre_ln_step(x, sublayer, norm):
    return x + sublayer(norm(x))                                    # X^[ℓ]: [B, n, d]

def post_ln_step(x, sublayer, norm):
    return norm(x + sublayer(x))                                    # X^[ℓ]: [B, n, d]

def transformer_layer(x, W_q, W_k, W_v, W_o, W_1, W_2, H, norm, step=pre_ln_step):
    x = step(x, lambda u: mha(u, W_q, W_k, W_v, W_o, H, causal=True), norm)
    return step(x, lambda u: ffn(u, W_1, W_2), norm)               # 两个子层

# 验证：子层输出为 0 时，Pre-LN 是恒等，Post-LN 仍要过 Norm
B, n, d, H, d_ff = 2, 5, 16, 4, 64
norm = lambda u: layer_norm(u, torch.ones(d), torch.zeros(d), eps=0.0)
x = torch.randn(B, n, d)
W_q, W_k, W_v, W_1 = torch.randn(d, d), torch.randn(d, d), torch.randn(d, d), torch.randn(d, d_ff)
W_o, W_2 = torch.zeros(d, d), torch.zeros(d_ff, d)
args = (W_q, W_k, W_v, W_o, W_1, W_2, H, norm)
assert torch.equal(transformer_layer(x, *args, step=pre_ln_step), x)
assert torch.allclose(transformer_layer(x, *args, step=post_ln_step), norm(x), atol=1e-5)

# 验证：LayerNorm 的雅可比
x1 = torch.randn(d)
y, sigma = norm(x1), x1.std(unbiased=False)
P = torch.eye(d) - torch.ones(d, d) / d
J = (P - torch.outer(y, y) / d) / sigma
assert torch.allclose(J, torch.autograd.functional.jacobian(norm, x1).T, atol=1e-5)  # .T：dy = dx J
assert torch.linalg.matrix_rank(J) == d - 2
assert torch.isclose(torch.linalg.matrix_norm(J, 2), 1 / sigma, rtol=1e-4)
```

## 辨析

- `torch.autograd.functional.jacobian` 返回 $\partial y_b/\partial x_a$ 排在 `[b, a]`，是本库行向量约定下 $J$ 的转置。
- Pre-LN 的最后一个 $\operatorname{Norm}$ 不能省：GPT-2 中为 `ln_f`。

---
aliases:
  - 位置编码
tags:
  - 索引
  - Transformer/位置编码
---

# 位置编码

上级：[[首页|笔记首页]]

> [!abstract] 本主题讨论什么
> 无 mask 的自注意力满足 $f(\Pi X)=\Pi f(X)$（[[Self-Attention#置换等变性|置换等变性]]），无法区分 token 的位置。位置编码负责把位置信息注入进来：要么加到输入 $X$ 上，要么作用在 $q,k$ 上。

## 学习顺序

[[Self-Attention#置换等变性|置换等变性]] → [[Learned Positional Embedding|可学习绝对位置编码]] → [[Sinusoidal positional encoding|正余弦位置编码]] → [[RoPE|旋转位置编码]] → [[YaRN|长上下文扩展]]

## 笔记导航

- [[Learned Positional Embedding|可学习绝对位置编码]]：学习位置矩阵 $P$，$\tilde X=X+P_{1:n}$。
- [[Sinusoidal positional encoding|正余弦位置编码]]：固定的多频率 $\operatorname{PE}(t)$；平移 $\Delta$ 等价于旋转 $R(-\Delta)$。
- [[RoPE|RoPE（旋转位置编码）]]：旋转 $q_i,k_j$，使分数显式依赖 $R(i-j)$。
- 长上下文扩展
	- [[YaRN|YaRN]]：在 RoPE 上做分频插值与温度校准，把 $n_{\text{train}}$ 扩展到 $n_{\text{ext}}$。

## 快速对比

| 方法 | 作用对象 | 参数 | 位置以什么形式进入分数 | 超出训练长度 |
| --- | --- | --- | --- | --- |
| [[Learned Positional Embedding\|可学习绝对位置编码]] | 输入 $X$（相加） | $P\in\mathbb{R}^{n_{\max}\times d}$ | 绝对位置 $p_i,p_j$，经投影混合 | 无法计算 |
| [[Sinusoidal positional encoding\|正余弦位置编码]] | 输入 $X$（相加） | 无 | 绝对位置；经投影后一般不只依赖 $i-j$ | 可计算，效果不保证 |
| [[RoPE]] | $q_i,k_j$（旋转） | 无 | 显式 $q_iR(i-j)k_j^\top$ | 可计算，效果不保证 |
| [[YaRN]] | RoPE 频率 $\omega_r$ 与温度 $\tau$ | 无新增参数（需少量继续训练） | 同 RoPE，频率换成 $\omega'_r$ | 目标长度 $n_{\text{ext}}$ |

## 相关主题

- [[Attention|注意力]]：位置编码改变的是注意力分数，先理解 $Q,K,V$ 与 mask。

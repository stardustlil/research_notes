---
aliases:
  - 注意力
  - 注意力机制
tags:
  - 索引
  - Transformer/注意力
---

# 注意力

上级：[[首页|笔记首页]]

> [!abstract] 本主题讨论什么
> 从输入 $X$ 出发，经 $Q,K,V$、缩放点积、mask、softmax 得到注意力输出；再从单头扩展到多头。

## 学习顺序

[[Self-Attention]] → [[MHA]] → 进入 [[Position Encode|位置编码]]，看位置信息如何影响注意力分数。

## 笔记导航

- [[Self-Attention|Self-Attention（自注意力）]]：单头计算的五个步骤。
	- [[Self-Attention#计算 Q、K、V|计算 Q、K、V]]
	- [[Self-Attention#缩放点积分数|缩放点积分数]]
	- [[Self-Attention#加入 mask|加入 mask]]
	- [[Self-Attention#Softmax 归一化|softmax 归一化]]
	- [[Self-Attention#加权求和|加权求和]]
	- 性质：[[Self-Attention#为什么要缩放|为什么除以 √d_k]]、[[Self-Attention#置换等变性|置换等变性]]
- [[MHA|MHA（多头注意力）]]：$H$ 个头并行计算，拼接后用 $W_O$ 融合。

## 快速对比

| 结构 | 投影参数 | 注意力权重 | 输出 |
| --- | --- | --- | --- |
| [[Self-Attention]]（单头） | $W_Q,W_K\in\mathbb{R}^{d\times d_k}$，$W_V\in\mathbb{R}^{d\times d_v}$ | 一个 $A\in\mathbb{R}^{n\times n}$ | $O=AV\in\mathbb{R}^{n\times d_v}$ |
| [[MHA]]（多头） | 每头一组 $W_Q^{(h)},W_K^{(h)},W_V^{(h)}$，再加 $W_O\in\mathbb{R}^{Hd_v\times d}$ | $H$ 个 $A^{(h)}$ | $Y=O_{\text{cat}}W_O\in\mathbb{R}^{n\times d}$ |

## 相关主题

- [[Position Encode|位置编码]]：注意力具有置换等变性，需要额外注入位置信息。
- [[Normalization|归一化]]：网络中与注意力配合使用的归一化方法。

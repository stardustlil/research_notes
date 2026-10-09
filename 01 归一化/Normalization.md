---
aliases:
  - 归一化
tags:
  - 索引
  - Transformer/归一化
---

# 归一化

上级：[[首页|笔记首页]]

> [!abstract] 本主题讨论什么
> 三种归一化都是「在一个统计集合 $\mathcal{S}$ 上算统计量，标准化，再做可学习的缩放」，区别只在于 $\mathcal{S}$ 取哪些元素、是否减均值、推理时用什么统计量。

## 学习顺序

[[BatchNorm]] → [[LayerNorm]] → [[RMSNorm]]

阅读时重点比较：统计集合 $\mathcal{S}$ 是什么、是否减去均值、有哪些可学习参数、训练与推理是否一致。

## 笔记导航

- [[BatchNorm|BatchNorm（批归一化）]]：每个通道在 batch 与空间维度上统计；推理时使用滑动统计量。
- [[LayerNorm|LayerNorm（层归一化）]]：每个 token 在自身 $d$ 个特征上统计；具有平移与缩放不变性。
- [[RMSNorm|RMSNorm（均方根归一化）]]：不减均值、不学 $\beta$，只用 RMS 缩放到半径 $\sqrt d$ 的球面。

## 快速对比

| 方法 | 统计集合 $\mathcal{S}$ | $m=\lvert\mathcal{S}\rvert$ | 减均值 | 可学习参数 | 推理时的统计量 |
| --- | --- | --- | --- | --- | --- |
| [[BatchNorm]] | 一个通道的全部 $(B,H_{\text{img}},W_{\text{img}})$ 元素 | $BH_{\text{img}}W_{\text{img}}$ | 是 | $\gamma,\beta\in\mathbb{R}^{C}$ | 训练中的滑动统计量 |
| [[LayerNorm]] | 一个 token 的 $d$ 个特征 | $d$ | 是 | $\gamma,\beta\in\mathbb{R}^{1\times d}$ | 当前 token 的统计量 |
| [[RMSNorm]] | 一个 token 的 $d$ 个特征 | $d$ | 否 | $\gamma\in\mathbb{R}^{1\times d}$ | 当前 token 的 RMS |

表中 LayerNorm 与 RMSNorm 取 `normalized_shape = d`；BatchNorm 为 PyTorch 默认行为（维护滑动统计量）。

## 相关主题

- [[Attention|注意力]]：归一化通常与注意力子层配合使用。

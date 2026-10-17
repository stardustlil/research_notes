---
aliases:
  - KV 缓存
  - KV Cache
---

# KV-Cache（KV 缓存）

前置：[[MHA]]、[[Residual]]

$$
o_i=\operatorname{softmax}\left(\frac{q_iK_{1:i}^\top}{\sqrt{d_k}}\right)V_{1:i},\qquad
K_{1:i}=\begin{bmatrix}K_{1:i-1}\\ k_i\end{bmatrix},\quad
V_{1:i}=\begin{bmatrix}V_{1:i-1}\\ v_i\end{bmatrix}.
$$

## 动机

自回归生成第 $i$ 个输出时，因果注意力需要前 $i$ 个 token 在每一层的 $k_j,v_j$。不缓存就要把整个前缀重新过一遍所有层：第 $i$ 步重复计算前 $i-1$ 步已经算过的全部投影。

## 定义

单头记号，省略 batch；多头时每个头各存一份。每个注意力子层保存缓存 $K_{1:i-1}\in\mathbb{R}^{(i-1)\times d_k}$、$V_{1:i-1}\in\mathbb{R}^{(i-1)\times d_v}$。

Prefill：输入 prompt $X\in\mathbb{R}^{n\times d}$，按 [[MHA]] 做一次带因果 mask 的完整计算，同时写入缓存：

$$
K_{1:n}=XW_K,\qquad V_{1:n}=XW_V.
$$

Decode：第 $i$ 步（$i>n$）只输入一个 token $x_i\in\mathbb{R}^{1\times d}$：

$$
\begin{aligned}
q_i&=x_iW_Q,\quad k_i=x_iW_K,\quad v_i=x_iW_V,\\
K_{1:i}&=\begin{bmatrix}K_{1:i-1}\\ k_i\end{bmatrix},\quad V_{1:i}=\begin{bmatrix}V_{1:i-1}\\ v_i\end{bmatrix},\\
o_i&=\operatorname{softmax}\left(\frac{q_iK_{1:i}^\top}{\sqrt{d_k}}\right)V_{1:i}\in\mathbb{R}^{1\times d_v}.
\end{aligned}
$$

这里的 $x_i$ 是该子层的输入 $\operatorname{Norm}(x_i^{[\ell-1]})$。query 只有一行，不需要 $M$。

## 推导

### 因果注意力第 i 行只依赖本行 query 和前 i 行 K、V

$j>i$ 时 $M_{ij}=-\infty$，所以 $A_{ij}=0$。于是 softmax 只在 $j\le i$ 上归一化：

$$
o_i=\sum_{j=1}^{i}\frac{\exp(S_{ij})}{\sum_{j'=1}^{i}\exp(S_{ij'})}v_j
=\operatorname{softmax}\left(\frac{q_iK_{1:i}^\top}{\sqrt{d_k}}\right)V_{1:i}.
$$

这就是标题下的公式。$o_i$ 只用到 $q_i$；历史的 $q_j$（$j<i$）只在算 $o_j$ 时用过一次，所以不用缓存 Q。

### 前缀的表示不随后续 token 改变

对残差步 $\ell$ 归纳证明：$X^{[\ell]}$ 的第 $i$ 行只依赖 $x_1^{[0]},\dots,x_i^{[0]}$。

- $\ell=0$：嵌入加位置编码是逐 token 的。
- 归纳：$\operatorname{Norm}$ 和前馈都是逐行运算；由上一条，注意力输出的第 $i$ 行只依赖输入的前 $i$ 行；残差也是逐行相加。

所以追加 $x_{i+1}$ 后，任意层前 $i$ 行的 $K,V$ 都不变，缓存一直有效。

### 计算量从三次降到二次

取 $d_k=d_v=d/H$，只统计一个注意力子层的乘加次数。第 $i$ 步：

|                  | 投影 $W_Q,W_K,W_V,W_O$ | $q K^\top$ 与 $AV$ |
| ---------------- | -------------------- | ----------------- |
| 无缓存（重算 $i$ 行）    | $4id^2$              | $2i^2d$           |
| KV 缓存（只算第 $i$ 行） | $4d^2$               | $2id$             |

对 $i=1,\dots,N$ 求和：无缓存为 $2N^2d^2+\tfrac23N^3d$，KV 缓存为 $4Nd^2+N^2d$（均只保留最高次项）。

### 缓存大小

每个 token 在每个注意力子层存 $k_i,v_i$，共 $H(d_k+d_v)$ 个元素。注意力子层数是 $L/2$，取 $d_k=d_v$ 时一条长 $n$ 的序列共存

$$
\frac L2\cdot 2Hd_k\cdot n=LHd_kn
$$

个元素，batch 为 $B$ 时再乘 $B$。换成字节，并把层数 $L/2$ 和 $Hd_k=d$ 直接写出来：

$$
\text{缓存字节数}=\underbrace{2}_{K,V}\times\underbrace{L/2}_{\text{层数}}\times\underbrace{Hd_k}_{=d}\times\underbrace{n}_{\text{序列长}}\times\underbrace{B}_{\text{batch}}\times\underbrace{2}_{\text{fp16 字节}}.
$$

以 Llama-2-7B 为例（$32$ 层，$H=32$，$d_k=128$，$d=4096$，fp16）：

| 情况 | 大小 |
| --- | --- |
| 1 个 token | $2\times32\times4096\times2$ 字节 $=512$ KiB |
| 1 条序列，4096 token | $2$ GiB |
| 8 条序列，各 4096 token | $16$ GiB |

### 解码时注意力的算术强度是常数

第 $i$ 步每条序列要从显存读出 $K_{1:i},V_{1:i}$，共 $2id$ 个元素，做 $2id$ 次乘加，比值是 $1$，与 $B$ 无关，因为每条序列有自己的缓存。投影读 $4d^2$ 个权重，做 $4Bd^2$ 次乘加，比值是 $B$。所以增大 batch 能摊薄权重的读取，却摊不薄缓存的读取：解码阶段的注意力受显存带宽限制。

## 代码

```python
import math
import torch

def split_heads(z, H):
    B, n, d = z.shape
    return z.view(B, n, H, d // H).transpose(1, 2)                  # [B, n, d] -> [B, H, n, d_k]

def prefill(x, W_q, W_k, W_v, W_o, H):
    # 依赖：[[MHA]] 中的 mha
    # x: [B, n, d]；返回 Y 和缓存 K_{1:n}, V_{1:n}: [B, H, n, d_k]
    return mha(x, W_q, W_k, W_v, W_o, H, causal=True), split_heads(x @ W_k, H), split_heads(x @ W_v, H)

def decode_step(x_i, W_q, W_k, W_v, W_o, H, k_cache, v_cache):
    # x_i: [B, 1, d]；k_cache, v_cache: [B, H, i-1, d_k]
    B, _, d = x_i.shape
    q, k, v = (split_heads(x_i @ W, H) for W in (W_q, W_k, W_v))    # q_i, k_i, v_i: [B, H, 1, d_k]
    k_cache = torch.cat([k_cache, k], dim=2)                        # K_{1:i}: [B, H, i, d_k]
    v_cache = torch.cat([v_cache, v], dim=2)                        # V_{1:i}: [B, H, i, d_v]
    scores = q @ k_cache.transpose(-2, -1) / math.sqrt(d // H)      # [B, H, 1, i]，不需要 mask
    out = torch.softmax(scores, dim=-1) @ v_cache                   # o_i: [B, H, 1, d_v]
    return out.transpose(1, 2).reshape(B, 1, d) @ W_o, k_cache, v_cache   # [B, 1, d]

# 验证：prefill 前 n_0 个 token 再逐个 decode，等于整段做因果 MHA
B, n, n_0, d, H = 2, 7, 3, 16, 4
x = torch.randn(B, n, d)
W_q, W_k, W_v, W_o = (torch.randn(d, d) for _ in range(4))
y, k_cache, v_cache = prefill(x[:, :n_0], W_q, W_k, W_v, W_o, H)
ys = [y]
for i in range(n_0, n):
    y_i, k_cache, v_cache = decode_step(x[:, i:i+1], W_q, W_k, W_v, W_o, H, k_cache, v_cache)
    ys.append(y_i)
assert torch.allclose(torch.cat(ys, dim=1), mha(x, W_q, W_k, W_v, W_o, H, causal=True), atol=1e-4)

# 逐步追加的缓存等于整段的 K、V（前缀不变）
assert torch.allclose(k_cache, split_heads(x @ W_k, H)) and torch.allclose(v_cache, split_heads(x @ W_v, H))

# 缓存大小：每个注意力子层 2 n H d_k 个元素；Llama-2-7B 每个 token 512 KiB
assert k_cache.numel() + v_cache.numel() == B * 2 * n * H * (d // H)
assert 64 * 32 * 128 * 2 == 512 * 1024
```

## 辨析

- [[RoPE]] 的缓存存的是旋转后的 $\tilde k_j=k_jR(t_j)$，新 token 的位置是 $t_i=i-1$，也就是当前缓存的长度。[[Learned-PE]] 和 [[Sinusoidal-PE]] 加在输入上，缓存不需要特殊处理。
- `F.scaled_dot_product_attention(q, k, v, is_causal=True)` 的 mask 按左上角对齐：query 长 $1$、key 长 $i$ 时，它只允许看第 $1$ 个 key。decode 时要传 `is_causal=False`。
- MQA/GQA 让多个 query 头共享一组 $K,V$，共 $H_{\text{kv}}<H$ 组；缓存大小按 $H_{\text{kv}}$ 计算，即 $LH_{\text{kv}}d_kn$。HuggingFace `past_key_values` 每层的形状是 `[B, H_kv, n, d_k]`。

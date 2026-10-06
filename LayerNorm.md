**归一化有助于稳定深层网络的训练。Transformer 通常采用 LayerNorm，因为它对每个 token 的特征独立归一化，不依赖 batch 统计量，更适合变长序列和自回归建模。**

## 公式

设 Transformer 输入：
$$
x \in \mathbb{R}^{B\times L\times D}
$$
当 `normalized_shape=D` 时，LayerNorm 对最后一维 $D$ 做归一化。对每个样本 $b$、每个位置 $l$：
$$
\begin{aligned}
\mu_{b,l} &= \frac{1}{D} \sum_{d=1}^{D}{x_{b,l,d}} \\
\sigma^{2}_{b,l} &= \frac{1}{D} \sum_{d = 1}^{D}(x_{b,l,d} - \mu_{b,l})^2 \\

\hat{x}_{b,l,d} &= \frac{x_{b,l,d} - \mu_{b,l}}{\sqrt{\sigma_{b,l}^2 + \epsilon}} \\
y_{b,l,d} &= \gamma_d \hat{x}_{b,l,d} + \beta_d

\end{aligned}
$$

其中：

- $\gamma,\beta\in\mathbb{R}^{D}$ 是可学习参数，所有样本和 token 共享；
- $\epsilon>0$ 是保证数值稳定的小常数，位于根号内；
- 均值和方差形状为 `[B, L, 1]`，每个 token 独立计算。

训练和推理都使用当前输入的统计量，无需维护滑动均值或滑动方差。

## 代码
```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self , normalized_shape , eps=1e-5):
        super().__init__()
        self.eps = eps
        #转换成元组
        if isinstance(normalized_shape , int):
            normalized_shape = (normalized_shape , )
        self.normalized_shape = tuple(normalized_shape)

        self.gamma = nn.Parameter(torch.ones(*self.normalized_shape))
        self.beta  = nn.Parameter(torch.zeros(*self.normalized_shape))

    def forward(self , x):
        dims = tuple(range(-len(self.normalized_shape) , 0))
        mean = x.mean(dim = dims , keepdim = True)
        var  = x.var( dim = dims , unbiased = False , keepdim = True)

        x_hat = (x-mean)/torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
```

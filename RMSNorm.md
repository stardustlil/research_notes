RMSNorm 不计算输入的均值，也不做去均值，只用均方根进行缩放，减少了计算步骤。常见实现只学习缩放参数 $\gamma$，不学习偏置 $\beta$，因此比同时学习 $\gamma$ 和 $\beta$ 的 LayerNorm 参数更少；实际速度取决于实现和硬件。

## 公式

不做去均值，计算加入数值稳定项 $\epsilon$ 的均方根（RMS）：
$$
\begin{aligned}
RMS(x) &= \sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2+ \epsilon} \\

y_i &= \gamma_i \frac{x_i}{RMS(x)}
\end{aligned}
$$

其中 $n$ 为归一化区域的元素数。对于形状 `[B, L, D]` 的输入、`normalized_shape=D`，每个 token 独立计算 RMS，此时 $n=D$。

## 代码
```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self , normalized_shape , eps = 1e-5):
        super().__init__()
        self.eps = eps
        if isinstance(normalized_shape , int):
            normalized_shape = (normalized_shape , )
        self.normalized_shape = tuple(normalized_shape)

        self.gamma = nn.Parameter(torch.ones(*self.normalized_shape))

    def forward(self , x):
        dims = tuple(range(-len(self.normalized_shape) , 0))
        mean_square = torch.mean(x*x , dim = dims , keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)

        #归一化并缩放
        x_hat = x / rms
        return self.gamma * x_hat

```

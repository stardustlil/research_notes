
## 公式

设一个 batch 里有 $m$ 个样本，某个特征维度上的值为 $x_1, x_2, \dots, x_m$。对于下方的 BatchNorm2d，每个通道在 batch 和空间维度 $(N,H,W)$ 上统计，此时 $m=NHW$。

$$
\begin{aligned}
\mu_B &= \frac{1}{m} \sum_{i=1}^{m} x_i \\
\sigma_B^2 &= \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
\end{aligned}
$$
然后归一化：
$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$
最后做仿射变换：

$$
y_i = \gamma \hat{x}_i + \beta
$$

其中 $\mu_B$ 是 batch 均值，$\sigma_B^2$ 是 batch 方差， $\gamma$ 与 $\beta$ 是可学习参数。

## 注意

默认情况下，BatchNorm 在训练时使用当前 batch 的均值和方差，并采用 EMA（指数滑动平均）更新 `running_mean` 和 `running_var`，供推理时使用。$\gamma$ 与 $\beta$ 是通过梯度下降学习的仿射参数，不由 EMA 估计。

按照 PyTorch 的约定，训练时归一化使用有偏方差（除以 $m$），而更新 `running_var` 使用无偏方差（除以 $m-1$）。下方代码实现这一默认行为，训练时要求每个通道的统计元素数 $m>1$。

## 代码
```python
import torch
import torch.nn as nn

class MyBatchNorm2d(nn.Module):
    def __init__(self , num_features , eps = 1e-5 , momentum = 0.1):
        super().__init__() # set self.training = true
        self.eps = eps
        self.momentum = momentum

        #beta and gamma
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        #EMA滑动平均
        self.register_buffer("running_mean" , torch.zeros(num_features))
        self.register_buffer("running_var" , torch.ones(num_features))


    def forward(self , x):
        # x : (N , C , H , W)
        N , C , H , W = x.shape
        if self.training:
            m = N * H * W
            if m <= 1:
                raise ValueError("训练时每个通道必须有至少两个统计元素")
            current_mean = x.mean(dim = (0 , 2 , 3) , keepdim = True)
            current_var  = x.var( dim = (0 , 2 , 3) , unbiased = False , keepdim = True)

            with torch.no_grad():
                # running_var 使用无偏方差更新
                unbiased_var = current_var * m / (m - 1)

                self.running_mean.mul_(1-self.momentum).add_(self.momentum*current_mean.view(-1))
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased_var.view(-1))

        else:
            current_mean = self.running_mean.view(1 , C , 1 , 1)
            current_var  = self.running_var.view(1, C , 1 , 1)

        x_hat = (x - current_mean) / torch.sqrt(current_var + self.eps)
        out   = x_hat * self.gamma.view(1 , C , 1 , 1) + self.beta.view(1 , C , 1 , 1)
        return out
```

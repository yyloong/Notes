### ELBO

#### 问题引入
对于一个生成式模型，通常被建模为
$$min_\theta\mathbb{E}_{x\sim p_{data}}[KL(p_{data}(x)\|p_\theta(x))]=min_\theta\mathbb{E}_{x\sim p_{data}}[log\frac{p_{data}(x)}{p_\theta(x)}]$$

$$min_\theta\mathbb{E}_{x\sim p_{data}}[KL(p_{data}(x)\|p_\theta(x))]\Leftrightarrow max_{\theta}\mathbb{E}_{x\sim p_{data}}[logp_\theta(x)]$$

即在数据分布中进行采样并且学习参数$\theta$估计采样出来的数据的对数似然最大
而为了实现"生成"的目的,往往需要假设随机变量$x$与某些隐藏的分布比较简单的随机变量$z$相关,这样我们就可以通过先根据$p(z)$采样$z$,再根据$p(x\mid z)$采样可以得到$x$

优化问题变为
$$max_{\theta}\mathbb{E}_{x\sim p_{data}}[log\int p_\theta(x,z)dz] \tag{1}$$ 

#### 问题求解

##### 简单蒙特卡洛估计
对于$(1)$,一个直觉的想法是直接对$z$进行采样,然后用蒙特卡洛估计来近似目标函数

$$\mathbb{E}_{x\sim p_{data}}[log\int p_\theta(x,z)dz] = \mathbb{E}_{x\sim p_{data}}[log\int p_\theta(x\mid z)p(z)dz] \approx \frac{1}{N(x)}\frac{1}{N(z)}\sum_{x\sim p_{data}}\sum_{z\sim p} logp_\theta(x\mid z)$$

但是这个不是一个无偏估计
$$\mathbb{E}_{x\sim p_{data}}[log\int p_\theta(x\mid z)p(z)dz]=\mathbb{E}_{x\sim p_{data}}[log \mathbb{E}_{z\sim p}[p_\theta(x\mid z)]] \geq \mathbb{E}_{x\sim p_{data},z\sim p}[log p_\theta(x\mid z)]$$

另一个想法,先估计内层期望,再估计外层期望
$$\mathbb{E}_{x\sim p_{data}}[log\int p_\theta(x,z)dz] = \mathbb{E}_{x\sim p_{data}}[log\int p_\theta(x\mid z)p(z)dz] \approx \frac{1}{N(x)}\sum_{x\sim p_{data}} log\frac{1}{N(z)}\sum_{z\sim p}p_\theta(x\mid z)$$
但是当$z$采样次数无法达到无穷时,内层得到的始终不是真实的期望,最终得到的只是一个渐进的无偏估计

###### 衡量蒙特卡洛的偏差程度
$$\mathbb{E}_{x\sim p_{data}}[log\int p_\theta(x\mid z)p(z)dz]=\mathbb{E}_{x\sim p_{data}}[log \mathbb{E}_{z\sim p}[p_\theta(x\mid z)]] \geq \mathbb{E}_{x\sim p_{data},z\sim p}[log p_\theta(x\mid z)]$$
考虑二者相减得到偏差的程度
$$\mathbb{E}_{x\sim p_{data}}[log\int p_\theta(x\mid z)p(z)dz]-\mathbb{E}_{x\sim p_{data},z\sim p}[log p_\theta(x\mid z)]$$
$$=\mathbb{E}_{x\sim p_{data}}[\int p(z)[logp_\theta(x)-logp_\theta(x\mid z)]dz]$$
$$=\mathbb{E}_{x\sim p_{data},z\sim p}[log\frac{p_\theta(x)}{p_\theta(x\mid z)}]=\mathbb{E}_{x\sim p_{data},z\sim p}[log\frac{p(z)}{p_\theta(z\mid x)}]$$
$$=\mathbb{E}_{x\sim p_{data},z\sim p}[KL(p(z)\|p_\theta(z\mid x))]$$
即偏差项为$p(z)$和$p_\theta(z\mid x)$的$KL$散度
由于$p_\theta(z\mid x)$可能非常复杂而导致$KL$项很大,问题难以优化
此外,对$p(z)$在整个分布空间进行采样会导致采样到很多低概率的点,采样效率很低,需要的样本数量极大增加

##### 重要性采样和变分推断

由于直接从整个分布空间进行采样效果很差,考虑采样某些比较重要的$z$(出现频率高),将新的分布记为$q(z)$

$$\mathbb{E}_{x\sim p_{data}}[log\int p_\theta(x,z)dz]=\mathbb{E}_{x\sim p_{data}}[log\int q(z)\frac{p_\theta(x,z)}{q(z)}dz]$$
由琴生不等式
$$\mathbb{E}_{x\sim p_{data}}[log\int q(z)\frac{p_\theta(x,z)}{q(z)}dz] \geq \mathbb{E}_{x\sim p_{data}}[\int q(z)log\frac{p_\theta(x,z)}{q(z)}dz]=\mathbb{E}_{x\sim p_{data},z\sim q}[log\frac{p_\theta(x,z)}{q(z)}]$$
得到$ELBO$不等式
误差项为
$$\mathbb{E}_{x\sim p_{data}}[log p_\theta(x)]-\mathbb{E}_{x\sim p_{data},z\sim q}[log\frac{p_\theta(x,z)}{q(z)}]=\mathbb{E}_{x\sim p_{data},z\sim q}[KL(q(z)\|p_\theta(z\mid x))]$$
(看起来和前面好像没啥区别,但是如果$p(z)$为了方便采样通常是一个很简单的分布,但是$q(z)$可以通过构造或者神经网络拟合增加复杂度从而拟合更加复杂的分布,对$q(z)$的不同建模可以引出不同的算法)

##### VAE
VAE的$q(z)$定义为一个参数待学习的高斯分布$\mathcal{N}(\mu,\sigma^2\mathcal{I}),$其中的$\mu,\sigma$用神经网络来拟合,同时为了引入$x,z$的关系,神经网络建模为$q_\phi(z\mid x)$,$p(z)$则建模为$\mathcal{N}(0,1)$
则问题变为
$$\mathbb{E}_{x\sim p_{data}}[log p_\theta(x)]=\mathbb{E}_{x\sim p_{data},z\sim q_\phi}[log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}]+\mathbb{E}_{x\sim p_{data},z\sim q_\phi}[KL(q_\phi(z)\|p_\theta(z\mid x))]$$

我们主要优化$ELBO$下界$\mathbb{E}_{x\sim p_{data},z\sim q_\phi}[log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}]=\mathbb{E}_{x\sim p_{data},z\sim q_\phi}[logp_\theta(x\mid z)p(z)-logq_\phi(z\mid x)]=\mathbb{E}_{x\sim p_{data}}[logp_\theta(x\mid z)]-\mathbb{E}_{x\sim p_{data}}[logq_\phi(z\mid x)-logp(z)]$
故损失函数$$Loss = \mathbb{E}_{x\sim p_{data}}[-logp_\theta(x\mid z)]+\mathbb{E}_{x\sim p_{data}}[KL(q_\phi(z\mid x)\|p(z))]$$
直观理解:
前一项为最大化$p_\theta(x\mid z)$的对数似然,相当于希望模型生成内容和实际内容保持一致,第二项为概率散度,相当于希望模型尽量拟合$p(z)$
而$ELBO$和原目标的gap相当于希望$q$还能尽可能拟合$p_\theta(z\mid x)$

###### 实际操作
我们需要学习两个判别式网络,先要学习从$x$到$z$,然后还要学习从$z$到$x$
这类似于一个自编码器,所以VAE全称为 Variational AutoEncoder(变分自编码器)

具体的数据流向
先从数据中采样$x_0$,经过$q_\phi$得到$\mu_0,\sigma_0$表示$q_\phi(z\mid x) \approx \mathcal{N}(\mu_0,\sigma_0)$,然后从$\mathcal{N}(\mu_0,\sigma_0)$采样得到一个$z_0$,计算$log\frac{\mathcal{N}(x_0,\mu_0,\sigma_0)}{\mathcal{N}(x_0,\mathbf{0},\mathbf{I})}$得到KL损失
假设$x$服从某个带参的分布,
然后$z_0$经过$p_\theta$网络得到分布的参数,并计算$-logp_\theta(x\mid z)$得到重构项

如果假设$x$的各个元素之间相互独立且服从高斯分布
则$-log(x\mid z) = -log\frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{\|x-\mu_\theta(z)\|^2}{2\sigma^2})$
$-log(x\mid z) \propto \|x-\mu_\theta(z)\|^2$
得到均方误差损失
如果假设$x$是$0,1$数据且各个元素服从
$$p_\theta(x_i \mid z) = y_i(z)^{x_i}(1-y_i(z))^{1-x_i}$$
则得到交叉熵损失
$$-logp_\theta(x_i\mid z) = x_ilogy_i(z) + (1-x_i)log(1-y_i(z))$$

###### 梯度问题
由于中间的$z_0$是采样得到的,导致实际计算过程中到这里梯度会直接断掉,通常使用重参数化技巧来解决这个问题,保持$\mu_\theta,\sigma_\theta$的梯度,而从$\mathcal{N}(\mathcal{0},\mathcal{I})$中采样一个随机变量$\epsilon$,得到
$$z_0 = \mu_\theta + \sigma_\theta \epsilon$$
这和直接采样得到的$z_0$分布一致,但是梯度可以正常传递

##### DDPM

DDPM 不使用神经网络来拟合$q_\phi$,而是直接定义了从$x$到$z$的分布$q(z\mid x)$,而且该分布可以解析的求解$q(x \mid z)$,则只需要用一个神经网络去拟合$q(x\mid z)$即可

###### 扩散过程
DDPM 假设因变量$z$为一系列的$x_1,x_2,...x_T$,设原数据为$x_0$,$x_i$为$x_{i-1}$通过一步高斯加噪获得,即$x_0$通过逐渐加噪最终得到一个完全高斯分布$x_T$,神经网络需要学习的是这个逆向过程
问题变为

$$logp_\theta(x_0) = \mathbb{E}_{x_{1:T}\sim q(x_{1:T}\mid x_0)}[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}] + KL(q(x_{1:T}\mid x_0)\| p_\theta(x_{1:T}\mid x_0))$$

这里的$KL$项可以理解为如果$p_\theta$逆向过程学习的比较好的话,对应的前向过程也应该会比较小

优化目标变为
$$\mathbb{E}_{x_{1:T}\sim q(x_{1:T}\mid x_0)}[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}]$$

$q(x_{1:T} \mid x_0)$是前向过程的概率,为了让神经网络学习反向过程的概率需要用贝叶斯公式进行变换

$$q(x_{1:T}\mid x_0) = q(x_1\mid x_0)\prod_{t=1}^{T-1} q(x_{t+1}\mid x_t,x_0)$$
$$=q(x_0 \mid x_1)\frac{q(x_1)}{q(x_0)}\prod_{t=1}^{T-1}q(x_t \mid x_{t+1},x_0)\frac{q(x_{t+1}\mid x_0)}{q(x_t \mid x_0)}$$
$$=q(x_T\mid x_0)\prod_{t=1}^{T-1}q(x_t\mid x_{t+1},x_0)$$

$$p_\theta(x_{0:T})= \prod_{t=1}^{T-1}p_\theta(x_t \mid x_{t+1})p_\theta(x_0 \mid x_1)p_\theta(x_T)$$

则优化目标变为
$$\sum_{t=1}^{T-1}\mathbb{E}_{x_t,x_{t+1}\sim q(x_t,x_{t+1}\mid x_0)}[\frac{p_\theta(x_t \mid x_{t+1})}{q(x_t \mid x_{t+1},x_0)}] + \mathbb{E}_{x_T\sim q(x_T \mid x_0)}[\frac{q(x_0\mid x_1)}{p_\theta(x_T)}] + \mathbb{E}_{x_1\sim q(x_1\mid x_0)}[p_\theta(x_0 \mid x_1)]$$

第一项是为了让神经网络学习方向过程刻意对齐的项,后面两项是多余的项
其中
$$\mathbb{E}_{x_T\sim q(x_T \mid x_0)}[\frac{q(x_0\mid x_1)}{p_\theta(x_T)}]$$
和学习反向过程无关通常直接忽略,其余两项都表示从后向前学习一个状态

###### 分布$q$的定义

##### TODO
完善DDPM 损失函数和具体计算过程
discrete mask llm







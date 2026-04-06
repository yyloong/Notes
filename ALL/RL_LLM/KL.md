### KL 散度及其估计

### Definition
$$KL(p(x)\| q(x)) = \mathbb{E}_{p(x)}\left[\log\frac{p(x)}{q(x)}\right]$$

用于衡量两个分布之间的差异  
可以写成
$$KL(p(x)\|q(x)) = -\mathbb{E}_{p(x)}[\log q(x)]-(-\mathbb{E}_{p(x)}[\log p(x)])=H(p(x),q(x))-H(p(x))$$

对于期望在工程上通常通过蒙特卡洛采样进行估计
$$\mathbb{E}_{p(x)}\left[\log\frac{p(x)}{q(x)}\right]=\sum_{x\sim p} p(x)\log\frac{p(x)}{q(x)}$$

### FKL(forward) 和 RKL(reverse)
在llm RL中假设参考策略为$q$,目标策略为$p_{\theta}$
则 $$FKL = KL(q(x)\|p_{\theta}(x))$$
$$RKL = KL(p_{\theta}(x)\| q(x))$$

二者都可以用来衡量分布之间的差异,但是$FKL$对$q$取期望,所以在优化$FKL$的时候,对于$q$可能采样到的点,$p$都要取尽量拟合$q$
而$RKL$对$p$取期望,只需要在$p$采样概率较高的点取拟合$q$即可
所以$FKL$倾向于更加全面拟合$q$,$RKL$倾向于拟合$q$的一个局部,代表着"模式寻优",RLHF通常采用$RKL$

### 3种RKL估计
令$$r(x) = \log\frac{p(x)}{q(x)}$$
- k1 估计,朴素的蒙特卡洛估计 $$KL(p,q) \approx \sum_{x\sim p} -\log r(x)$$
实际计算不一定是正数,存在高方差问题(正负抵消)

- k2 估计 $$KL(p,q) \approx \sum_{x\sim p}\frac{1}{2}(\log r(x))^2$$
小方差但是有偏

- k3 估计 $$KL(p,q) \approx \sum_{x\sim p} r(x) -1 - \log r(x)$$
无偏且减小了方差(恒大于0),相当于$f(x)=-\log x$的Bregman散度

参考链接
[【新瓶旧酒】简单理解 RL 中的 KL 散度估计器：从数值估计到梯度估计](https://zhuanlan.zhihu.com/p/1978993413425763764)

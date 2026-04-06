# DPO 建模与求解

## RLHF 的标准流程

**1. 训练一个 Reward Model 满足人类偏好**

具体而言，对于数据 $(x, y_w, y_l)$，求解问题：

$$
\max_{(x, y_w, y_l) \sim p_{\text{data}}} \log \left( \sigma \left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right)
$$

假设我们训练好了 $r_\phi$。

**2. 训练目标模型**

$$
\max_{\theta} \mathbb{E}_{x \sim p_{\text{data}}, y \sim \pi_\theta(y|x)} \left[ r_\phi(x,y) \right] - \beta \mathbb{KL} \left( \pi_\theta(y|x) \parallel \pi_{\text{ref}}(y|x) \right)
$$

---

## 数学推导与等价求解

上述目标函数的最大化，等价于求解以下期望的最小化（取负号）：

$$
\mathbb{E}_{x \sim p_{\text{data}}, y \sim \pi_\theta(y|x)} \left[ \log \frac{\pi_\theta(y|x)}{\exp(\frac{1}{\beta} r_\phi(x,y)) \pi_{\text{ref}}(y|x)} \right]
$$

对分母进行归一化，可得到 KL 散度的形式：

$$
\mathbb{E}_{x \sim p_{\text{data}}, y \sim \pi_\theta(y|x)} \left[ \log \frac{\pi_\theta(y|x)}{\frac{1}{Z(x)} \exp(\frac{1}{\beta} r_\phi(x,y)) \pi_{\text{ref}}(y|x)} \right]
$$

由此可见，最优策略 $\pi^*(y|x)$ 为：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \exp \left( \frac{1}{\beta} r_\phi(x,y) \right) \pi_{\text{ref}}(y|x)
$$

---

## DPO 的核心思想

如果可以用策略 $\pi_\theta$ （去逼近 $\pi^*$）来表示 $r_\phi$，则可一步到位，跳过 $r_\phi$ 的单独训练过程。通过上述公式反解 $r_\phi$：

$$
r_\phi(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

**代入原式，得到最终的 DPO 目标函数：**

$$
\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

> 💡 **注**：在代入相减的过程中，$Z(x)$ 作为只与 $x$ 相关的项，恰好被抵消掉。
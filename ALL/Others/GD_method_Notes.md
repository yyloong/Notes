#### Gradient descent
**Iteration**:$x_{t+1}=\Pi_{\mathcal{X}}[x_t-\eta_t\nabla f(x_t)]$
**goal**:
bound $f(x_t)-f(x^*),\|x_t-x^*\|$
**some skills**:
$\|x_t-x^*\|$
$\|x_t-x^*\|^2=\|\Pi_{\mathcal{X}}[x_t-\eta_t\nabla f(x_t)]-x^*\|^2\leq \|x_t-x^*-\eta_t\nabla f(x_t)\|=
\|x_t-x^*\|^2-2\eta_t \langle x_t-x^*,\nabla f(x_t)\rangle+\eta_t^2\|\nabla f(x_t)\|^2$
对$\langle x-x^*,f(x_t) \rangle$依据不同条件进行放缩
可以得到不同的式子
$convex$:
$\langle x-x^*,f(x_t)\rangle\geq f(x_t)-f(x^*)$
可以将$\|x_t-x^*\|$和$f(x_t)-f(x^*)$联系起来
$L-smooth$:
$\langle \nabla f(x)-\nabla f(y),x-y\rangle \geq \frac{1}{L}\|\nabla f(x)-\nabla f(y)\|_*^2$
$\langle \nabla f(x_t),x_t-x^*\rangle\geq \frac{1}{L}\|\nabla f(x_t)\|_*^2$


## Multi-Armed Bandit Problem
#### 1. 개요

Multi-Armed Bandit(MAB) problem이란 강화 학습에서 다루는 분야 중 하나로, 여러 선택지가 주어지고 각 선택지는 특정한 확률 분포를 따르는 보상을 돌려준다고 할 때 어떤 선택지를 고르는 것이 이득인지 판단하는 문제이다. 가장 쉬운 예로는 웹사이트의 광고 자리에 어떤 회사의 광고를 노출시켜야 사람들이 많이 클릭해 수익을 최대한 얻을 수 있을지 판단하는 것이 있다.

각 선택지를 arm, \(i\)번째 arm의 보상의 기댓값을 \(θ_i\), \(i\)번째 arm의 보상이 따르는 확률 분포를 \(P_i\)라고 쓰자. 통상적으로 각 arm은 같은 모양의 확률 분포를 가진다고 가정하므로 \(P_i\)=\(P(θ_i )\)라고 쓸 수 있다. 선택지를 고르는 행동을 draw, \(t\)번째 draw한 arm의 번호를 \(A(t)\), 얻은 보상을 \(r_t\)라고 쓰면 \(r_t\)~\(P(θ_(A(t)))\)가 성립한다.

MAB 알고리즘의 목표는 \(n\)개의 arm이 주어지고 draw를 \(T\)번 시행해야 할 때, 다음과 같이 정의되는 loss function을 최소화 하는 것이다.

$$E[R(T)]=\sum_{t=1}^T{(θ^*-E[r_t])=\sum_{i=1}^n\left[(θ^*-θ_i)E\left[\sum_{t=1}^T{I(A(t)=i)}\right]\right]}$$

\(θ^*\)는 최선의 행동을 했을 때 얻는 보상의 기댓값, \(I\)는 indicator 함수이다. 이 목표를 달성하기 위해선 보상의 기댓값이 높은 arm을 찾아야 하는데, 여기서 exploration-exploitation 트레이드오프가 발생한다. 이때까지 얻은 정보를 바탕으로 가장 좋은 arm을 draw 할 경우 안정적인 보상을 얻겠지만 혹시 있을지도 모르는 더 좋은 arm을 간과하게 된다. 더 좋은 arm을 찾기 위해 여러 arm을 draw 할 경우 당장 loss가 커지게 된다. 이 트레이드오프 관계를 어떻게 해결하느냐가 서로 다른 MAB 알고리즘의 핵심이 된다.

#### 2. 알고리즘

##### 2-1. UCB1
아래와 같은 기준을 통해 arm을 draw하는 MAB 알고리즘을 upper confidence bound(UCB) 계열 알고리즘이라고 통칭한다.
$$i=argmax_{{}_i} \space \mu_i+P_i$$
\(μ_i\)는 이때까지 얻은 정보를 기반으로 계산한 \(i\)번째 arm의 보상의 기댓값에, \(P_i\)는 확률 분포에 대한 불확실성에 관련된 항이다. 즉 전자는 exploitation을, 후자는 exploration을 뜻하고 그 둘을 모두 고려해서 선택하는 것이다.

UCB1은 \(T+1\)번째 시도일 때 \(μ_i=\frac{\sum_{t=1}^Tr_t*I(A(t)=i)}{\sum_{t_1}^TI(A(t)=i)}\) 로, \(P_i\)=\(\sqrt{\frac{2ln(t)}{n_i}}\) 로 계산한다. 즉 \(μ_i\)는 \(T\)번째 시도까지의 \(i\)번째 arm의 보상의 평균값(empirical mean)을 사용하고, \(P_i\)의 분모에 \(i\)번째 arm을 draw한 횟수인 \(n_i\)를 설정해 충분히 exploration 되지 않은 arm에 가중치를 준다. \(P_i\)가 불확실성을 가장 낙관적으로 여겼을 때 해당 arm의 기대치를 나타내기 때문에 upper confidence bound라고 부른다.

대부분의 MAB 알고리즘은 regret bound가 시간에 대해 asymptotically logarithmic (optimal) 한 것만 보이지만, UCB1의 경우 가능한 모든 확률 분포 조건과 모든 시간(asymptotical 하지 않아도)에 대해 logarithmic 한 것이 알려져 있다[1].

##### 2-2. UCB1-tuned
UCB1의 exploration term을 아래와 같이 개량한 버전이다.
$$P_i = \sqrt{\frac{ln(t)}{n_i}}min\left(\frac{1}{4},V_i(n_i)\right)$$
$$where \space\space V_i(n_i)\equiv\left(\frac{1}{n}\sum_{j=1}^{n_i} r_{i,j}^2 \right)-\left(\overline{r_{i,n_i}}\right)^2+\sqrt{\frac{2ln(t)}{n_i}}$$

이는 보상의 분산 값도 exploration의 척도로 사용한 것이다. \(r_{i,j}\)는 \(i\)번째 arm을 draw한 \(n_i\)번 중 \(j\)번째로 draw 했을 때 얻은 보상값이고, \(\left(\overline{r_{i,n_i}}\right)\)는 \(n_i\)번 동안 얻은 보상값의 평균이다. 1/4와 minimum을 취하는 것은 Bernoulli distribution에서 분산의 최댓값이 1/4이기 때문이다. 이 경우 regret bound가 optimal한 것을 보일 수는 없지만 arm의 보상이 Bernoulli distribution를 따를 때 UCB1에 비해 더 좋은 실제 성능을 보인다. 따라서 실제 서비스에 알고리즘을 적용할 경우 UCB1보다 UCB1-tuned를 사용하는 것이 권장된다[2].

##### 2-3. Thompson Sampling
Thompson Sampling은 비교적 최근에 쓰이기 시작한 알고리즘으로, deterministic하게 동작하는 UCB 계열의 알고리즘과 달리 expectation의 posterior distribution을 계산한 뒤 sampling한 값을 기준으로 arm을 선택한다.

웹에 노출되는 광고와 같이 arm의 보상이 binary reward이고 Bernoulli distribution을 따르는 경우를 생각해보자. 광고를 클릭할 확률은 정해져있고 피드백으로는 노출된 광고가 클릭되었는지 여부가 주어지는 경우이다. \(θ_i\)의 prior distribution을 \(Beta(1,1)\)로 두면 Beta distribution은 Bernoulli distribution의 conjugate prior이기 때문에 계산하기 쉽다. 만약 \(i\)번째 arm을 \(n_i\)번 draw 했을 때, 보상이 \(1\)인 경우가 \(k_i\)번 있었고 \(0\)인 경우가 \(n_i-k_i\)번 있었으면 \(θ_i\)의 posterior distribution은 \(Beta(1+k_i,1+n-k_i)\)가 된다. 그 뒤 각각의 arm에 대해 expectation의 posterior distribution에서 값을 sampling 한 뒤, 값이 가장 큰 arm을 draw하면 된다.

Thompson Sampling 또한 optimal 함이 증명되어 있으며 실제 구현에서도 UCB 계열의 알고리즘보다 나은 성능을 보인다[3]. 이 성질은 한번에 여러 개의 arm을 draw 할 때(multiple plays)도 성립한다[4].

##### 2-4. Bayes-UCB
[5]에서 제시된 UCB 계열의 알고리즘으로 upper confidence bound를 계산할 때 베이지안 모델링을 적용한다. 각 arm의 expectation의 posterior distribution을 베이지안 모델을 이용해 계산하고, 시간에 관련된 term에 대해 quantile을 계산해 UCB로 삼는다. Formal하게 나타내면 다음과 같다.
$$draw \space\space i = argmax_{{}_i} Q\left(1-\frac{1}{t}, \lambda_i^{t-1}\right)$$
여기서 \(Q\) 함수는 실수와 probability distribution을 받아 quantile을 반환하는 함수로, \(Q(t,ρ)\)는 \(P_ρ (X\le Q(t,ρ))=t\)를 만족시키는 값이다. \(λ_i^t\)는 \(t\)번째 draw 이후 estimate 된 \(i\)번째 arm의 expectation의 posterior distribution이다. arm의 보상이 Bernoulli distribution를 따르는 경우 Thompson Sampling과 마찬가지 방법으로 posterior distribution을 계산할 수 있다. 즉 Thompson Sampling의 deterministic한 버전이라고 생각할 수 있다.

아래 그림은 알고리즘 별로 보상이 Bernoulli distribution을 따르는, 각각의 expectation이 [0.45, 0.44, 0.40, 0.25, 0.05]인 5개의 arm에 대해 T=5000번 draw 했을 때의 regret을 50번의 시뮬레이션을 통해 측정한 평균값이다.

![](http://cfile3.uf.tistory.com/image/998AEA4F5A78A28C15D681)


그 외에도 UCB 계열의 알고리즘인 KL-UCB 등이 있고, problem에 position bias를 적용하거나 bandit이 stochastic이 아닌, adversarial나 contextual인 경우를 다루는 알고리즘 등 여러가지 variation이 존재한다.

[1] Burtini, G., Loeppky, J., Lawrence, R.: A survey of online experiment design with the stochastic multi-armed bandit. https://arxiv.org/pdf/1510.00757.pdf (2015).
[2] Auer, Peter, Cesa-bianchi, Nicol´o, and Fischer, Paul.: Finite-time Analysis of the Multiarmed Bandit Problem. Machine Learning, 47:235–256 (2002).
[3] Kaufmann, E., Korda, N., andMunos, R.: Thompson sampling: An asymptotically optimal finite-time analysis. In Algorithmic Learning Theory. Springer, 199–213. (2012).
[4] Junpei Komiyama, J. H. and Nakagawa, H.: Optimal regret analysis of Thompson Sampling in stochastic multi-armed bandit problem with multiple plays. In International Conference on Machine Learning. Vol. 37. (2015).
[5] Kaufmann, E., Capp´e, O., and Garivier, A.: On Bayesian upper confidence bounds for bandit problems. In International Conference on Artificial Intelligence and Statistics. 592–600. (2012).

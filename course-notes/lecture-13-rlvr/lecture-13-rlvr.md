- ---
  title: "Lecture 13 — Reinforcement Learning with Verifiable Rewards"
  slug: "lecture-13"
  date: "2026-10-07"
  updated: "2026-10-07"
  status: "draft"
  type: "learning-note"
  course: "CS 395T"
  source: "https://noahgol.github.io/teaching/cs395t-f26/lecture13-v2.pdf"
  summary: "An AI-assisted reconstruction of RLVR from REINFORCE and PPO through RLOO, Dr. GRPO, GRPO, and MaxRL, with explicit derivations of their baselines, biases, normalization effects, and implicit objectives."
  public: true
  tags:
    - RLVR
    - policy gradients
    - REINFORCE
    - PPO
    - RLOO
    - GRPO
    - MaxRL
    - pass@k
  related_research_notes: []
  authorship: "ai-assisted"
  ---
- > **AI-assisted first draft.** This note reconstructs the supplied Lecture 13 PDF and expands its estimators and proofs into step-by-step derivations. It has not yet been manually verified or rewritten by the author. Statements tagged `[DIRECT]` follow the lecture; `[DERIVATION]`, `[EXPLANATION]`, and `[SYNTHESIS]` make the reasoning explicit.
- ## Lecture context
- `[DIRECT]` Source: <https://noahgol.github.io/teaching/cs395t-f26/lecture13-v2.pdf>
- `[DIRECT]` The supplied PDF is ten pages long. Its substantive sections develop reinforcement learning with verifiable rewards (RLVR), review REINFORCE and PPO, compare three group-based estimators, and derive maximum-likelihood reinforcement learning (MaxRL).
- `[SOURCE BOUNDARY]` The lecture assumes an outcome-level binary verifier $R(x,y)\in\{0,1\}$ and no intermediate rewards. Some arguments extend beyond this setting, but this draft does not silently generalize binary-reward identities to arbitrary rewards.
- `[SYNTHESIS]` The central question is not merely how to reduce the variance of policy gradients. It is: **when an implementation changes a baseline, normalization, sampling rule, or denominator, is it still estimating the gradient of the same objective?**
- `[SYNTHESIS]` The lecture's four named algorithms form a useful progression. RLOO supplies an unbiased group baseline; Dr. GRPO introduces a harmless constant scaling; GRPO changes the relative weighting of response lengths and prompt difficulties; MaxRL intentionally changes the objective from success probability to log success probability.
- ## Reading map

  | Part | Central question | Main conclusion |
  | --- | --- | --- |
  | RLVR setup | What exactly is the policy and reward? | A complete response is a trajectory; the verifier rewards only the terminal response. |
  | REINFORCE | How do we differentiate success probability? | The reward-weighted response score is an unbiased policy-gradient estimator. |
  | Values and baselines | What may be subtracted without changing the gradient? | A baseline independent of the sampled continuation has zero expected score contribution. |
  | PPO | How can sampled responses be reused? | Token ratios and clipping create a practical surrogate that is exact only at the sampling policy. |
  | RLOO | Can a group replace a learned value model? | Leave-one-out rewards give an unbiased prompt-level baseline. |
  | Dr. GRPO | What happens if a response enters its own baseline? | The RLOO gradient is scaled by $(G-1)/G$ but keeps its direction. |
  | GRPO | What do its normalizations do? | Length and reward-standard-deviation factors reweight responses and prompts. |
  | MaxRL | How can training emphasize hard prompts? | Optimize $\log p_\theta(x)$, or a finite pass@k-based approximation, instead of $p_\theta(x)$. |
- ## 1. RLVR as a sequence-level policy-gradient problem
- ### 1.1 Generation, trajectories, and rewards
- `[DIRECT]` Draw a training prompt $x\sim\mathcal D$. The language model generates a response $y=(y_1,\ldots,y_T)$ autoregressively:

  $$
  y_t\sim\pi_\theta(\cdot\mid x,y_{1:t-1}),
  \qquad t=1,\ldots,T.
  $$
- `[DIRECT]` Generation stops at an end-of-sequence token or a fixed maximum length $T_{\max}$. The response length $T=T(y)$ is therefore random but bounded.
- `[DERIVATION]` The chain rule gives the probability of the complete response:

  $$
  \pi_\theta(y\mid x)
  =\prod_{t=1}^{T}
  \pi_\theta(y_t\mid x,y_{1:t-1}).
  $$
- `[EXPLANATION]` This is a token-level Markov decision process. The state is the prompt plus the generated prefix, the action is the next token, and the transition appends that token. Once $x$ is fixed, the policy is the only source of randomness.
- `[DIRECT]` A verifier observes the completed response and returns

  $$
  R(x,y)\in\{0,1\}.
  $$
- `[EXPLANATION]` Examples include checking a final numerical answer or running a generated program against tests. The verifier does not need to provide a target reasoning trace.
- `[DIRECT]` Define the success probability for one prompt and the population objective by

  $$
  p_\theta(x)
  :=\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)],
  \qquad
  J(\theta)
  :=\mathbb E_{x\sim\mathcal D}[p_\theta(x)].
  $$
- `[SYNTHESIS]` $p_\theta(x)$ is the probability that one sampled response passes the verifier. $J(\theta)$ averages absolute improvements in this probability across the prompt distribution.
- ### 1.2 The response score decomposes over tokens
- `[DERIVATION · Step 1]` Take logs of the response probability:

  $$
  \log\pi_\theta(y\mid x)
  =\sum_{t=1}^{T}
  \log\pi_\theta(y_t\mid x,y_{1:t-1}).
  $$
- `[DERIVATION · Step 2]` Differentiate:

  $$
  \nabla_\theta\log\pi_\theta(y\mid x)
  =\sum_{t=1}^{T}
  \nabla_\theta\log\pi_\theta(y_t\mid x,y_{1:t-1}).
  $$
- `[EXPLANATION]` The gradient of a complete response's log-probability is the sum of its token-level score vectors. This is why one outcome-level reward can be applied to every token in the response.
- ### 1.3 REINFORCE from the log-derivative identity
- `[DERIVATION · Step 1]` Expand success probability as a sum over all possible completed responses:

  $$
  p_\theta(x)
  =\sum_y R(x,y)\pi_\theta(y\mid x).
  $$
- `[DERIVATION · Step 2]` Differentiate term by term:

  $$
  \nabla_\theta p_\theta(x)
  =\sum_y R(x,y)\nabla_\theta\pi_\theta(y\mid x).
  $$
- `[DERIVATION · Step 3]` Use $\nabla p=p\nabla\log p$:

  $$
  \nabla_\theta p_\theta(x)
  =\sum_y R(x,y)\pi_\theta(y\mid x)
  \nabla_\theta\log\pi_\theta(y\mid x).
  $$
- `[DERIVATION · Step 4]` Recognize an expectation:

  $$
  \nabla_\theta p_\theta(x)
  =\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}
  \left[R(x,y)\nabla_\theta\log\pi_\theta(y\mid x)\right].
  $$
- `[DERIVATION · Step 5]` Average over prompts and substitute the token-score decomposition:

  $$
  \nabla_\theta J(\theta)
  =\mathbb E_{x,y}
  \left[
  \sum_{t=1}^{T}
  R(x,y)
  \nabla_\theta\log\pi_\theta(y_t\mid x,y_{1:t-1})
  \right].
  $$
- `[CONCLUSION]` A sampled prompt-response pair gives an unbiased gradient estimator. A successful response contributes the sum of its token scores; a failed response contributes zero in the uncentered estimator.
- `[LIMIT]` “A failed response contributes zero” describes the raw REINFORCE estimator. Once a baseline is subtracted, a failed response can carry a negative coefficient and therefore affect the update.
- ## 2. Values, advantages, and baselines
- ### 2.1 Prefix values and token advantages
- `[DIRECT]` For any policy $\pi$, define the value of a prefix as the probability of eventual success when the continuation is sampled from $\pi$:

  $$
  V^\pi(x,y_{1:t-1})
  :=\mathbb E_{\text{continuation from }\pi}
  [R(x,y)\mid x,y_{1:t-1}].
  $$
- `[DIRECT]` Define the advantage of choosing token $a$ next by

  $$
  A^\pi(x,y_{1:t-1},a)
  :=
  \mathbb E[R(x,y)\mid x,y_{1:t-1},y_t=a]
  -V^\pi(x,y_{1:t-1}).
  $$
- `[EXPLANATION]` The value asks “how likely is success from here?” The advantage asks “how much does this particular next token change that chance compared with sampling the next token normally?”
- `[CHECK]` Averaging the advantage over $a\sim\pi(\cdot\mid x,y_{1:t-1})$ gives zero by construction.
- ### 2.2 Why a valid baseline leaves the gradient unchanged
- `[DERIVATION · Step 1]` Fix a prefix. The expected next-token score is

  $$
  \begin{aligned}
  &\mathbb E_{y_t\sim\pi_\theta}
  [\nabla_\theta\log\pi_\theta(y_t\mid x,y_{1:t-1})]\\
  &\quad=
  \sum_{a\in\mathcal V}
  \pi_\theta(a\mid x,y_{1:t-1})
  \nabla_\theta\log\pi_\theta(a\mid x,y_{1:t-1})\\
  &\quad=
  \nabla_\theta
  \sum_{a\in\mathcal V}
  \pi_\theta(a\mid x,y_{1:t-1})
  =\nabla_\theta 1=0.
  \end{aligned}
  $$
- `[DERIVATION · Step 2]` If $b(x,y_{1:t-1})$ is fixed once the prefix is known, it can be taken outside the conditional expectation:

  $$
  \mathbb E
  \left[
  b(x,y_{1:t-1})
  \nabla_\theta\log\pi_\theta(y_t\mid x,y_{1:t-1})
  \mid x,y_{1:t-1}
  \right]=0.
  $$
- `[DERIVATION · Step 3]` Subtracting the baseline therefore gives the same expected gradient:

  $$
  \nabla_\theta J(\theta)
  =\mathbb E
  \left[
  \sum_{t=1}^{T}
  \bigl(R(x,y)-b(x,y_{1:t-1})\bigr)
  \nabla_\theta\log\pi_\theta(y_t\mid x,y_{1:t-1})
  \right].
  $$
- `[DIRECT]` With $b=V^{\pi_\theta}$, the Monte Carlo weight $R-V^{\pi_\theta}$ has conditional expectation equal to the token advantage.
- `[LIMIT]` The independence condition matters. Fitting a baseline on the same sampled continuation can make the baseline depend on the action noise whose score it multiplies; the zero-mean argument does not automatically cover that case.
- ### 2.3 Which scalar baseline minimizes variance?
- `[DIRECT]` At a fixed prefix, the trace of the covariance of one token's centered estimator depends on the scalar $b$ through

  $$
  \mathbb E
  \left[
  (R-b)^2
  \left\|\nabla_\theta\log\pi_\theta(y_t\mid x,y_{1:t-1})\right\|_2^2
  \right].
  $$
- `[DERIVATION · Step 1]` Let

  $$
  S=
  \left\|\nabla_\theta\log\pi_\theta(y_t\mid x,y_{1:t-1})\right\|_2^2.
  $$
- `[DERIVATION · Step 1 continued]` Expanding the quadratic gives $\mathbb E[R^2S]-2b\mathbb E[RS]+b^2\mathbb E[S]$.
- `[DERIVATION · Step 2]` Differentiate with respect to $b$ and set the derivative to zero:

  $$
  -2\mathbb E[RS]+2b\mathbb E[S]=0.
  $$
- `[DERIVATION · Step 3]` The minimizing baseline is therefore

  $$
  b^*
  =
  \frac{
  \mathbb E\left[R
  \left\|\nabla_\theta\log\pi_\theta(y_t\mid x,y_{1:t-1})\right\|_2^2\right]
  }{
  \mathbb E\left[
  \left\|\nabla_\theta\log\pi_\theta(y_t\mid x,y_{1:t-1})\right\|_2^2\right]
  }.
  $$
- `[EXPLANATION]` The value function is exactly optimal only in the special case where the score norm is constant across possible next tokens. In general, it is a useful approximation, not a universal variance minimizer.
- `[LIMIT]` This calculation concerns one token term. The variance of the sum over tokens also includes cross-token covariance.
- ### 2.4 PPO and sample reuse
- `[DIRECT]` REINFORCE is on-policy. PPO instead samples responses from a frozen policy $\pi_{\mathrm{old}}$ and takes several updates before drawing a new batch.
- `[DIRECT]` For token $t$, define the probability ratio

  $$
  \rho_t(\theta;x,y)
  :=
  \frac{
  \pi_\theta(y_t\mid x,y_{1:t-1})
  }{
  \pi_{\mathrm{old}}(y_t\mid x,y_{1:t-1})
  }.
  $$
- `[EXPLANATION]` Exact trajectory-level importance sampling would multiply all token ratios. That product can become extremely variable for long responses, so PPO uses a local token-level surrogate instead.
- `[DIRECT]` With clipping width $\epsilon$, define

  $$
  \ell_\epsilon(\rho,A)
  =\min\{\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A\}.
  $$
- `[DIRECT]` A response-level reward with a frozen value baseline gives the surrogate

  $$
  L_{\mathrm{PPO}}(\theta)
  =
  \mathbb E_{x,y\sim\pi_{\mathrm{old}}}
  \left[
  \sum_{t=1}^{T}
  \ell_\epsilon
  \left(
  \rho_t(\theta;x,y),
  R(x,y)-V_\phi(x,y_{1:t-1})
  \right)
  \right].
  $$
- `[DERIVATION]` At $\theta=\theta_{\mathrm{old}}$, every ratio equals one, clipping is inactive, and

  $$
  \nabla_\theta\rho_t(\theta;x,y)
  \big|_{\theta=\theta_{\mathrm{old}}}
  =
  \nabla_\theta\log\pi_\theta(y_t\mid x,y_{1:t-1})
  \big|_{\theta=\theta_{\mathrm{old}}}.
  $$
- `[CONCLUSION]` The surrogate gradient matches the baseline REINFORCE gradient at the sampling policy. After several updates, the data still come from $\pi_{\mathrm{old}}$ and clipping modifies the gradient, so the surrogate is generally biased for the current policy's exact expected-reward gradient.
- ## 3. Group-based estimators
- ### 3.1 Replacing a value model with same-prompt comparisons
- `[DIRECT]` Fix one prompt $x$ and sample $G\ge2$ independent responses

  $$
  y^{(1)},\ldots,y^{(G)}
  \overset{\mathrm{iid}}{\sim}
  \pi_\theta(\cdot\mid x).
  $$
- `[DIRECT]` Write

  $$
  R_i:=R(x,y^{(i)}),
  \qquad
  \bar R:=\frac1G\sum_{j=1}^{G}R_j.
  $$
- `[SYNTHESIS]` The group estimates how difficult the prompt is for the current policy. It avoids training a separate prefix-level value model, but it supplies only a prompt-level comparison: the same scalar multiplies every token of one response.
- ### 3.2 RLOO: leave the current response out
- `[DIRECT]` REINFORCE leave-one-out uses the mean reward of the other $G-1$ responses as response $i$'s baseline:

  $$
  \widehat g_{\mathrm{RLOO}}(x)
  =
  \frac1G\sum_{i=1}^{G}
  \left(
  R_i-
  \frac1{G-1}\sum_{j\ne i}R_j
  \right)
  \nabla_\theta\log\pi_\theta(y^{(i)}\mid x).
  $$
- `[DERIVATION · Step 1]` For $j\ne i$, $y^{(i)}$ and $y^{(j)}$ are independent conditional on $x$.
- `[DERIVATION · Step 2]` The response score has conditional mean zero:

  $$
  \mathbb E
  [\nabla_\theta\log\pi_\theta(y^{(i)}\mid x)\mid x]=0.
  $$
- `[DERIVATION · Step 3]` Therefore every cross-response baseline term has zero expectation:

  $$
  \mathbb E
  \left[
  R_j\nabla_\theta\log\pi_\theta(y^{(i)}\mid x)
  \mid x
  \right]
  =
  \mathbb E[R_j\mid x]\cdot0=0.
  $$
- `[DERIVATION · Step 4]` The remaining own-reward terms are ordinary REINFORCE terms, so

  $$
  \mathbb E[\widehat g_{\mathrm{RLOO}}(x)\mid x]
  =\nabla_\theta p_\theta(x).
  $$
- `[CONCLUSION]` RLOO is unbiased for the per-prompt expected-reward gradient.
- `[EXPLANATION]` Leaving $R_i$ out is essential. $R_i$ and the score of $y^{(i)}$ are generated by the same response and are generally dependent.
- `[DIRECT]` Because the leave-one-out baseline averages $G-1$ independent Bernoulli rewards, its conditional variance is

  $$
  \frac{p_\theta(x)(1-p_\theta(x))}{G-1}.
  $$
- `[LIMIT]` The $G$ response terms are not independent because every response appears in the baselines of the others. The estimator's variance is therefore not obtained by simply dividing one term's variance by $G$.
- ### 3.3 Dr. GRPO: include the response in the group mean
- `[DIRECT]` Dr. GRPO uses

  $$
  \widehat g_{\mathrm{Dr}}(x)
  =
  \frac1G\sum_{i=1}^{G}
  (R_i-\bar R)
  \nabla_\theta\log\pi_\theta(y^{(i)}\mid x).
  $$
- `[DERIVATION · Step 1]` Split the group mean into the current reward and the other rewards:

  $$
  \bar R
  =\frac1G R_i+
  \frac{G-1}{G}
  \left(\frac1{G-1}\sum_{j\ne i}R_j\right).
  $$
- `[DERIVATION · Step 2]` Subtract from $R_i$:

  $$
  R_i-\bar R
  =
  \frac{G-1}{G}
  \left(
  R_i-\frac1{G-1}\sum_{j\ne i}R_j
  \right).
  $$
- `[DERIVATION · Step 3]` Therefore, sample by sample,

  $$
  \widehat g_{\mathrm{Dr}}(x)
  =\frac{G-1}{G}
  \widehat g_{\mathrm{RLOO}}(x),
  $$
  and hence

  $$
  \mathbb E[\widehat g_{\mathrm{Dr}}(x)\mid x]
  =\frac{G-1}{G}\nabla_\theta p_\theta(x).
  $$
- `[CONCLUSION]` Including the response in its own baseline introduces a constant multiplicative bias, but no change of direction. Multiplying by $G/(G-1)$ removes it; with fixed $G$, a learning-rate change can absorb it.
- `[DIRECT]` In practical off-policy training, groups are sampled from $\pi_{\mathrm{old}}$, the coefficients $R_i-\bar R$ are frozen, and PPO-style token clipping is used. At $\theta_{\mathrm{old}}$ the surrogate differentiates to the on-policy Dr. GRPO estimator; later steps inherit off-policy and clipping bias.
- ### 3.4 GRPO: standardization, length normalization, and KL
- `[DIRECT]` In the binary-reward setting, the group reward standard deviation is

  $$
  \sqrt{\frac1G\sum_{j=1}^{G}(R_j-\bar R)^2}
  =\sqrt{\bar R(1-\bar R)}.
  $$
- `[DIRECT]` When the group contains both successes and failures, GRPO assigns response $i$ the standardized group advantage

  $$
  \frac{R_i-\bar R}{\sqrt{\bar R(1-\bar R)}}.
  $$
- `[DIRECT]` When all group rewards agree, the standard deviation is zero and the reward advantages are defined to be zero.
- `[DIRECT]` The original GRPO objective also averages over the response length $T(y^{(i)})$ and penalizes divergence from a fixed reference policy $\pi_{\mathrm{ref}}$.
- `[EXPLANATION]` $\pi_{\mathrm{old}}$ and $\pi_{\mathrm{ref}}$ have different roles. The former generated the current batch; the latter anchors the updated policy.
- #### Why the sampled KL term represents a forward KL
- `[DIRECT]` For a next token sampled on-policy, let

  $$
  u(a)=
  \frac{\pi_{\mathrm{ref}}(a\mid s)}
  {\pi_\theta(a\mid s)},
  $$
- `[EXPLANATION]` Here $s=(x,y_{1:t-1})$ is a fixed prefix. GRPO uses the nonnegative expression $u-\log u-1$.
- `[DERIVATION]` Taking expectation under $a\sim\pi_\theta(\cdot\mid s)$ gives

  $$
  \begin{aligned}
  &\sum_a\pi_\theta(a\mid s)
  \left[
  \frac{\pi_{\mathrm{ref}}(a\mid s)}{\pi_\theta(a\mid s)}
  -1+
  \log\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{ref}}(a\mid s)}
  \right]\\
  &\quad=
  1-1+
  \sum_a\pi_\theta(a\mid s)
  \log\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{ref}}(a\mid s)}\\
  &\quad=
  D_{\mathrm{KL}}
  \left(
  \pi_\theta(\cdot\mid s)
  \|\pi_{\mathrm{ref}}(\cdot\mid s)
  \right).
  \end{aligned}
  $$
- `[LIMIT]` This identity assumes the next token is sampled from $\pi_\theta$. When a stored batch comes from $\pi_{\mathrm{old}}$, the unweighted penalty is a surrogate rather than an exact on-policy KL expectation.
- `[DIRECT]` Even when every reward in a group agrees and the reward update vanishes, the KL penalty can still generate an update.
- ### 3.5 What GRPO's two normalizations change
- `[DIRECT]` Ignoring KL and evaluating at the sampling policy, the token coefficient is proportional to

  $$
  \frac{R_i-\bar R}
  {G\,T(y^{(i)})\sqrt{\bar R(1-\bar R)}}
  \quad\text{for GRPO},
  $$
  versus

  $$
  \frac{R_i-\bar R}{G}
  \quad\text{for Dr. GRPO}.
  $$
- #### Length normalization
- `[DERIVATION]` The log-probability of a complete response is a **sum** of token scores. Dividing by $T(y^{(i)})$ replaces that sum with an average token score.
- `[CONSEQUENCE]` Among successful responses, each token in a short response receives a larger positive coefficient than each token in a long response.
- `[CONSEQUENCE]` Among failed responses, each token in a long response receives a smaller negative coefficient. This can weaken the penalty on long incorrect responses.
- `[SYNTHESIS]` The length factor is not a neutral scale shared by the batch. Because it varies across responses, it changes their relative influence.
- #### Reward-standard-deviation normalization
- `[DIRECT]` For large groups, $\bar R\approx p_\theta(x)$, so the prompt-level multiplier behaves like

  $$
  \frac1{\sqrt{p_\theta(x)(1-p_\theta(x))}}.
  $$
- `[CONSEQUENCE]` This multiplier is large for very easy and very hard prompts and smaller for prompts near $p_\theta(x)=1/2$. GRPO therefore changes how prompt difficulties are weighted relative to the original objective $\mathbb E_x[p_\theta(x)]$.
- `[LIMIT]` A finite group with all successes or all failures still produces zero centered reward update, even though the limiting normalization factor would be large.
- #### Four-response numerical check
- `[EXAMPLE]` Let $G=4$ and rewards be $(1,0,0,0)$. Then

  $$
  \bar R=\frac14,
  \qquad
  \sqrt{\bar R(1-\bar R)}
  =\frac{\sqrt3}{4}.
  $$
- `[DERIVATION]` Dr. GRPO's centered coefficients before the common $1/G$ factor are

  $$
  \frac34,-\frac14,-\frac14,-\frac14.
  $$
- `[DERIVATION]` GRPO's standardized advantages are

  $$
  \sqrt3,
  -\frac1{\sqrt3},
  -\frac1{\sqrt3},
  -\frac1{\sqrt3}.
  $$
- `[CHECK]` Both sets sum to zero, but GRPO has rescaled the whole prompt according to its observed success rate. Response-length normalization can then further change the ratios between individual response contributions.
- ## 4. Maximum-likelihood reinforcement learning
- ### 4.1 Change the objective, not merely the estimator
- `[DIRECT]` MaxRL replaces the success-probability objective with

  $$
  J_{\mathrm{ML}}(\theta)
  :=\mathbb E_{x\sim\mathcal D}[\log p_\theta(x)],
  $$
- `[ASSUMPTION]` This section assumes $p_\theta(x)>0$ for the prompts under consideration.
- `[EXPLANATION]` If the verifier extracts and checks a final answer, $p_\theta(x)$ is the marginal likelihood of producing an accepted answer, summed over every reasoning trace that leads to acceptance.
- `[DERIVATION]` The two prompt-level gradients are

  $$
  \nabla_\theta p_\theta(x)
  \quad\text{and}\quad
  \nabla_\theta\log p_\theta(x)
  =\frac1{p_\theta(x)}\nabla_\theta p_\theta(x).
  $$
- `[CONCLUSION]` Expected reward values the same absolute probability improvement equally across prompts. Log success probability values relative improvement and therefore upweights prompts with small $p_\theta(x)$.
- `[EXAMPLE]` Increasing success probability from $0.01$ to $0.02$ and from $0.50$ to $0.51$ both add $0.01$ to expected reward. The corresponding log improvements are $\log 2$ and $\log(1.02)$, so MaxRL strongly prefers the first improvement.
- ### 4.2 pass@k and a finite-order approximation
- `[DIRECT]` For $k$ independent responses, the probability that at least one succeeds is

  $$
  \operatorname{pass@}k_\theta(x)
  =1-(1-p_\theta(x))^k.
  $$
- `[DERIVATION · Step 1]` Write

  $$
  \log p_\theta(x)
  =\log\bigl(1-(1-p_\theta(x))\bigr).
  $$
- `[DERIVATION · Step 2]` Apply $\log(1-z)=-\sum_{k=1}^{\infty}z^k/k$ with $z=1-p_\theta(x)$:

  $$
  \log p_\theta(x)
  =-
  \sum_{k=1}^{\infty}
  \frac{(1-p_\theta(x))^k}{k}.
  $$
- `[DERIVATION · Step 3]` Since $(1-p_\theta(x))^k=1-\operatorname{pass@}k_\theta(x)$,

  $$
  \log p_\theta(x)
  =-
  \sum_{k=1}^{\infty}
  \frac{1-\operatorname{pass@}k_\theta(x)}{k}.
  $$
- `[DIRECT]` Truncate after order $m$:

  $$
  F_m(\theta;x)
  :=-
  \sum_{k=1}^{m}
  \frac{(1-p_\theta(x))^k}{k}.
  $$
- `[CHECK]` At $m=1$, $F_1(\theta;x)=p_\theta(x)-1$, so its gradient is exactly the expected-reward gradient.
- `[DIRECT]` As $m\to\infty$, $F_m(\theta;x)\to\log p_\theta(x)$.
- ### 4.3 How the truncation reweights difficulty
- `[DERIVATION · Step 1]` Differentiate each term:

  $$
  \nabla_\theta F_m(\theta;x)
  =\sum_{k=1}^{m}
  (1-p_\theta(x))^{k-1}
  \nabla_\theta p_\theta(x).
  $$
- `[DERIVATION · Step 2]` Sum the finite geometric series:

  $$
  \nabla_\theta F_m(\theta;x)
  =
  \frac{1-(1-p_\theta(x))^m}{p_\theta(x)}
  \nabla_\theta p_\theta(x).
  $$
- `[EXPLANATION]` The multiplier is at most $\min\{m,1/p_\theta(x)\}$. For extremely hard prompts with $p_\theta(x)\ll1/m$, it is approximately $m$. As $m$ grows, it approaches the maximum-likelihood weight $1/p_\theta(x)$.
- `[SYNTHESIS]` The finite order prevents the weight on an extremely hard prompt from becoming arbitrarily large. Sampling budget therefore acts as an implicit cap on difficulty reweighting.
- ### 4.4 The conditional maximum-likelihood gradient
- `[DIRECT]` For a fixed prompt with $p_\theta(x)>0$,

  $$
  \nabla_\theta\log p_\theta(x)
  =
  \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}
  \left[
  \nabla_\theta\log\pi_\theta(y\mid x)
  \mid R(x,y)=1
  \right].
  $$
- `[DERIVATION · Step 1]` From REINFORCE and $\nabla\log p=\nabla p/p$,

  $$
  \nabla_\theta\log p_\theta(x)
  =
  \frac1{p_\theta(x)}
  \sum_{y:R(x,y)=1}
  \pi_\theta(y\mid x)
  \nabla_\theta\log\pi_\theta(y\mid x).
  $$
- `[DERIVATION · Step 2]` Conditional on success, an accepted response has probability

  $$
  \Pr_\theta(y\mid x,R=1)
  =\frac{\pi_\theta(y\mid x)}{p_\theta(x)}.
  $$
- `[CONCLUSION]` The sum is exactly the expected response score under the policy conditioned on verifier success.
- `[LIMIT]` Sampling until one success would give an unbiased estimator, but it needs $1/p_\theta(x)$ trials on average. This is prohibitive on hard prompts and motivates fixed-budget estimators.
- ### 4.5 A fixed group estimates an exact truncated objective
- `[DIRECT]` Draw $G$ responses and let the number of successes be $G\bar R=\sum_iR_i$. Define

  $$
  \widehat g_G(x)
  =
  \begin{cases}
  \displaystyle
  \frac1{G\bar R}
  \sum_{i=1}^{G}
  R_i\nabla_\theta\log\pi_\theta(y^{(i)}\mid x),
  &\bar R>0,\\[1.1em]
  0,&\bar R=0.
  \end{cases}
  $$
- `[EXPLANATION]` When the group contains successes, this is simply the average score of its successful responses.
- `[DERIVATION · Step 1]` Condition on a particular success-indicator vector with at least one success. Each successful response is distributed as $\pi_\theta(\cdot\mid x)$ conditioned on success.
- `[DERIVATION · Step 2]` By the conditional-gradient identity, the conditional expectation of every successful score is $\nabla_\theta\log p_\theta(x)$. Averaging the successful scores leaves the same expectation.
- `[DERIVATION · Step 3]` The probability that the group contains at least one success is

  $$
  1-(1-p_\theta(x))^G.
  $$
- `[DERIVATION · Step 4]` Therefore

  $$
  \begin{aligned}
  \mathbb E[\widehat g_G(x)\mid x]
  &=\bigl(1-(1-p_\theta(x))^G\bigr)
  \nabla_\theta\log p_\theta(x)\\
  &=
  \frac{1-(1-p_\theta(x))^G}{p_\theta(x)}
  \nabla_\theta p_\theta(x)\\
  &=\nabla_\theta F_G(\theta;x).
  \end{aligned}
  $$
- `[CONCLUSION]` The estimator is biased for the infinite-order log-success gradient, but exactly unbiased for the order-$G$ truncated MaxRL objective.
- `[LIMIT]` If $p_\theta(x)=0$ for every possible parameter value, the log-likelihood and conditioning-on-success statements are undefined. The finite-order gradient and the estimator are still zero.
- ### 4.6 The practical centered MaxRL update
- `[DIRECT]` A zero-mean control variate is the average score of all responses:

  $$
  \mathbb E
  \left[
  \frac1G\sum_{i=1}^{G}
  \nabla_\theta\log\pi_\theta(y^{(i)}\mid x)
  \mid x
  \right]=0.
  $$
- `[DIRECT]` Practical MaxRL performs the centered update only when the group has at least one success:

  $$
  \widehat g_{\mathrm{practical}}(x)
  =
  \begin{cases}
  \displaystyle
  \frac1G\sum_{i=1}^{G}
  \frac{R_i-\bar R}{\bar R}
  \nabla_\theta\log\pi_\theta(y^{(i)}\mid x),
  &\bar R>0,\\[1.1em]
  0,&\bar R=0.
  \end{cases}
  $$
- `[COMPARISON]` Dr. GRPO uses $R_i-\bar R$. Practical MaxRL divides this centered reward by $\bar R$, amplifying groups with a low observed success rate.
- `[COMPARISON]` GRPO divides by $\sqrt{\bar R(1-\bar R)}$ and by response length. Practical MaxRL divides by $\bar R$ and does not normalize by response length.
- `[DIRECT]` Because practical MaxRL skips the all-failure group instead of applying the control variate there, its expectation changes:

  $$
  \mathbb E[\widehat g_{\mathrm{practical}}(x)\mid x]
  =\nabla_\theta F_{G-1}(\theta;x),
  \qquad G\ge2.
  $$
- `[SYNTHESIS]` The uncentered successful-score estimator targets order $G$. The commonly implemented centered-and-skipped update targets order $G-1$. A seemingly small implementation decision changes the exact objective.
- #### Four-response coefficient check
- `[EXAMPLE]` Again take $(R_1,R_2,R_3,R_4)=(1,0,0,0)$, so $\bar R=1/4$.
- `[DERIVATION]` Practical MaxRL's per-response coefficients before the common $1/G$ factor are

  $$
  \frac{R_i-\bar R}{\bar R}
  =3,-1,-1,-1.
  $$
- `[CHECK]` These are exactly four times the Dr. GRPO coefficients $(3/4,-1/4,-1/4,-1/4)$. The factor $1/\bar R=4$ is prompt-dependent, so unlike the constant Dr.-to-RLOO scaling, it cannot be absorbed globally into one learning rate.
- ## 5. Algorithm comparison

  | Method | Response coefficient, ignoring shared factors | Target at the on-policy point | Main source of changed weighting |
  | --- | --- | --- | --- |
  | REINFORCE | $R_i$ | $\nabla p_\theta(x)$ | No baseline; high variance |
  | RLOO | $R_i-\frac1{G-1}\sum_{j\ne i}R_j$ | $\nabla p_\theta(x)$ | Unbiased leave-one-out baseline |
  | Dr. GRPO | $R_i-\bar R$ | $\frac{G-1}{G}\nabla p_\theta(x)$ | Only a constant scale for fixed $G$ |
  | GRPO | $\frac{R_i-\bar R}{T(y^{(i)})\sqrt{\bar R(1-\bar R)}}$ | Reweighted surrogate | Response length and observed prompt difficulty |
  | MaxRL, successful-score form | Average scores of successes | $\nabla F_G(\theta;x)$ | Finite-order approximation to $\nabla\log p_\theta(x)$ |
  | Practical MaxRL | $(R_i-\bar R)/\bar R$, skip all-failure groups | $\nabla F_{G-1}(\theta;x)$ | Inverse group success rate and skip rule |

- `[SYNTHESIS]` “Biased” is incomplete without naming the target. Dr. GRPO is biased for the unscaled expected-reward gradient but keeps its direction. The fixed-budget MaxRL estimator is biased for $\nabla\log p$ but unbiased for a precise truncated objective. PPO and clipped group surrogates are exact at the policy that generated the data but not after multiple reuse steps.
- ## 6. Interactive algorithm lab
- `[EXPLANATION]` The simulator below applies several estimators to the same four sampled responses. Toggle verifier outcomes and switch algorithms to see which changes are a baseline, a global scale, a length-dependent weight, or a prompt-difficulty weight.

<!-- component:rlvr-algorithm-lab -->

- ## 7. Claims to verify manually
- [ ] Re-derive the response-level REINFORCE gradient without looking at the note.
- [ ] State exactly what independence condition a baseline needs.
- [ ] Verify that the optimal scalar baseline is score-norm weighted and need not equal the value function.
- [ ] Prove that the RLOO cross-response terms have zero expectation.
- [ ] Recover the factor $(G-1)/G$ relating Dr. GRPO to RLOO.
- [ ] Explain separately how GRPO's $1/T(y)$ and $1/\sqrt{\bar R(1-\bar R)}$ factors change weighting.
- [ ] Derive the sampled GRPO penalty's on-policy expectation as $D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$.
- [ ] Derive $\log p$ as an infinite weighted sum of pass@k terms.
- [ ] Prove the conditional maximum-likelihood gradient identity.
- [ ] Explain why $G$ samples yield the order-$G$ MaxRL objective, while the practical centered update yields order $G-1$.
- ## 8. Questions to carry into study
- `[QUESTION]` In a real language model, how large are the covariance terms between token-score contributions, and when does a prefix-level value model outperform a group baseline at equal compute?
- `[QUESTION]` How much of GRPO's empirical behavior comes from its intended comparison-based baseline versus its length and standard-deviation normalizations?
- `[QUESTION]` What happens when reward is graded rather than binary? Which derivations survive, and which identities rely specifically on Bernoulli rewards?
- `[QUESTION]` MaxRL emphasizes relative improvement on difficult prompts, but how should prompts with effectively zero success probability be handled without wasting most of the sampling budget?
- `[QUESTION]` How do KL regularization, off-policy sample reuse, and MaxRL-style difficulty weighting interact when combined in one practical training system?
- `[QUESTION]` Is pass@k the right deployment-aligned target when inference uses a verifier to select among multiple responses, or should training model the selection rule explicitly?
- ## 9. What I can now explain
- `[RECONSTRUCTION]` RLVR treats the full generated response as a trajectory and a verifier result as a terminal reward. REINFORCE differentiates the probability of success using reward-weighted response scores.
- `[RECONSTRUCTION]` A baseline is safe when its score contribution has conditional expectation zero. RLOO obtains such a baseline from independent responses to the same prompt; Dr. GRPO includes the current response and therefore introduces only a known constant scaling.
- `[RECONSTRUCTION]` GRPO is not just RLOO with lower variance. Its length and standard-deviation normalizations change the relative influence of responses and prompts, while its reference-policy term adds a separate regularization force.
- `[RECONSTRUCTION]` MaxRL deliberately optimizes log success probability, which emphasizes relative improvement on hard prompts. A finite response budget corresponds to an exact truncated pass@k objective rather than an informal approximation with no target.
- ## References recorded from the supplied lecture
- `[DIRECT]` Ronald J. Williams, “Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning,” *Machine Learning*, 1992. <https://doi.org/10.1007/BF00992696>
- `[DIRECT]` John Schulman et al., “Proximal Policy Optimization Algorithms,” 2017. <https://arxiv.org/abs/1707.06347>
- `[DIRECT]` Arash Ahmadian et al., “Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs,” 2024. <https://arxiv.org/abs/2402.14740>
- `[DIRECT]` Zichen Liu et al., “Understanding R1-Zero-Like Training: A Critical Perspective,” COLM 2025. <https://arxiv.org/abs/2503.20783>
- `[DIRECT]` Zhihong Shao et al., “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models,” 2024. <https://arxiv.org/abs/2402.03300>
- `[DIRECT]` Fahim Tajwar et al., “Maximum Likelihood Reinforcement Learning,” 2026. <https://arxiv.org/abs/2602.02710>

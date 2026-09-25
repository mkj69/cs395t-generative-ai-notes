- ---
  title: "Lecture 05 — Mixture-of-Experts and Scaling Laws"
  slug: "lecture-05"
  date: "2026-09-25"
  updated: "2026-09-25"
  status: "draft"
  type: "learning-note"
  course: "CS 395T"
  source: "https://noahgol.github.io/teaching/cs395t-f26/lecture5.pdf"
  summary: "A first reconstruction of sparse expert routing, load balancing, empirical power laws, compute-optimal parameter-data allocation, and a linear-regression scaling-law example."
  public: true
  tags:
    - mixture of experts
    - routing
    - load balancing
    - scaling laws
    - compute optimality
  related_research_notes: []
  authorship: "ai-assisted"
  ---
- > **AI-assisted first draft.** This note follows the mathematical route of Lecture 05 and makes the main router gradients and scaling-law derivations explicit. It is a study scaffold, not the author's finished interpretation.
- ## Lecture context
- `[DIRECT]` Source: <https://noahgol.github.io/teaching/cs395t-f26/lecture5.pdf>
- `[SYNTHESIS]` The lecture studies two ways to scale a model. Mixture-of-experts increases parameter capacity without activating every parameter on every token. Scaling laws describe how parameters and data should grow together under a compute budget.
- ## Reading map

  | Part | Mechanism | Central tradeoff |
  | --- | --- | --- |
  | MoE routing | top-$K$ expert selection | total capacity versus active compute |
  | Load balancing | auxiliary router loss | specialization versus collapse |
  | Empirical scaling | power-law loss model | parameters versus data |
  | Compute allocation | $C\approx6PD$ | which model and dataset size minimize loss? |
  | Linear-regression example | covariance spectrum | where do scaling exponents come from? |
- ## 1. Mixture-of-experts layers
- ### 1.1 Replacing the dense MLP
- `[DIRECT]` After attention and normalization, let the token representation entering the MLP be $\widetilde u_t^{(r)}\in\mathbb{R}^d$. An MoE layer has $N$ expert MLPs

  $$
  \operatorname{MLP}_{r,1},\ldots,\operatorname{MLP}_{r,N}:\mathbb{R}^d\to\mathbb{R}^d.
  $$
- `[DIRECT]` The router logits and probabilities are

  $$
  z_t^{(r)}=W_R^{(r)}\widetilde u_t^{(r)},
  \qquad
  p_{i,t}^{(r)}=
  \frac{e^{z_{i,t}^{(r)}}}{\sum_{j=1}^{N}e^{z_{j,t}^{(r)}}}.
  $$
- `[DIRECT]` If $S_t^{(r)}$ contains the indices of the top $K$ probabilities, the sparse gate is

  $$
  g_{i,t}^{(r)}=p_{i,t}^{(r)}\mathbf 1\{i\in S_t^{(r)}\}.
  $$
- `[DIRECT]` The residual update is

  $$
  x_t^{(r)}
  =u_t^{(r)}+
  \sum_{i\in S_t^{(r)}}g_{i,t}^{(r)}
  \operatorname{MLP}_{r,i}(\widetilde u_t^{(r)}).
  $$
- `[CHECK]` In the lecture's base convention, selected probabilities are not renormalized after top-$K$ selection. Some architectures divide by their selected sum instead.
- ### 1.2 Capacity versus active computation
- `[DIRECT]` For SwiGLU experts of width $d_{\mathrm{ff}}$, expert weights contribute roughly

  $$
  3Ndd_{\mathrm{ff}}
  $$

  parameters, while one token activates only

  $$
  3Kdd_{\mathrm{ff}}
  $$

  of them in expert matrix multiplications.
- `[SYNTHESIS]` $K/N$ is the active expert fraction. MoE decouples stored capacity from per-token expert compute, although routing, communication, and memory traffic remain real costs.
- `[DIRECT]` Shared experts, when present, process every token. If there are $N_s$ shared experts, each token evaluates $K+N_s$ experts even though the router chooses only among the $N$ routed experts.
- ## 2. Load balancing
- ### 2.1 Hard assignments and soft probability
- `[DIRECT]` For top-1 routing over a batch of $B$ tokens, define

  $$
  f_i=\frac1B\sum_{t=1}^{B}\mathbf 1\{a_t=i\},
  \qquad
  P_i=\frac1B\sum_{t=1}^{B}p_{i,t},
  $$

  where $a_t=\arg\max_i p_{i,t}$.
- `[DIRECT]` The Switch auxiliary loss is

  $$
  \mathcal L_{\mathrm{bal}}
  =\alpha N\sum_{i=1}^{N}f_iP_i.
  $$
- `[SYNTHESIS]` $f_i$ measures actual discrete traffic; $P_i$ measures the router's soft preference. They need not agree: probabilities $0.51$ and $0.99$ create the same top-1 decision.
- ### 2.2 Uniform load when hard and soft loads agree
- `[DIRECT]` If $P_i=f_i$, then

  $$
  \mathcal L_{\mathrm{bal}}
  =\alpha N\sum_i f_i^2.
  $$
- `[DERIVATION]` Since $\sum_i f_i=1$,

  $$
  \sum_{i=1}^{N}\left(f_i-\frac1N\right)^2
  =\sum_i f_i^2-\frac1N.
  $$
- `[DERIVATION]` Therefore

  $$
  \mathcal L_{\mathrm{bal}}
  =\alpha+alpha N
  \sum_{i=1}^{N}\left(f_i-\frac1N\right)^2
  \ge\alpha.
  $$
- `[CONCLUSION]` Equality holds exactly at $f_i=P_i=1/N$. This proof does **not** say the unconstrained loss over arbitrary independent $(f,P)$ has the same minimum; it uses $P=f$.
- ### 2.3 Router-logit gradient
- `[DIRECT]` Hard assignments are treated as constants during backpropagation. For one token,

  $$
  \frac{\partial\mathcal L_{\mathrm{bal}}}{\partial p_{i,t}}
  =\frac{\alpha N}{B}f_i.
  $$
- `[DERIVATION]` Softmax has Jacobian

  $$
  \frac{\partial p_{i,t}}{\partial z_{j,t}}
  =p_{i,t}(\mathbf 1\{i=j\}-p_{j,t}).
  $$
- `[DIRECT]` Define the probability-weighted average load

  $$
  \mu_t=\sum_{i=1}^{N}f_ip_{i,t}.
  $$
- `[DERIVATION]` Applying the chain rule and collecting the $j$ terms gives

  $$
  \frac{\partial\mathcal L_{\mathrm{bal}}}{\partial z_{j,t}}
  =\frac{\alpha N}{B}p_{j,t}(f_j-\mu_t).
  $$
- `[SYNTHESIS]` Gradient descent lowers the logit of an expert whose load is above the token's soft average and raises the logit of one below that average.
- `[DERIVATION]` Moving probability mass $\delta$ from a more-loaded expert $i$ to a less-loaded expert $j$ changes the loss by

  $$
  \Delta\mathcal L_{\mathrm{bal}}
  =\frac{\alpha N}{B}\delta(f_j-f_i)<0.
  $$
- `[DIRECT]` For top-$K$, normalize the hard load by $BK$:

  $$
  f_i^{(K)}=\frac1{BK}\sum_t\mathbf 1\{i\in S_t\}.
  $$

  The same fixed-routing gradient formula holds with $f_i^{(K)}$ replacing $f_i$.
- ## 3. Why power laws appear
- ### 3.1 Mean estimation
- `[EXAMPLE]` For $x_i\overset{\mathrm{iid}}\sim\mathcal N(\mu,\sigma^2)$ and $\widehat\mu=n^{-1}\sum_i x_i$,

  $$
  \mathbb E[(\widehat\mu-\mu)^2]
  =\operatorname{Var}(\widehat\mu)
  =\frac{\sigma^2}{n}.
  $$
- `[SYNTHESIS]` Mean squared error follows an $n^{-1}$ power law, while root mean squared error follows $n^{-1/2}$.
- ### 3.2 Approximation on a grid
- `[DIRECT]` Partition $[0,1]^d$ into cubes of side length $s$, so the number of cells is $P=s^{-d}$.
- `[DERIVATION]` For a $K$-Lipschitz function, replacing each cell by its center value gives pointwise error $O(s)$ and integrated squared error

  $$
  L(P)=O(s^2)=O(P^{-2/d}).
  $$
- `[DERIVATION]` If bounded second derivatives permit a local affine approximation, the remainder is $O(s^2)$, so squared error becomes

  $$
  L(P)=O(s^4)=O(P^{-4/d}).
  $$
- `[SYNTHESIS]` The exponent reflects both local approximation order and effective dimension. These are upper-bound intuitions, not a universal explanation of neural scaling.
- ## 4. Compute-optimal parameters and data
- ### 4.1 Loss model
- `[DIRECT]` Let $P$ be dense-model parameter count and $D$ the number of training tokens. The Hoffmann et al. model is

  $$
  L(D,P)=E+\frac{A}{P^\alpha}+\frac{B}{D^\beta},
  \qquad
  A,B,\alpha,\beta>0.
  $$
- `[SYNTHESIS]` $E$ models irreducible conditional entropy, $AP^{-\alpha}$ model-class approximation error, and $BD^{-\beta}$ finite-data plus finite-training error.
- `[DIRECT]` Dense training compute is approximated by

  $$
  C\approx6PD.
  $$
- ### 4.2 Deriving the optimal allocation
- `[DERIVATION · Step 1]` On the compute boundary, substitute

  $$
  D=\frac{C}{6P}.
  $$
- `[DERIVATION · Step 2]` The variable part of loss becomes

  $$
  \ell(P)=AP^{-\alpha}+B\left(\frac{6P}{C}\right)^\beta.
  $$
- `[DERIVATION · Step 3]` Differentiate and set the derivative to zero:

  $$
  -\alpha AP^{-\alpha-1}
  +\beta B\left(\frac6C\right)^\beta P^{\beta-1}=0.
  $$
- `[DERIVATION · Step 4]` Multiplying by $P$ reveals the balance condition

  $$
  \alpha AP^{-\alpha}=\beta BD^{-\beta}.
  $$
- `[DIRECT]` Solving gives

  $$
  P_{\mathrm{opt}}(C)
  =G\left(\frac C6\right)^{\beta/(\alpha+\beta)},
  \qquad
  D_{\mathrm{opt}}(C)
  =G^{-1}\left(\frac C6\right)^{\alpha/(\alpha+\beta)},
  $$

  where

  $$
  G=\left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}.
  $$
- `[CONCLUSION]` If $\alpha=\beta$, both parameters and tokens grow as $C^{1/2}$. The empirical methods discussed in the lecture estimate exponents close to one half for each.
- ### 4.3 Three empirical routes
- `[DIRECT]` The lecture distinguishes:
  1. the lower envelope of loss-versus-compute training curves;
  2. IsoFLOP profiles that vary model size while holding compute fixed; and
  3. a joint parametric fit of $(E,A,B,\alpha,\beta)$.
- `[SYNTHESIS]` Agreement between these routes is stronger evidence than a single extrapolation because the fitting geometries differ.
- ## 5. A linear-regression scaling law
- ### 5.1 Spectrum-driven setup
- `[DIRECT]` Assume diagonal covariance eigenvalues

  $$
  \lambda_j\asymp j^{-a},
  \qquad a>1,
  $$

  and use a $P$-parameter random sketch with one-pass SGD on $D$ examples.
- `[DIRECT]` The theorem in the lecture gives excess risk of order

  $$
  P^{-(a-1)}+(\gamma D_{\mathrm{eff}})^{-(a-1)/a},
  $$

  up to constants and the schedule's logarithmic factor.
- ### 5.2 Parameter exponent
- `[DERIVATION]` If a $P$-dimensional model captures roughly the first $P$ covariance directions, the remaining expected error is the spectral tail

  $$
  \sum_{j>P}\lambda_j
  \asymp
  \int_P^\infty u^{-a}\,du
  \asymp P^{-(a-1)}.
  $$
- `[CONCLUSION]` The parameter exponent is $a-1$.
- ### 5.3 Data exponent
- `[DIRECT]` In the coordinate idealization, the expected coefficient error contracts by approximately

  $$
  \prod_{t=1}^{D}(1-\gamma_t\lambda_i)
  \approx
  \exp(-\Theta(\gamma D_{\mathrm{eff}}\lambda_i)).
  $$
- `[DERIVATION]` A direction is substantially learned when

  $$
  \gamma D_{\mathrm{eff}}\lambda_i\gtrsim1.
  $$

  With $\lambda_i\asymp i^{-a}$, the number of learned directions is

  $$
  k\asymp(\gamma D_{\mathrm{eff}})^{1/a}.
  $$
- `[DERIVATION]` The unresolved spectral tail is therefore

  $$
  \sum_{j>k}\lambda_j
  \asymp k^{-(a-1)}
  \asymp(\gamma D_{\mathrm{eff}})^{-(a-1)/a}.
  $$
- `[SYNTHESIS]` The covariance spectrum controls both exponents. Parameters limit how many directions the model can represent; training data and step sizes limit how many of those directions optimization can actually learn.
- `[CHECK]` The lecture also bounds SGD variance under its geometric step-size schedule. Reconstruct Appendix A and the variance sum before treating the theorem as fully proved.
- ## 6. Minimal takeaways
- MoE increases stored capacity by routing each token through only a small active expert subset.
- A balancing loss couples hard traffic with soft probabilities and supplies a differentiable anti-collapse signal.
- Power-law exponents can arise from estimation variance, approximation dimension, or spectral decay.
- Under $C\approx6PD$, compute-optimal parameters and tokens balance their marginal loss reductions.
- Scaling-law fits describe an empirical regime; extrapolation depends on architecture, data, optimization, and the validity of the fitted model.
- ## 7. Questions for manual reconstruction
- [ ] For a top-2 router, compute the selected gates both with and without renormalization.
- [ ] Re-derive the load-balancing logit gradient from the softmax Jacobian.
- [ ] Explain why the proof of uniform load assumes $P=f$.
- [ ] Derive $P_{\mathrm{opt}}(C)$ and $D_{\mathrm{opt}}(C)$ without looking above.
- [ ] Explain the parameter and data exponents in the linear-regression example using a spectral cutoff picture.
- ## 8. Open questions
- `[CHECK]` How do capacity limits, dropped tokens, and cross-device communication change the simple active-compute picture?
- `[CHECK]` How should $C\approx6PD$ be modified for sparse MoE models and long contexts?
- `[HYPOTHESIS]` Expert specialization may alter the effective approximation exponent rather than merely the constant in front of a dense scaling law. This is a question, not a lecture conclusion.
- ## References
- `[DIRECT]` Noah Golowich. *CS 395T — Lecture 5: Mixture-of-Experts and Scaling Laws.* <https://noahgol.github.io/teaching/cs395t-f26/lecture5.pdf>
- `[DIRECT]` The central reading pointers are Switch Transformers for routing and load balance, Hoffmann et al. for compute-optimal language-model scaling, and Lin et al. for the linear-regression theorem. They have not been independently reconstructed here.

- ---
  title: "Lecture 06 — Optimization I"
  slug: "lecture-06"
  date: "2026-09-25"
  updated: "2026-09-25"
  status: "draft"
  type: "learning-note"
  course: "CS 395T"
  source: "https://noahgol.github.io/teaching/cs395t-f26/lecture6.pdf"
  summary: "A first reconstruction of stochastic-gradient convergence, AdaGrad and Adam, adaptive regret, and the Hessian-block motivation for Transformer optimization."
  public: true
  tags:
    - stochastic gradient descent
    - AdaGrad
    - Adam
    - regret
    - Hessian structure
  related_research_notes: []
  authorship: "ai-assisted"
  ---
- > **AI-assisted first draft.** This note reconstructs Lecture 06 and expands the central SGD and adaptive-optimization arguments. It should be verified and rewritten during manual study.
- ## Lecture context
- `[DIRECT]` Source: <https://noahgol.github.io/teaching/cs395t-f26/lecture6.pdf>
- `[SYNTHESIS]` The lecture moves from a basic nonconvex SGD guarantee to coordinate-adaptive methods, then asks why Transformer parameter blocks may benefit from different effective learning rates.
- ## Reading map

  | Part | Question | Main mathematical tool |
  | --- | --- | --- |
  | SGD | Does noisy descent approach stationarity? | smoothness and telescoping |
  | AdaGrad / Adam | Why rescale coordinates? | accumulated or moving second moments |
  | AdaGrad regret | What does adaptivity buy online? | coordinatewise potential bounds |
  | Hessian blocks | Why can one global step size be inefficient? | curvature eigenvalues |
- ## 1. Optimization and stochastic gradients
- `[DIRECT]` The population objective is

  $$
  \min_{\theta\in\mathcal B}F(\theta),
  \qquad
  F(\theta)=\mathbb E_{z\sim\mathcal D}[f(\theta,z)].
  $$
- `[DIRECT]` At iteration $t$, a batch $B_t=(z_{t,1},\ldots,z_{t,b})$ defines

  $$
  g_t
  =\frac1b\sum_{r=1}^{b}\nabla_\theta f(\theta_t,z_{t,r}).
  $$
- `[DIRECT]` Conditional on the history before drawing the batch,

  $$
  \mathbb E_t[g_t]=\nabla F(\theta_t),
  \qquad
  \mathbb E_t\|g_t-\nabla F(\theta_t)\|_2^2
  \le\frac{\sigma_0^2}{b}=:\sigma^2.
  $$
- `[SYNTHESIS]` Larger batches reduce this variance bound by $1/b$, but they do not change the underlying population gradient.
- ## 2. SGD reaches an approximate stationary point
- ### 2.1 Smoothness inequality
- `[DIRECT]` $F$ is $L$-smooth when

  $$
  \|\nabla F(\theta)-\nabla F(\theta')\|_2
  \le L\|\theta-\theta'\|_2.
  $$
- `[DERIVATION]` Along the segment $\theta+us$,

  $$
  F(\theta+s)-F(\theta)
  =\int_0^1\langle\nabla F(\theta+us),s\rangle\,du.
  $$
- `[DERIVATION]` Add and subtract $\nabla F(\theta)$ and use smoothness:

  $$
  F(\theta+s)
  \le F(\theta)+\langle\nabla F(\theta),s\rangle
  +\int_0^1Lu\|s\|_2^2\,du.
  $$
- `[CONCLUSION]` Therefore

  $$
  F(\theta+s)
  \le F(\theta)+\langle\nabla F(\theta),s\rangle
  +\frac L2\|s\|_2^2.
  $$
- ### 2.2 One-step expected descent
- `[DIRECT]` SGD updates

  $$
  \theta_{t+1}=\theta_t-\alpha g_t.
  $$
- `[DERIVATION · Step 1]` Substitute $s=-\alpha g_t$ into the smoothness inequality and take conditional expectation:

  $$
  \mathbb E_t[F(\theta_{t+1})]
  \le F(\theta_t)
  -\alpha\langle\nabla F(\theta_t),\mathbb E_tg_t\rangle
  +\frac{L\alpha^2}{2}\mathbb E_t\|g_t\|_2^2.
  $$
- `[DERIVATION · Step 2]` Unbiasedness and the bias-variance identity give

  $$
  \mathbb E_t\|g_t\|_2^2
  =\|\nabla F(\theta_t)\|_2^2
  +\mathbb E_t\|g_t-\nabla F(\theta_t)\|_2^2
  \le\|\nabla F(\theta_t)\|_2^2+\sigma^2.
  $$
- `[DERIVATION · Step 3]` If $\alpha\le1/L$, then

  $$
  \mathbb E_t[F(\theta_{t+1})]
  \le F(\theta_t)
  -\frac\alpha2\|\nabla F(\theta_t)\|_2^2
  +\frac{L\alpha^2\sigma^2}{2}.
  $$
- ### 2.3 Telescoping the proof
- `[DIRECT]` Let $F_*=\min_{\theta\in\mathcal B}F(\theta)$ and $\Delta=F(\theta_1)-F_*$. Sum the one-step bound from $1$ to $T$:

  $$
  \frac\alpha2
  \sum_{t=1}^{T}\mathbb E\|\nabla F(\theta_t)\|_2^2
  \le
  \Delta+rac{TL\alpha^2\sigma^2}{2}.
  $$
- `[DIRECT]` Dividing by $\alpha T/2$ gives

  $$
  \frac1T\sum_{t=1}^{T}
  \mathbb E\|\nabla F(\theta_t)\|_2^2
  \le
  \frac{2\Delta}{\alpha T}+L\alpha\sigma^2.
  $$
- `[SYNTHESIS]` The first term rewards a larger step; the second term amplifies gradient noise. The theorem exposes a bias-variance-like tradeoff in the step size.
- `[DIRECT]` Choosing $\alpha=1/(L\sqrt T)$ gives

  $$
  \mathbb E\|\nabla F(\theta_\tau)\|_2^2
  \le
  \frac{2L\Delta+\sigma_0^2/b}{\sqrt T},
  $$

  for a uniformly random iterate $\tau$.
- `[CHECK]` This proves approximate stationarity, not convergence to a global minimum. A small gradient can occur at a local minimum, saddle, or flat nonoptimal region.
- ## 3. Adaptive optimizers
- ### 3.1 Generic diagonal preconditioning
- `[DIRECT]` Adaptive methods maintain a direction $m_t$ and a positive diagonal scale $V_t=\operatorname{diag}(v_t)$:

  $$
  \theta_{t+1}
  =\theta_t-\alpha_tV_t^{-1/2}m_t.
  $$
- `[SYNTHESIS]` Coordinate $i$ receives effective step size $\alpha_t/\sqrt{v_{t,i}}$. Ordinary SGD corresponds to $m_t=g_t$ and $V_t=I$.
- ### 3.2 AdaGrad
- `[DIRECT]` AdaGrad accumulates squared gradients:

  $$
  s_t=s_{t-1}+g_t^2,
  \qquad
  \theta_{t+1,i}
  =\theta_{t,i}
  -\frac{\alpha g_{t,i}}{\sqrt{\epsilon+s_{t,i}}}.
  $$
- `[DERIVATION]` Ignoring $\epsilon$, if every historical gradient in one coordinate is multiplied by $c>0$, then both numerator and denominator scale by $c$. The update in that coordinate is unchanged.
- `[SYNTHESIS]` Rarely active coordinates accumulate less squared gradient and retain larger effective rates. The limitation is permanent memory: the denominator never forgets old large gradients.
- ### 3.3 Adam
- `[DIRECT]` Adam uses exponential moving averages

  $$
  m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
  $$

  $$
  v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
  $$

  followed by

  $$
  \theta_{t+1,i}
  =\theta_{t,i}
  -\alpha_t\frac{m_{t,i}}{\sqrt{v_{t,i}+\epsilon}}.
  $$
- `[DIRECT]` Bias correction replaces

  $$
  m_t\mapsto\frac{m_t}{1-\beta_1^t},
  \qquad
  v_t\mapsto\frac{v_t}{1-\beta_2^t},
  $$

  compensating for initialization at zero.
- `[DERIVATION]` For independent scalar gradients with variance $\nu^2$, an EMA with decay $\beta$ has limiting variance

  $$
  (1-\beta)^2\sum_{k=0}^{\infty}\beta^{2k}\nu^2
  =\frac{1-\beta}{1+\beta}\nu^2.
  $$
- `[SYNTHESIS]` Momentum reduces noise in a persistent direction but introduces lag when the gradient changes. Adam's second moment forgets old scale information, unlike AdaGrad.
- ## 4. AdaGrad's adaptive regret
- ### 4.1 Online setting
- `[DIRECT]` At round $t$, choose $\theta_t$ before seeing a convex loss $f_t$, observe $g_t=\nabla f_t(\theta_t)$, and compare with a fixed $\theta^*$ using

  $$
  R_T(\theta^*)
  =\sum_{t=1}^{T}\bigl[f_t(\theta_t)-f_t(\theta^*)\bigr].
  $$
- `[DIRECT]` With $s_{t,i}=\sum_{j=1}^{t}g_{j,i}^2$, coordinate AdaGrad yields a bound proportional to

  $$
  D_\infty\sum_{i=1}^{d}\sqrt{s_{T,i}}.
  $$
- `[SYNTHESIS]` This quantity adapts to the observed coordinate geometry rather than paying only for the global Euclidean norm.
- `[DERIVATION]` If all gradients are supported on the same $k$ coordinates, Cauchy-Schwarz gives

  $$
  \sum_{i=1}^{d}\sqrt{s_{T,i}}
  =\sum_{i\in S}\sqrt{s_{T,i}}
  \le\sqrt{k}
  \left(\sum_{t=1}^{T}\|g_t\|_2^2\right)^{1/2}.
  $$
- `[CONCLUSION]` The adaptive bound can replace ambient dimension dependence by effective active-coordinate dependence.
- `[CHECK]` Reconstruct the exact potential argument and constants in Theorem 3.2 before treating the regret proof as complete.
- ## 5. Hessian structure and Transformer optimization
- ### 5.1 Curvature controls stable step sizes
- `[DIRECT]` Near $\theta$, a twice-differentiable objective has the local expansion

  $$
  F(\theta+s)
  =F(\theta)+\langle\nabla F(\theta),s\rangle
  +\frac12s^\top H(\theta)s+R_3(s).
  $$
- `[DIRECT]` For a convex quadratic with Hessian $H\succeq0$, gradient descent error obeys

  $$
  e_{t+1}=(I-\alpha H)e_t.
  $$
- `[DERIVATION]` Along an eigenvector with eigenvalue $\lambda$, the scalar error is multiplied by $1-\alpha\lambda$. Every positive-curvature direction contracts when

  $$
  0<\alpha<\frac{2}{\lambda_{\max}(H)}.
  $$
- ### 5.2 Why one global rate can be inefficient
- `[EXAMPLE]` For

  $$
  Q(a,b)=\frac12(Ma^2+mb^2),
  \qquad M\gg m>0,
  $$

  the globally safe choice $\alpha=1/M$ removes the $a$ error in one step but shrinks $b$ only by $1-m/M$ per step. Separate rates $1/M$ and $1/m$ remove both immediately.
- `[DIRECT]` If the Hessian is approximately block diagonal,

  $$
  H\approx\operatorname{diag}(H_{11},\ldots,H_{KK}),
  $$

  a global step must accommodate the most curved block.
- `[DIRECT]` The lecture reports that sampled Transformer blocks have substantially different Hessian spectra, whereas sampled CNN blocks in the cited experiment are more similar.
- `[SYNTHESIS]` This block heterogeneity is evidence for one explanation of Adam's advantage: adaptive scaling can supply different effective rates to parameter groups with different curvature.
- `[CHECK]` This is empirical motivation, not a proof that Adam is optimal or that Hessian block structure is the only cause of its behavior.
- ## 6. Minimal takeaways
- Smoothness plus unbiased gradients gives a clean SGD stationarity bound through one-step descent and telescoping.
- A constant nonzero step leaves a noise floor; a horizon-dependent step balances progress and variance.
- AdaGrad remembers all squared gradients; Adam uses moving averages and can forget obsolete scale.
- Coordinate adaptivity can exploit sparse or heterogeneous gradient geometry.
- Transformer parameter blocks may have different curvature spectra, making one global learning rate inefficient.
- ## 7. Questions for manual reconstruction
- [ ] Prove the smoothness descent inequality from the line integral.
- [ ] Derive the SGD theorem without skipping the conditional expectation step.
- [ ] Explain why the theorem selects a random iterate.
- [ ] Compare AdaGrad and Adam memory on a coordinate whose gradients become small halfway through training.
- [ ] Derive the scalar contraction factor $1-\alpha\lambda$ for a quadratic.
- ## 8. Open questions
- `[CHECK]` How do momentum and preconditioning alter the simple Hessian-eigenvector picture?
- `[CHECK]` Which parameter partition best exposes Transformer block heterogeneity: tensors, layers, or functional groups such as $Q/K/V$?
- `[HYPOTHESIS]` Much of adaptive optimization's benefit may come from matching architectural blocks rather than individual scalar coordinates. Lecture 07–08 will provide matrix-aware alternatives for testing this idea.
- ## References
- `[DIRECT]` Noah Golowich. *CS 395T — Lecture 6: Optimization I.* <https://noahgol.github.io/teaching/cs395t-f26/lecture6.pdf>
- `[DIRECT]` Primary reading pointers in the lecture are Kingma and Ba for Adam, Duchi et al. for AdaGrad, and Zhang et al. for the Hessian perspective. They have not been independently reconstructed here.

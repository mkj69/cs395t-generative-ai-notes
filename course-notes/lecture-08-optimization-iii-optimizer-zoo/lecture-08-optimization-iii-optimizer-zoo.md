- ---
  title: "Lecture 08 — Optimization III: The Optimizer Zoo"
  slug: "lecture-08"
  date: "2026-09-25"
  updated: "2026-09-25"
  status: "draft"
  type: "learning-note"
  course: "CS 395T"
  source: "https://noahgol.github.io/teaching/cs395t-f26/lecture8.pdf"
  summary: "A first reconstruction of optimizer updates as norm-dependent steepest descent, together with full-matrix AdaGrad, Shampoo, and SOAP."
  public: true
  tags:
    - steepest descent
    - Shampoo
    - SOAP
    - full-matrix AdaGrad
    - optimizer geometry
  related_research_notes: []
  authorship: "ai-assisted"
  ---
- > **AI-assisted first draft.** This note reconstructs Lecture 08 and expands the norm-duality, Kronecker, and rotated-basis derivations. It is a learning scaffold, not a finished personal account.
- ## Lecture context
- `[DIRECT]` Source: <https://noahgol.github.io/teaching/cs395t-f26/lecture8.pdf>
- `[SYNTHESIS]` Optimizers can be organized by the geometry in which they define a small step. SGD uses Euclidean geometry, sign descent uses coordinatewise $\ell_\infty$ geometry, Muon uses the matrix spectral norm, Shampoo uses Kronecker-factored second moments, and SOAP applies Adam in a changing matrix-derived basis.
- ## Reading map

  | Optimizer | Stored statistics | Geometry / transformation |
  | --- | --- | --- |
  | SGD | current gradient | Euclidean |
  | AdaGrad | coordinate squares | diagonal rescaling |
  | Full-matrix AdaGrad | outer products | unrestricted linear preconditioner |
  | Muon | momentum matrix | spectral normalization |
  | Shampoo | row/column Gram matrices | Kronecker-factored preconditioning |
  | SOAP | Gram eigenvectors + Adam moments | Adam in a rotated basis |
- ## 1. Review of update rules
- `[DIRECT]` Diagonal AdaGrad uses

  $$
  s_t=\sum_{\tau=1}^{t}g_\tau^2,
  \qquad
  \theta_{t+1,i}
  =\theta_{t,i}
  -\alpha_t\frac{g_{t,i}}{\sqrt{s_{t,i}+\epsilon}}.
  $$
- `[DIRECT]` Full-matrix AdaGrad accumulates

  $$
  C_t=\sum_{\tau=1}^{t}g_\tau g_\tau^\top
  $$

  and updates

  $$
  \theta_{t+1}=\theta_t-\alpha_tC_t^{-1/2}g_t,
  $$

  with stabilization in practice.
- `[DIRECT]` Idealized Muon, for $B_t=U_t\Sigma_tV_t^\top$, uses

  $$
  W_{t+1}=W_t-\alpha_tU_tV_t^\top.
  $$
- ## 2. Optimizers as steepest descent
- ### 2.1 Gradient descent from a Euclidean penalty
- `[DIRECT]` Linearize the loss at $\theta$ with $g=\nabla f(\theta)$ and solve

  $$
  \min_\Delta
  \left\{
  f(\theta)+\langle g,\Delta\rangle
  +\frac1{2\alpha}\|\Delta\|_2^2
  \right\}.
  $$
- `[DERIVATION]` Complete the square:

  $$
  \langle g,\Delta\rangle
  +\frac1{2\alpha}\|\Delta\|_2^2
  =-\frac\alpha2\|g\|_2^2
  +\frac1{2\alpha}\|\Delta+\alpha g\|_2^2.
  $$
- `[CONCLUSION]` The unique minimizer is

  $$
  \Delta^*=-\alpha g.
  $$
- ### 2.2 Sign descent from a max-norm penalty
- `[DIRECT]` Consider

  $$
  \min_\Delta
  \left\{
  \langle g,\Delta\rangle
  +\frac\lambda2\|\Delta\|_\infty^2
  \right\}.
  $$
- `[DERIVATION]` Put $a=\|\Delta\|_\infty$. Coordinatewise,

  $$
  g_i\Delta_i\ge-|g_i|a,
  $$

  so

  $$
  \langle g,\Delta\rangle
  +\frac\lambda2a^2
  \ge-a\|g\|_1+\frac\lambda2a^2.
  $$
- `[DERIVATION]` The scalar quadratic is minimized at $a=\|g\|_1/\lambda$. Equality in the coordinate bounds is reached by choosing every nonzero coordinate opposite to $g_i$.
- `[CONCLUSION]` One minimizer is

  $$
  \Delta^*
  =-\frac{\|g\|_1}{\lambda}\operatorname{sign}(g).
  $$
- `[SYNTHESIS]` Sign descent is steepest descent under an $\ell_\infty$ notion of step size; $\ell_1$ appears because it is the dual norm.
- ### 2.3 Matrix analogue: spectral norm
- `[DIRECT]` For a matrix,

  $$
  \|A\|_{\mathrm{op}}
  =\sup_{\|x\|_2=1}\|Ax\|_2
  =\max_i\sigma_i(A).
  $$
- `[DIRECT]` Its dual under the Frobenius inner product is the nuclear norm

  $$
  \|G\|_*=\sum_i\sigma_i(G).
  $$
- `[DIRECT]` For $G=U\Sigma V^\top$, spectral steepest descent solves

  $$
  \min_\Delta
  \left\{
  \langle G,\Delta\rangle_F
  +\frac\lambda2\|\Delta\|_{\mathrm{op}}^2
  \right\}.
  $$
- `[DERIVATION]` With $a=\|\Delta\|_{\mathrm{op}}$,

  $$
  \langle G,\Delta\rangle_F
  =\sum_i\sigma_i u_i^\top\Delta v_i
  \ge-a\sum_i\sigma_i
  =-a\|G\|_*.
  $$
- `[DERIVATION]` Minimizing $-a\|G\|_*+\lambda a^2/2$ gives $a=\|G\|_*/\lambda$.
- `[CONCLUSION]` The bound is attained by

  $$
  \Delta^*
  =-\frac{\|G\|_*}{\lambda}UV^\top.
  $$
- `[SYNTHESIS]` Orthogonalized gradients are the matrix analogue of sign gradients when step size is measured by spectral norm.
- ## 3. Shampoo
- ### 3.1 Row and column statistics
- `[DIRECT]` For $W_t\in\mathbb{R}^{m\times n}$ and gradient $G_t$, Shampoo accumulates

  $$
  L_t=\epsilon I_m+\sum_{s=1}^{t}G_sG_s^\top,
  \qquad
  R_t=\epsilon I_n+\sum_{s=1}^{t}G_s^\top G_s.
  $$
- `[DIRECT]` Its update is

  $$
  W_{t+1}
  =W_t-\alpha L_t^{-1/4}G_tR_t^{-1/4}.
  $$
- ### 3.2 Removing accumulation recovers Muon without momentum
- `[DERIVATION]` Replace $L_t,R_t$ by current Gram matrices and let $G=U\Sigma V^\top$. Then

  $$
  (GG^\top)^{-1/4}=U\Sigma^{-1/2}U^\top,
  $$

  $$
  (G^\top G)^{-1/4}=V\Sigma^{-1/2}V^\top.
  $$
- `[DERIVATION]` Multiplying gives

  $$
  (GG^\top)^{-1/4}G(G^\top G)^{-1/4}
  =UV^\top.
  $$
- `[CONCLUSION]` Without second-moment accumulation, Shampoo's two-sided preconditioning is Muon's exact orthogonalization with no momentum.
- ### 3.3 Kronecker-factored full-matrix view
- `[DIRECT]` Stack matrix rows into $\operatorname{vec}(G)$. The identity

  $$
  \operatorname{vec}(LGR)
  =(L\otimes R^\top)\operatorname{vec}(G)
  $$

  turns two-sided multiplication into one vector-space preconditioner.
- `[DERIVATION]` Because $L_t,R_t$ are symmetric,

  $$
  H_t=L_t^{1/4}\otimes R_t^{1/4}
  $$

  yields

  $$
  w_{t+1}=w_t-\alpha H_t^{-1}g_t.
  $$
- `[SYNTHESIS]` Full-matrix AdaGrad stores an unrestricted $mn\times mn$ statistic. Shampoo approximates its geometry with row and column factors, storing $m^2+n^2$ entries instead of $m^2n^2$.
- ### 3.4 Regret comparison
- `[DIRECT]` Under rank-$r$ gradients and a Frobenius iterate-distance bound $D$, the lecture states the Shampoo guarantee

  $$
  R_T(W^*)
  \le\sqrt{2r}\,D
  \operatorname{tr}(L_T^{1/4})
  \operatorname{tr}(R_T^{1/4}).
  $$
- `[DIRECT]` The comparable full-matrix AdaGrad bound is

  $$
  R_T(W^*)
  \le\sqrt2D\operatorname{tr}(C_T^{1/2}).
  $$
- `[SYNTHESIS]` The stated Shampoo bound is not automatically tighter. Its advantage is a structured, tractable preconditioner; the price can be a looser guarantee.
- `[CHECK]` Reconstruct the rank-$r$ example and the positive-semidefinite inequality comparing these two bounds before treating the regret section as complete.
- ## 4. SOAP
- ### 4.1 A tensor-product basis
- `[DIRECT]` Let

  $$
  L_{t-1}=Q_{L,t}\Lambda_{L,t}Q_{L,t}^\top,
  \qquad
  R_{t-1}=Q_{R,t}\Lambda_{R,t}Q_{R,t}^\top.
  $$
- `[DIRECT]` The induced vector-space basis is

  $$
  Q_t=Q_{L,t}\otimes Q_{R,t}.
  $$
- `[DERIVATION]` It is orthogonal because

  $$
  Q_t^\top Q_t
  =(Q_{L,t}^\top Q_{L,t})
  \otimes
  (Q_{R,t}^\top Q_{R,t})
  =I_{mn}.
  $$
- ### 4.2 Matrix-coordinate implementation
- `[DIRECT]` Rotate the gradient into the current basis:

  $$
  G_t'=Q_{L,t}^\top G_tQ_{R,t}.
  $$
- `[DIRECT]` Maintain original-coordinate first momentum

  $$
  M_t=\beta_1M_{t-1}+(1-\beta_1)G_t
  $$

  and rotate it when needed:

  $$
  M_t'=Q_{L,t}^\top M_tQ_{R,t}.
  $$
- `[DIRECT]` Maintain the second moment in rotated coordinates:

  $$
  V_t=\beta_2V_{t-1}+(1-\beta_2)(G_t'\odot G_t').
  $$
- `[DIRECT]` Normalize entrywise, rotate back, and update:

  $$
  N_t'=\frac{M_t'}{\sqrt{V_t+\epsilon}},
  \qquad
  N_t=Q_{L,t}N_t'Q_{R,t}^\top,
  \qquad
  W_{t+1}=W_t-\alpha_tN_t.
  $$
- ### 4.3 Why this equals the vectorized algorithm
- `[DERIVATION]` The vec identity gives

  $$
  \operatorname{vec}(Q_{L,t}^\top AQ_{R,t})
  =Q_t^\top\operatorname{vec}(A).
  $$
- `[SYNTHESIS]` Every matrix-side rotation, entrywise operation, and inverse rotation therefore matches the corresponding vector-space basis change without constructing the $mn\times mn$ matrix $Q_t$.
- ### 4.4 Adam in a rotated basis
- `[DIRECT]` If $Q$ were fixed and $z=Q^\top w$, then $g'=Q^\top g$ is the gradient in $z$ coordinates. SOAP's moment and parameter recurrences become exactly Adam in those coordinates.
- `[CHECK]` In actual SOAP, the basis changes with $t$. The first moment is maintained in original coordinates and reprojected exactly; the second moment stores squared gradients measured in their historical bases.
- ### 4.5 Connection back to Shampoo
- `[DIRECT]` The tensor-product eigenbasis also diagonalizes Shampoo's preconditioner:

  $$
  L_t^{1/4}\otimes R_t^{1/4}
  =Q_t
  \left(\Lambda_{L,t}^{1/4}\otimes\Lambda_{R,t}^{1/4}\right)
  Q_t^\top.
  $$
- `[SYNTHESIS]` Shampoo and SOAP share a matrix-derived basis. Shampoo uses the analytic scaling $(\ell_i\rho_j)^{-1/4}$; SOAP learns an Adam-style coordinate scaling inside that basis.
- ## 5. Optimizer-family map
- `[SYNTHESIS]` The lecture's optimizer zoo can be read as a sequence of design decisions:
  1. Should scaling be scalar, coordinatewise, or matrix-aware?
  2. Should second moments accumulate forever or use an EMA?
  3. Should matrix structure be represented fully or with Kronecker row/column factors?
  4. Should the factors directly define the scale, or only define a basis in which Adam learns the scale?
- ## 6. Minimal takeaways
- A gradient direction is incomplete without specifying the norm that measures allowable steps.
- SGD, sign descent, and Muon are steepest-descent updates in Euclidean, $\ell_\infty$, and spectral geometries respectively.
- Shampoo approximates full-matrix adaptation using row and column Gram factors.
- Without accumulation, Shampoo collapses algebraically to Muon without momentum.
- SOAP uses Shampoo-like eigenvectors to choose coordinates and Adam to normalize within those coordinates.
- Structured preconditioners trade expressivity for feasible storage and computation.
- ## 7. Questions for manual reconstruction
- [ ] Complete the square to recover gradient descent.
- [ ] Derive sign descent using $\ell_1$-$\ell_\infty$ duality.
- [ ] Repeat the spectral steepest-descent proof using nuclear-spectral duality.
- [ ] Starting from an SVD, show that current-step Shampoo equals $UV^\top$.
- [ ] Prove the vec/Kronecker identity coordinate by coordinate.
- [ ] Explain precisely what SOAP stores in the current basis and what remains tied to past bases.
- ## 8. Open questions
- `[CHECK]` How often can practical SOAP recompute eigendecompositions before their cost dominates?
- `[CHECK]` How do numerical stabilizers and low-rank Gram matrices change the ideal formulas?
- `[HYPOTHESIS]` An optimizer's most important inductive bias may be its invariance under reparameterization. Compare the invariance groups of Adam, Muon, Shampoo, and SOAP.
- ## References
- `[DIRECT]` Noah Golowich. *CS 395T — Lecture 8: Optimization III — The Optimizer Zoo.* <https://noahgol.github.io/teaching/cs395t-f26/lecture8.pdf>
- `[DIRECT]` The lecture's primary reading pointers are Duchi et al. for AdaGrad, Gupta et al. for Shampoo, Vyas et al. for SOAP, and Bernstein and Newhouse for norm-based optimizer geometry. They have not been independently reconstructed here.

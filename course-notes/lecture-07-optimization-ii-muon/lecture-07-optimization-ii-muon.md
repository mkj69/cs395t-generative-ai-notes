- ---
  title: "Lecture 07 — Optimization II: Muon"
  slug: "lecture-07"
  date: "2026-09-25"
  updated: "2026-09-25"
  status: "draft"
  type: "learning-note"
  course: "CS 395T"
  source: "https://noahgol.github.io/teaching/cs395t-f26/lecture7.pdf"
  summary: "A first reconstruction of matrix orthogonalization, Muon's momentum update, and the Newton-Schulz polynomial iteration used to approximate the polar factor."
  public: true
  tags:
    - Muon
    - Newton-Schulz
    - orthogonalization
    - singular values
    - matrix-aware optimization
  related_research_notes: []
  authorship: "ai-assisted"
  ---
- > **AI-assisted first draft.** This note follows Lecture 07 and expands the proofs that orthogonalization is the nearest semi-orthogonal matrix and that Newton-Schulz acts independently on singular values.
- ## Lecture context
- `[DIRECT]` Source: <https://noahgol.github.io/teaching/cs395t-f26/lecture7.pdf>
- `[SYNTHESIS]` Adam normalizes scalar coordinates. Muon instead respects the fact that neural-network weights are matrices acting between activation spaces: it keeps a momentum matrix's singular vectors and approximately normalizes its singular values.
- ## Reading map

  | Part | Scalar/vector analogue | Matrix operation |
  | --- | --- | --- |
  | Sign gradient | normalize coordinates | entrywise sign |
  | Orthogonalization | normalize magnitude | set positive singular values to one |
  | Muon | momentum plus normalization | orthogonalize momentum buffer |
  | Newton-Schulz | scalar fixed-point iteration | polynomial matrix multiplication |
- ## 1. Why entrywise signs are not matrix geometry
- `[DIRECT]` Adam with $\beta_1=\beta_2=0$ and no stabilizer becomes

  $$
  \theta_{t+1,i}
  =\theta_{t,i}
  -\alpha_t\frac{g_{t,i}}{\sqrt{g_{t,i}^2}}
  =\theta_{t,i}-\alpha_t\operatorname{sign}(g_{t,i}).
  $$
- `[DIRECT]` For

  $$
  G=
  \begin{pmatrix}
  2&1\\
  1&2
  \end{pmatrix},
  \qquad
  \operatorname{sign}(G)=
  \begin{pmatrix}
  1&1\\
  1&1
  \end{pmatrix},
  $$

  the original matrix has eigenvalues $3$ and $1$, while its sign matrix annihilates the $(1,-1)^\top$ direction.
- `[SYNTHESIS]` Entrywise normalization depends on the chosen coordinate basis and can destroy a direction that the original linear map treats nontrivially.
- ## 2. Orthogonalization
- ### 2.1 Target definition
- `[DIRECT]` Let $G\in\mathbb{R}^{m\times n}$ have full rank $r=\min\{m,n\}$. A matrix $O$ is semi-orthogonal if

  $$
  O^\top O=I_n\quad(m\ge n),
  \qquad\text{or}\qquad
  OO^\top=I_m\quad(m\le n).
  $$
- `[DIRECT]` Orthogonalization is the nearest semi-orthogonal matrix in Frobenius norm:

  $$
  \operatorname{Ortho}(G)
  =\arg\min_{O\text{ semi-orthogonal}}\|O-G\|_F.
  $$
- `[DIRECT]` For a thin SVD

  $$
  G=U\Sigma V^\top,
  $$

  with positive diagonal singular values, the result is

  $$
  \operatorname{Ortho}(G)=UV^\top.
  $$
- ### 2.2 Why the polar factor is nearest
- `[DERIVATION · Step 1]` Every feasible semi-orthogonal $O$ has

  $$
  \|O\|_F^2=r.
  $$
- `[DERIVATION · Step 2]` Expand the objective:

  $$
  \|O-G\|_F^2
  =r+\|G\|_F^2-2\operatorname{tr}(O^\top G).
  $$
- `[DERIVATION · Step 3]` Writing $G=\sum_{i=1}^{r}\sigma_i u_iv_i^\top$,

  $$
  \operatorname{tr}(O^\top G)
  =\sum_{i=1}^{r}\sigma_i u_i^\top Ov_i.
  $$
- `[DERIVATION · Step 4]` A semi-orthogonal map is nonexpansive, so

  $$
  u_i^\top Ov_i
  \le\|u_i\|_2\|Ov_i\|_2
  \le1.
  $$

  Consequently,

  $$
  \operatorname{tr}(O^\top G)\le\sum_i\sigma_i.
  $$
- `[DERIVATION · Step 5]` $O=UV^\top$ is feasible and satisfies $Ov_i=u_i$, attaining equality in every term.
- `[CONCLUSION]` Maximizing the trace minimizes the squared distance, so $UV^\top$ is the desired orthogonalization.
- `[SYNTHESIS]` $Gv_i=\sigma_i u_i$ becomes $\operatorname{Ortho}(G)v_i=u_i$: singular directions remain, but their positive singular values become one.
- ### 2.3 Change-of-basis behavior
- `[DIRECT]` For orthogonal $P$ and $Q$,

  $$
  \operatorname{Ortho}(PGQ^\top)
  =P\operatorname{Ortho}(G)Q^\top.
  $$
- `[DERIVATION]` If $G=U\Sigma V^\top$, then $PGQ^\top=(PU)\Sigma(QV)^\top$ is another SVD. Removing $\Sigma$ produces the identity above.
- `[SYNTHESIS]` This equivariance makes orthogonalization compatible with Euclidean changes of basis in the input and output activation spaces.
- ## 3. Muon
- `[DIRECT]` For one hidden-layer weight matrix $W_t$, let $G_t=\nabla_W f_t(W_t)$. Muon maintains momentum

  $$
  B_t=\mu B_{t-1}+G_t,
  $$

  approximates its orthogonalization,

  $$
  O_t=\operatorname{NewtonSchulz5}(B_t),
  $$

  and updates

  $$
  W_{t+1}=W_t-\alpha_tO_t.
  $$
- `[SYNTHESIS]` Momentum is accumulated **before** normalization. Magnitudes influence the temporal average $B_t$, but the final update approximately equalizes the singular values of that averaged direction.
- ## 4. Newton-Schulz as a singular-value iteration
- ### 4.1 Polynomial matrix update
- `[DIRECT]` Consider

  $$
  X_{k+1}
  =aX_k+b(X_kX_k^\top)X_k
  +c(X_kX_k^\top)^2X_k.
  $$
- `[DERIVATION]` If $X_k=US_kV^\top$, then

  $$
  X_kX_k^\top=US_k^2U^\top,
  $$

  $$
  (X_kX_k^\top)X_k=US_k^3V^\top,
  $$

  and

  $$
  (X_kX_k^\top)^2X_k=US_k^5V^\top.
  $$
- `[CONCLUSION]` Hence

  $$
  X_{k+1}
  =U\bigl(aS_k+bS_k^3+cS_k^5\bigr)V^\top.
  $$

  The singular vectors stay fixed, while each singular value follows

  $$
  \phi(x)=ax+bx^3+cx^5.
  $$
- ### 4.2 Initialization
- `[DIRECT]` Start from

  $$
  X_0=\frac{G}{\|G\|_F}.
  $$
- `[DERIVATION]` Since $\|G\|_F^2=\sum_i\sigma_i^2$, every normalized singular value lies in $[0,1]$.
- `[DIRECT]` After $K$ steps,

  $$
  X_K
  =U\operatorname{diag}\!\left(
  \phi^{\circ K}\!\left(\frac{\sigma_1}{\|G\|_F}\right),
  \ldots,
  \phi^{\circ K}\!\left(\frac{\sigma_r}{\|G\|_F}\right)
  \right)V^\top.
  $$
- `[CHECK]` $\phi^{\circ K}$ means composition $K$ times, not a scalar power.
- ### 4.3 Classical cubic convergence
- `[DIRECT]` The classical choice is

  $$
  \phi(x)=\frac{3x-x^3}{2}.
  $$
- `[DERIVATION]` For $0<x<1$,

  $$
  \phi(x)-x=\frac{x(1-x^2)}2>0,
  $$

  while

  $$
  1-\phi(x)=\frac{(1-x)^2(x+2)}2>0.
  $$
- `[CONCLUSION]` Iterates increase but remain below one, so they converge. A positive limit $\ell$ must satisfy $\ell=\phi(\ell)$, whose only positive solution in $(0,1]$ is $1$. Zero singular values remain zero.
- ## 5. NewtonSchulz5 in Muon
- `[DIRECT]` Muon uses

  $$
  (a,b,c)=(3.4445,-4.7750,2.0315),
  \qquad
  \epsilon_{\mathrm{NS}}=10^{-7}.
  $$
- `[DIRECT]` Normalize

  $$
  X\leftarrow\frac{G}{\|G\|_F+\epsilon_{\mathrm{NS}}}.
  $$
- `[DIRECT]` If $m>n$, transpose $X$ so the Gram matrix is formed in the smaller dimension. Repeat five times:

  $$
  A\leftarrow XX^\top,
  \qquad
  D\leftarrow bA+cA^2,
  \qquad
  X\leftarrow aX+DX.
  $$
- `[DERIVATION]` Expanding $DX$ gives

  $$
  aX+b(XX^\top)X+c(XX^\top)^2X,
  $$

  exactly the polynomial update above.
- `[DIRECT]` Transpose back if necessary and return $X$.
- `[SYNTHESIS]` The tuned quintic is not required to converge asymptotically to exact orthogonalization in the same monotone way as the classical cubic. Its purpose is to move small singular values rapidly toward an acceptable band within only five matrix-multiplication rounds.
- ## 6. Worked singular-value picture
- `[EXAMPLE]` Suppose

  $$
  G=U\operatorname{diag}(4,1)V^\top.
  $$
- `[DERIVATION]` Its Frobenius norm is $\sqrt{17}$, so Newton-Schulz starts with singular values

  $$
  \left(\frac4{\sqrt{17}},\frac1{\sqrt{17}}\right).
  $$
- `[SYNTHESIS]` Every polynomial step changes these two scalars but leaves $U,V$ fixed. Exact orthogonalization would end at $(1,1)$ and return $UV^\top$.
- ## 7. Minimal takeaways
- Entrywise sign normalization is basis-dependent and may destroy matrix directions.
- Orthogonalization preserves singular vectors while replacing positive singular values by one.
- Muon orthogonalizes a momentum buffer rather than the instantaneous gradient.
- Newton-Schulz avoids an explicit SVD by applying a polynomial to all singular values in parallel.
- The matrix implementation is composed of accelerator-friendly matrix multiplications.
- ## 8. Questions for manual reconstruction
- [ ] Prove the nearest semi-orthogonal matrix result from the trace expansion.
- [ ] Verify the change-of-basis equivariance using an SVD.
- [ ] Starting from $X_k=US_kV^\top$, derive the $S_k^3$ and $S_k^5$ terms.
- [ ] Prove monotone convergence of the classical cubic scalar map.
- [ ] Explain why the optional transpose reduces the Gram-matrix size.
- ## 9. Open questions
- `[CHECK]` Which parameter matrices should use Muon, and which should remain under AdamW in practical implementations?
- `[CHECK]` How sensitive are five-step approximations to very small singular values and low precision?
- `[HYPOTHESIS]` Orthogonalizing the update may be understood as choosing a geometry on linear maps rather than merely a numerical preconditioner. Lecture 08 makes this norm-based interpretation precise.
- ## References
- `[DIRECT]` Noah Golowich. *CS 395T — Lecture 7: Optimization II — Muon.* <https://noahgol.github.io/teaching/cs395t-f26/lecture7.pdf>
- `[DIRECT]` The lecture points to the Muon post by Jordan et al. and Bernstein and Newhouse's norm-based optimizer anthology. They have not been independently reconstructed here.

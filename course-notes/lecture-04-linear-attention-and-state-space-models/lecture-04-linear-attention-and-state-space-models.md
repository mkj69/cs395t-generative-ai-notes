- ---
  title: "Lecture 04 — Linear Attention and State Space Models"
  slug: "lecture-04"
  date: "2026-09-25"
  updated: "2026-09-25"
  status: "draft"
  type: "learning-note"
  course: "CS 395T"
  source: "https://noahgol.github.io/teaching/cs395t-f26/lecture4.pdf"
  summary: "A first reconstruction of attention FLOPs, linear attention as a recurrent computation, state space duality, Mamba-2, and conditional barriers to subquadratic attention alternatives."
  public: true
  tags:
    - linear attention
    - state space models
    - Mamba-2
    - semiseparable matrices
    - fine-grained complexity
  related_research_notes: []
  authorship: "ai-assisted"
  ---
- > **AI-assisted first draft.** This note reconstructs the route of the Lecture 04 PDF and adds explicit algebra, proof outlines, and examples. It should be checked and rewritten during manual study. `[DIRECT]` marks lecture content; `[DERIVATION]`, `[SYNTHESIS]`, and `[EXAMPLE]` are explanatory additions.
- ## Lecture context
- `[DIRECT]` Source: <https://noahgol.github.io/teaching/cs395t-f26/lecture4.pdf>
- `[SYNTHESIS]` The lecture asks whether we can preserve useful sequence mixing while avoiding the $T^2$ pairwise cost of ordinary attention. Linear attention, recurrent state space models, and Mamba-2 offer structured alternatives; fine-grained lower bounds clarify what such structure may sacrifice.
- ## Reading map

  | Part | Central object | Main question |
  | --- | --- | --- |
  | FLOP accounting | $T,d,H,d_{\mathrm{ff}}$ | Where does Transformer compute go? |
  | Linear attention | feature map $\phi$ | Can associativity remove the $T\times T$ matrix? |
  | State space models | recurrent state $H_t$ | Which causal matrices admit a small recurrent state? |
  | Mamba-2 / SSD | scalar transition $a_tI$ | How do recurrence and masked matrix multiplication become dual views? |
  | Conditional barriers | Orthogonal Vectors | Why can exact global pairwise interactions resist truly subquadratic algorithms? |
- ## 1. FLOP accounting
- `[DIRECT]` Let $P_M$ be the number of key-query pairs allowed by a mask $M$:

  $$
  P_M=\left|\{(j,i)\in[T]^2:M_{ji}=1\}\right|.
  $$
- `[DIRECT]` Thus $P_M=T^2$ for unmasked attention and $P_M=T(T+1)/2$ for a causal mask, provided the implementation actually skips the forbidden triangle.
- ### 1.1 One-layer leading terms

  | Operation | Leading FLOPs |
  | --- | ---: |
  | Query, key, value projections | $6Td^2$ |
  | Attention scores | $2P_Md$ |
  | Weighted value sums | $2P_Md$ |
  | Attention output projection | $2Td^2$ |
  | Two-matrix MLP | $4Tdd_{\mathrm{ff}}$ |
  | SwiGLU MLP | $6Tdd_{\mathrm{ff}}$ |
- `[DERIVATION]` For unmasked attention, the leading layer cost is

  $$
  8Td^2+4T^2d+O(T^2H+Td).
  $$
- `[DERIVATION]` Efficient triangular causal attention replaces $4T^2d$ by

  $$
  2T(T+1)d,
  $$

  but a dense implementation that still materializes every score pays the unmasked cost.
- `[DIRECT]` With $d_{\mathrm{ff}}=4d$, the ordinary MLP costs $16Td^2$. A parameter-matched SwiGLU using $d_{\mathrm{ff}}\approx 8d/3$ has the same leading count.
- ### 1.2 Prefill versus generation
- `[DIRECT]` A full-sequence pass computes all required positions together. During autoregressive generation, a key-value cache stores earlier keys and values, so the new token attends to a length-$t$ prefix in $\Theta(td)$ work per layer rather than recomputing the full $t^2$ score matrix.
- `[DERIVATION]` Generating $T$ new tokens still sums the growing prefix costs:

  $$
  \sum_{t=1}^{T}\Theta(td)=\Theta(T^2d).
  $$
- `[SYNTHESIS]` KV caching removes repeated computation, but it does not remove the need for each new query to interact with an increasingly long history.
- ## 2. Linear attention
- ### 2.1 Replacing softmax by a feature kernel
- `[DIRECT]` The exponential dot-product kernel has the formal feature expansion

  $$
  e^{\langle q,k\rangle}
  =\sum_{a=0}^{\infty}\frac{\langle q,k\rangle^a}{a!}
  =\langle\phi_\infty(q),\phi_\infty(k)\rangle.
  $$
- `[DIRECT]` Linear attention instead chooses a finite feature map $\phi:\mathbb{R}^m\to\mathbb{R}^r$. With

  $$
  q_t=W_Qx_t,\qquad k_t=W_Kx_t,\qquad v_t=W_Vx_t,
  $$

  its unnormalized output is

  $$
  y_t=\sum_{s=1}^{T}\langle\phi(q_t),\phi(k_s)\rangle v_s.
  $$
- `[CHECK]` Unlike softmax attention, these coefficients are not automatically nonnegative and need not sum to one.
- ### 2.2 Associativity removes the pairwise matrix
- `[DIRECT]` Collect the columns as

  $$
  \Phi_Q=[\phi(q_1),\ldots,\phi(q_T)]\in\mathbb{R}^{r\times T},
  \quad
  \Phi_K=[\phi(k_1),\ldots,\phi(k_T)]\in\mathbb{R}^{r\times T},
  $$

  and $V=[v_1,\ldots,v_T]\in\mathbb{R}^{d\times T}$. Then

  $$
  Y=V\Phi_K^\top\Phi_Q.
  $$
- `[DERIVATION]` Direct evaluation forms the $T\times T$ kernel matrix. Associativity allows

  $$
  Y=(V\Phi_K^\top)\Phi_Q.
  $$
- `[DERIVATION]` The summary

  $$
  S:=V\Phi_K^\top=\sum_{s=1}^{T}v_s\phi(k_s)^\top\in\mathbb{R}^{d\times r}
  $$

  costs $O(Tdr)$ to form. Reading all queries via $y_t=S\phi(q_t)$ costs another $O(Tdr)$. Only $S$ must persist between the two stages.
- `[CONCLUSION]` For fixed feature dimension $r$, the sequence-length dependence is linear rather than quadratic.
- ### 2.3 Causal linear attention is an RNN
- `[DIRECT]` The causal output is

  $$
  y_t=\sum_{s=1}^{t}\langle\phi(q_t),\phi(k_s)\rangle v_s.
  $$
- `[DIRECT]` Initialize $S_0=0\in\mathbb{R}^{d\times r}$ and update

  $$
  S_t=S_{t-1}+v_t\phi(k_t)^\top,
  \qquad
  y_t=S_t\phi(q_t).
  $$
- `[DERIVATION]` Induction gives

  $$
  S_t=\sum_{s=1}^{t}v_s\phi(k_s)^\top.
  $$

  Substituting into the read operation recovers the causal attention formula exactly.
- `[SYNTHESIS]` The state is a compressed collection of all past key-value contributions. The key writes, the value supplies content, and the query reads from the fixed-size state.
- ## 3. State space models
- ### 3.1 General recurrence
- `[DIRECT]` With state dimension $N$, an SSM uses

  $$
  H_0=0,
  \qquad
  H_t=A_tH_{t-1}+B_tx_t^\top,
  \qquad
  y_t=H_t^\top C_t,
  $$

  where $H_t\in\mathbb{R}^{N\times d}$, $A_t\in\mathbb{R}^{N\times N}$, and $B_t,C_t\in\mathbb{R}^N$.
- `[SYNTHESIS]` Linear attention copies the old state and adds a rank-one write. An SSM additionally transforms the old state with $A_t$, allowing coordinates to decay, persist, or mix.
- ### 3.2 Unrolling the recurrence
- `[DIRECT]` Define

  $$
  A_{t:s+1}=A_tA_{t-1}\cdots A_{s+1},
  \qquad
  A_{t:t+1}=I_N.
  $$
- `[DERIVATION]` Repeated substitution yields

  $$
  H_t=\sum_{s=1}^{t}A_{t:s+1}B_sx_s^\top,
  $$

  and therefore

  $$
  y_t
  =\sum_{s=1}^{t}\left(C_t^\top A_{t:s+1}B_s\right)x_s.
  $$
- `[DIRECT]` Hence $Y=XM$ for the causal matrix

  $$
  M_{st}=
  \begin{cases}
  C_t^\top A_{t:s+1}B_s,&s\le t,\\
  0,&s>t.
  \end{cases}
  $$
- ### 3.3 Why semiseparable structure appears
- `[DIRECT]` A causal matrix is $N$-semiseparable when every rectangular block lying on or above the diagonal has rank at most $N$.
- `[DERIVATION]` Choose a block with source indices $s$ to the left of target indices $t$, and split every transition at a common boundary $a$:

  $$
  M_{st}
  =\left(A_{a:s+1}B_s\right)^\top
  \left(A_{t:a+1}^\top C_t\right).
  $$
- `[DERIVATION]` Stack the left factors into $L\in\mathbb{R}^{|I|\times N}$ and the right factors into $R\in\mathbb{R}^{N\times|J|}$. The entire block factors as

  $$
  M[I,J]=LR,
  $$

  so its rank is at most $N$.
- `[DIRECT]` The lecture also proves the converse: every $N$-semiseparable upper-triangular matrix admits such sequential generators $(A,B,C)$. The recurrent representation and the structured-matrix rank condition therefore describe the same class.
- `[CHECK]` Reconstruct the converse proof from the nested upper-right blocks in Appendix A.1 before treating this equivalence as internalized.
- ### 3.4 Two computation modes
- `[DIRECT]` The recurrent form performs a left-to-right scan and is natural for generation. The quadratic form materializes

  $$
  Y=X\operatorname{SSS}(A,B,C),
  $$

  exposing a structured causal matrix multiplication that may be accelerator-friendly during training.
- `[DIRECT]` A compressed block representation multiplies an $N$-semiseparable $T\times T$ matrix by a vector in $O(NT)$ time with $O(NT)$ stored scalars.
- `[SYNTHESIS]` Recurrence and matrix multiplication are not two different models. They are two algorithms for the same structured transformation, with different hardware tradeoffs.
- ## 4. Mamba-2 and structured state space duality
- ### 4.1 Scalar-identity transitions
- `[DIRECT]` Mamba-2's SSD specialization uses

  $$
  A_t=a_tI_N.
  $$
- `[DERIVATION]` The transition product becomes

  $$
  A_{t:s+1}
  =\left(\prod_{r=s+1}^{t}a_r\right)I_N.
  $$
- `[DIRECT]` Define the causal persistence mask

  $$
  L_{st}=
  \begin{cases}
  \prod_{r=s+1}^{t}a_r,&s\le t,\\
  0,&s>t.
  \end{cases}
  $$
- `[DERIVATION]` With $G_{st}=C_t^\top B_s$, the induced sequence matrix is

  $$
  M=L\odot G,
  \qquad
  Y=X(L\odot G).
  $$
- `[SYNTHESIS]` This looks like masked linear attention: $B_s$ behaves like a key, $C_t$ like a query, $x_s$ like a value, and $L_{st}$ controls how information survives between positions.
- `[CHECK]` If every $a_t=1$, then $L$ becomes the ordinary causal mask. If $|a_t|<1$, older information is multiplicatively attenuated.
- ### 4.2 Block-level picture
- `[DIRECT]` Mamba-2 produces the SSD inputs and parameters in parallel, applies a short causal convolution and nonlinear activation, performs the SSD sequence transformation, gates its output, normalizes, and finally projects back to the residual stream.
- `[SYNTHESIS]` The architectural goal is to keep the recurrent interpretation needed for cheap decoding while arranging training around large matrix multiplications.
- ## 5. Conditional barriers to subquadratic alternatives
- ### 5.1 Orthogonal Vectors and document similarity
- `[DIRECT]` Orthogonal Vectors asks whether Boolean vectors $v_1,\ldots,v_T\in\{0,1\}^{\ell}$ contain distinct $i,j$ with

  $$
  \langle v_i,v_j\rangle=0.
  $$
- `[DIRECT]` Under the Orthogonal Vectors Conjecture, which follows from SETH, there is no truly subquadratic algorithm for all logarithmic dimensions.
- `[DIRECT]` Least Similar Documents minimizes cosine similarity

  $$
  \operatorname{sim}(u,v)=\frac{\langle u,v\rangle}{\|u\|_2\|v\|_2}.
  $$
- `[DERIVATION]` When all input vectors are nonzero,

  $$
  \min_{i\ne j}\operatorname{sim}(v_i,v_j)=0
  \quad\Longleftrightarrow\quad
  \exists i\ne j:\langle v_i,v_j\rangle=0.
  $$
- `[CONCLUSION]` A truly subquadratic exact LSD algorithm in the lecture's logarithmic-dimensional regime would solve Orthogonal Vectors equally fast and contradict OVC.
- ### 5.2 Why one ordinary attention head can detect OV
- `[DIRECT]` Append a zero vector $v_{T+1}=0$ and choose query/key projections so the scaled score equals

  $$
  -C\log T\,\langle v_i,v_j\rangle.
  $$
- `[DERIVATION]` Exponentiation gives weight

  $$
  T^{-C\langle v_i,v_j\rangle}.
  $$

  An orthogonal partner contributes $1$, whereas every nonorthogonal vector contributes at most $T^{-C}$.
- `[DIRECT]` Choose values whose first coordinate is $\|v_j\|_1$ and whose other coordinates vanish. The appended zero vector contributes $1$ to the softmax denominator but $0$ to the numerator.
- `[DERIVATION]` If query $i$ has an orthogonal partner, its first output coordinate satisfies

  $$
  r_i\ge\frac{1}{T+1}.
  $$
- `[DERIVATION]` If no orthogonal pair exists, then

  $$
  r_i\le \ell T^{1-C}.
  $$
- `[CONCLUSION]` A sufficiently large $C$ creates a nonempty decision gap. Exact pairwise attention can therefore express the hard interaction that the conditional lower bound targets.
- `[SYNTHESIS]` This does not say every linear-time alternative is useless. It says that a universally exact replacement for all such pairwise computations should not be expected without additional assumptions or a complexity-theoretic breakthrough.
- ## 6. Minimal takeaways
- Associativity turns kernel attention into a fixed-size summary when its weights factor through a finite feature map.
- Causal linear attention is literally an RNN with state $S_t$.
- SSMs generalize the state update and induce semiseparable causal matrices.
- Mamba-2's scalar transition makes the recurrence resemble linear attention with a learned persistence mask.
- Recurrent and quadratic forms compute the same structured transformation but suit different hardware regimes.
- Conditional lower bounds identify global pairwise problems that exact standard attention can express and generic subquadratic alternatives may not solve.
- ## 7. Questions for manual reconstruction
- [ ] Re-derive every FLOP term in the first table from matrix dimensions.
- [ ] Starting from $Y=V\Phi_K^\top\Phi_Q$, derive both the batch and recurrent algorithms.
- [ ] Unroll the SSM recurrence for $t=3$ before writing the general formula.
- [ ] Prove the low-rank factorization of one off-diagonal semiseparable block.
- [ ] Explain why $A_t=a_tI$ turns transition products into scalar persistence weights.
- [ ] Reconstruct the yes/no output gap in the attention solution to Orthogonal Vectors.
- ## 8. Open questions
- `[CHECK]` Which normalization is required in practical linear attention to prevent state magnitude from growing with sequence length?
- `[CHECK]` Which Mamba-2 quantities are input-dependent, and how does that affect linearity in the sequence $X$ once parameters are conditioned on the input?
- `[HYPOTHESIS]` The semiseparable order $N$ may be viewed as a communication bandwidth across sequence cuts. This analogy should be tested against the exact rank definition.
- ## References
- `[DIRECT]` Noah Golowich. *CS 395T — Lecture 4: Linear Attention and State Space Models.* <https://noahgol.github.io/teaching/cs395t-f26/lecture4.pdf>
- `[DIRECT]` The lecture draws primarily on Katharopoulos et al. for recurrent linear attention, Dao and Gu for structured state space duality and Mamba-2, and Alman and Yu for conditional limitations. These are reading pointers; the papers have not been independently reconstructed here.

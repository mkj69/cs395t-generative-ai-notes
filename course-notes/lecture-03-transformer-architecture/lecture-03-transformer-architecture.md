- ---
  title: "Lecture 03 — Transformer Architecture"
  slug: "lecture-03"
  date: "2026-09-25"
  updated: "2026-09-25"
  status: "draft"
  type: "learning-note"
  course: "CS 395T"
  source: "https://noahgol.github.io/teaching/cs395t-f26/lecture3.pdf"
  summary: "An AI-assisted first reconstruction of the decoder-only Transformer: tokenization, attention, positional information, normalization, MLPs, unembedding, and the full pre-norm residual stack."
  public: true
  tags:
    - transformer architecture
    - attention
    - RoPE
    - normalization
    - decoder-only language models
  related_research_notes: []
  authorship: "ai-assisted"
  ---
- > **AI-assisted first draft.** This note reconstructs the mathematical route of the Lecture 03 PDF and adds step-by-step derivations and small examples. It is not the author's manual reconstruction. Claims tagged `[DIRECT]` come from the lecture; the remaining explanations should be checked, rewritten, or corrected in Logseq during study.
- ## Lecture context
- `[DIRECT]` Source: <https://noahgol.github.io/teaching/cs395t-f26/lecture3.pdf>
- `[DIRECT]` The lecture builds a decoder-only Transformer from seven pieces: tokenization, embedding lookup, attention, positional information, normalization, position-wise MLPs, and unembedding.
- `[SYNTHESIS]` The architecture alternates two operations. Attention moves information **between token positions**; an MLP transforms information **within each position**. Residual connections preserve a running state while normalization controls the scale of each update.
- ### Label key
- `[DIRECT]`: a definition, equation, algorithm, or claim stated in the lecture PDF.
- `[DERIVATION]`: an intermediate mathematical step written out in this note.
- `[SYNTHESIS]`: a connection or interpretation assembled across lecture sections.
- `[EXAMPLE]`: a toy construction added to make an operation concrete.
- `[CHECK]`: something to verify manually before treating the note as complete.
- ## Reading map

  | Stage | Object entering the stage | What changes | Object leaving the stage |
  | --- | --- | --- | --- |
  | Tokenization | text | strings become discrete IDs | $v_1,\ldots,v_T$ |
  | Embedding | token IDs | lookup supplies learned vectors | $X^{(0)}\in\mathbb{R}^{d\times T}$ |
  | Attention | token representations | positions exchange information | context-dependent vectors |
  | MLP | one vector per position | features are transformed locally | updated vectors |
  | Unembedding | final contextual vector | vector becomes vocabulary logits | $z\in\mathbb{R}^{N_V}$ |
  | Softmax | logits | scores become probabilities | $p(\cdot\mid v_{\le t})$ |
- ## Learning goals
- After working through the note manually, I should be able to:
- [ ] explain why BPE keeps open-vocabulary coverage without using a word-sized vocabulary;
- [ ] derive the shape and meaning of every matrix in a single attention head;
- [ ] prove that a causal mask prevents position $i$ from depending on future positions;
- [ ] derive why RoPE makes query-key scores depend on relative displacement;
- [ ] distinguish LayerNorm from RMSNorm and state their invariances carefully;
- [ ] trace one token representation through a complete pre-norm decoder block; and
- [ ] explain why the probability in column $t$ predicts token $v_{t+1}$ rather than $v_t$.
- ## 1. From text to vectors
- ### 1.1 Why subword tokenization is a compromise
- `[DIRECT]` Let $\mathcal V$ be a finite vocabulary. A tokenizer maps a string to a sequence of elements of $\mathcal V$.
- `[SYNTHESIS]` Character tokenization has excellent coverage but creates long sequences. Word tokenization shortens common text but requires a huge vocabulary and still encounters unseen words. Subword tokenization trades between those extremes: frequent strings can become single tokens, while rare strings can be assembled from smaller units.
- ### 1.2 Byte-pair encoding
- `[DIRECT]` The lecture's BPE procedure starts with words represented as byte sequences. At each training round it counts adjacent pairs without crossing word boundaries, selects a most frequent pair, creates a new symbol for that pair, and replaces nonoverlapping occurrences from left to right. The ordered merge list is the learned tokenizer rule set.
- `[DERIVATION]` The order of the rules matters. A later rule may refer to a symbol that did not exist before an earlier merge. Therefore the learned object is not merely a set of pairs; it is an ordered program

  $$
  \mathcal R=(r_1,r_2,\ldots,r_M).
  $$
- `[EXAMPLE]` Consider a tiny corpus with the words `low`, `lower`, and `lowest`, initially split into bytes. If `(l,o)` is the most frequent pair, the first rule creates `lo`. A later rule can merge `(lo,w)` into `low`. The base bytes remain available, so a string that never appeared during training is still representable.
- `[CHECK]` Run two BPE merge rounds by hand on a corpus with repeated words. Record pair counts before and after each replacement, and verify that overlapping occurrences are not both merged in one left-to-right pass.
- ### 1.3 Embedding lookup
- `[DIRECT]` Let $N_V=|\mathcal V|$, identify tokens with $[N_V]$, and let $\delta_v\in\mathbb{R}^{N_V}$ be the one-hot vector for token $v$. The learned embedding matrix is

  $$
  W_E\in\mathbb{R}^{d\times N_V}.
  $$
- `[DIRECT]` Token $v$ is represented by

  $$
  e(v)=W_E\delta_v=W_E[:,v]\in\mathbb{R}^{d}.
  $$
- `[DERIVATION]` Multiplying by a one-hot vector selects a column because

  $$
  W_E\delta_v
  =\sum_{u=1}^{N_V}\delta_v[u]W_E[:,u]
  =W_E[:,v].
  $$
- `[SYNTHESIS]` The token ID contains no geometry by itself. Similarity between embedding vectors is learned jointly with the rest of the language model because the training loss updates $W_E$ through every downstream layer.
- ## 2. Attention as content-dependent information routing
- ### 2.1 Single-head attention and its shapes
- `[DIRECT]` Put the primary sequence in columns of

  $$
  X=[x_1,\ldots,x_{T_x}]\in\mathbb{R}^{d_x\times T_x}
  $$

  and the context sequence in columns of

  $$
  Z=[z_1,\ldots,z_{T_z}]\in\mathbb{R}^{d_z\times T_z}.
  $$
- `[DIRECT]` A head has learned matrices

  $$
  W_Q\in\mathbb{R}^{d_k\times d_x},\qquad
  W_K\in\mathbb{R}^{d_k\times d_z},\qquad
  W_V\in\mathbb{R}^{d_v\times d_z}.
  $$
- `[DERIVATION · Step 1]` Project inputs into queries, keys, and values:

  $$
  Q=W_QX\in\mathbb{R}^{d_k\times T_x},\qquad
  K=W_KZ\in\mathbb{R}^{d_k\times T_z},\qquad
  V=W_VZ\in\mathbb{R}^{d_v\times T_z}.
  $$
- `[DERIVATION · Step 2]` Every key is compared with every query:

  $$
  S=\frac{K^\top Q}{\sqrt{d_k}}\in\mathbb{R}^{T_z\times T_x}.
  $$
- `[EXPLANATION]` Entry $S_{ji}=k_j^\top q_i/\sqrt{d_k}$ measures how strongly query position $i$ matches context position $j$. Columns correspond to queries; rows correspond to candidate context positions.
- `[DERIVATION · Step 3]` Apply the mask by replacing forbidden scores with $-\infty$, then normalize each column:

  $$
  A_{ji}
  =\frac{\exp(S_{ji})}{\sum_{u=1}^{T_z}\exp(S_{ui})},
  \qquad
  A\in\mathbb{R}^{T_z\times T_x}.
  $$
- `[DERIVATION · Step 4]` Mix the values:

  $$
  Y=VA\in\mathbb{R}^{d_v\times T_x},
  \qquad
  y_i=\sum_{j=1}^{T_z}A_{ji}v_j.
  $$
- `[SYNTHESIS]` The keys decide **where to read**, the queries express **what a position is looking for**, and the values determine **what information is returned**.
- ### 2.2 Why the softmax output is a weighted average
- `[DERIVATION]` For a fixed query $i$, exponentials are nonnegative and the denominator is their sum. Therefore

  $$
  A_{ji}\ge 0,
  \qquad
  \sum_{j=1}^{T_z}A_{ji}=1.
  $$
- `[DERIVATION]` Hence $y_i$ lies in the convex hull of the value vectors before the output projection. A head cannot directly return an arbitrary linear combination with negative attention coefficients, although the value and output projections can still introduce signed features.
- `[DIRECT]` The factor $1/\sqrt{d_k}$ controls the scale of dot products when query and key coordinates have roughly unit variance.
- `[DERIVATION]` If coordinates are independent, mean zero, and unit variance, then

  $$
  \operatorname{Var}(q_i^\top k_j)
  =\operatorname{Var}\!\left(\sum_{r=1}^{d_k}q_{ri}k_{rj}\right)
  \approx d_k.
  $$

  Dividing by $\sqrt{d_k}$ brings the variance back to order one, which helps prevent the softmax from becoming extremely sharp solely because the head dimension is large.
- ### 2.3 A three-position masked example
- `[EXAMPLE]` At query position $i=2$, suppose the unmasked scores for three context positions are

  $$
  (S_{1,2},S_{2,2},S_{3,2})=(0,\log 2,5).
  $$
- `[DERIVATION]` A causal mask forbids $j=3$, so the effective scores become $(0,\log 2,-\infty)$. Their exponentials are $(1,2,0)$, giving

  $$
  (A_{1,2},A_{2,2},A_{3,2})
  =\left(\frac13,\frac23,0\right).
  $$
- `[CONCLUSION]` Even though the future token originally had the largest score, masking forces its contribution to zero:

  $$
  y_2=\frac13v_1+\frac23v_2.
  $$
- ### 2.4 Causal self-attention
- `[DIRECT]` Self-attention sets $Z=X$. Decoder-only language modeling uses the causal mask

  $$
  M_{ji}=\mathbf{1}\{j\le i\}.
  $$
- `[DERIVATION · Base layer]` At position $i$, masked attention can use only value vectors at positions $1,\ldots,i$. A position-wise MLP changes the vector at $i$ but cannot introduce information from another position.
- `[DERIVATION · Induction step]` Assume every representation at layer $r-1$ and position $j$ depends only on input tokens $v_1,\ldots,v_j$. At layer $r$, position $i$ attends only to positions $j\le i$. Each of those representations depends only on $v_1,\ldots,v_j\subseteq v_1,\ldots,v_i$. Therefore the new position-$i$ representation also depends only on the prefix $v_{\le i}$.
- `[CONCLUSION]` By induction over layers, the final vector at position $i$ cannot contain information from $v_{i+1},\ldots,v_T$. It can therefore parameterize the next-token distribution for $v_{i+1}$ without leaking the target.
- ### 2.5 Multi-head attention
- `[DIRECT]` Head $h\in[H]$ has its own $(W_Q^{(h)},W_K^{(h)},W_V^{(h)})$ and produces $Y^{(h)}\in\mathbb{R}^{d_v\times T_x}$. Concatenate the head outputs vertically and mix them with

  $$
  W_O\in\mathbb{R}^{d_{\mathrm{out}}\times Hd_v}.
  $$
- `[DIRECT]` The multi-head output is

  $$
  \operatorname{MHA}(X,Z;M)
  =W_O
  \begin{bmatrix}
  Y^{(1)}\\
  \vdots\\
  Y^{(H)}
  \end{bmatrix}.
  $$
- `[DIRECT]` In the common balanced configuration, $d_k=d_v=d/H$. Concatenating $H$ heads then restores total width $d$ before $W_O$ mixes the head features.
- `[SYNTHESIS]` Multiple heads do not merely repeat one calculation. They provide multiple attention distributions and multiple value projections, so different information-routing patterns can coexist at one layer.
- ## 3. Position must enter somewhere
- ### 3.1 Why plain attention does not know order
- `[DIRECT]` Attention without positional information is permutation equivariant: permuting input columns produces the same permutation of output columns.
- `[DERIVATION]` Let $\Pi$ be a permutation matrix and set $X'=X\Pi$. In self-attention,

  $$
  Q'=Q\Pi,\qquad K'=K\Pi,\qquad V'=V\Pi.
  $$
- `[DERIVATION]` The score matrix becomes

  $$
  S'=(K\Pi)^\top(Q\Pi)=\Pi^\top S\Pi.
  $$
- `[DERIVATION]` Column-wise softmax respects this simultaneous row-column permutation, so $A'=\Pi^\top A\Pi$. Therefore

  $$
  Y'=V\Pi A'=VA\Pi=Y\Pi.
  $$
- `[CONCLUSION]` The mechanism can follow content but cannot infer which token was first or how far apart two tokens were. Positional information must break this symmetry.
- ### 3.2 Absolute sinusoidal embeddings
- `[DIRECT]` The original Transformer adds a fixed position vector $p_t\in\mathbb{R}^d$ to the token embedding. Using zero-based frequency index $r=0,\ldots,d/2-1$,

  $$
  p_t[2r+1]=\sin\!\left(\frac{t}{10000^{2r/d}}\right),
  \qquad
  p_t[2r+2]=\cos\!\left(\frac{t}{10000^{2r/d}}\right).
  $$
- `[DIRECT]` The initial representation becomes

  $$
  x_t^{(0)}=W_E\delta_{v_t}+p_t.
  $$
- `[SYNTHESIS]` Each sine-cosine pair acts like a clock running at a different frequency. The collection gives the model several spatial scales on which to compare positions.
- ### 3.3 Rotary position embeddings
- `[DIRECT]` For a two-dimensional coordinate pair, define

  $$
  R(\phi)=
  \begin{pmatrix}
  \cos\phi&-\sin\phi\\
  \sin\phi&\cos\phi
  \end{pmatrix}.
  $$
- `[DIRECT]` If the even head dimension is $d_k$, let $\theta_r=10000^{-2r/d_k}$ and construct the block-diagonal rotation

  $$
  R_t=\operatorname{diag}\!\left(R(t\theta_0),\ldots,R(t\theta_{d_k/2-1})\right).
  $$
- `[DIRECT]` Given content-only vectors $\bar q_i=W_Qx_i$ and $\bar k_j=W_Kx_j$, RoPE uses

  $$
  q_i=R_i\bar q_i,
  \qquad
  k_j=R_j\bar k_j.
  $$
- #### Relative-position derivation
- `[DERIVATION · Step 1]` A two-dimensional rotation is orthogonal, so

  $$
  R(a)^\top=R(-a).
  $$
- `[DERIVATION · Step 2]` Rotations add their angles:

  $$
  R(-a)R(b)=R(b-a).
  $$
- `[DERIVATION · Step 3]` Apply this independently in every RoPE block:

  $$
  R_i^\top R_j=R_{j-i}.
  $$
- `[DERIVATION · Step 4]` The attention dot product is therefore

  $$
  q_i^\top k_j
  =(R_i\bar q_i)^\top(R_j\bar k_j)
  =\bar q_i^\top R_{j-i}\bar k_j.
  $$
- `[CONCLUSION]` Absolute indices $i$ and $j$ enter the rotations, but their contribution to the query-key score appears through the relative displacement $j-i$.
- `[CHECK]` Verify $R(a)^\top R(b)=R(b-a)$ by multiplying the two $2\times2$ matrices and applying the sine and cosine angle-difference identities.
- ## 4. Transforming each position
- ### 4.1 LayerNorm
- `[DIRECT]` For $x\in\mathbb{R}^d$, define the feature mean and variance

  $$
  \mu(x)=\frac1d\sum_{r=1}^d x_r,
  \qquad
  \sigma^2(x)=\frac1d\sum_{r=1}^d(x_r-\mu(x))^2.
  $$
- `[DIRECT]` With learned scale $\gamma\in\mathbb{R}^d$, bias $\beta\in\mathbb{R}^d$, and numerical stabilizer $\epsilon_{\mathrm{norm}}>0$,

  $$
  \operatorname{LN}_{\gamma,\beta}(x)
  =\gamma\odot
  \frac{x-\mu(x)\mathbf 1_d}{\sqrt{\sigma^2(x)+\epsilon_{\mathrm{norm}}}}
  +\beta.
  $$
- `[SYNTHESIS]` LayerNorm removes the shared feature offset, normalizes feature scale, and then restores learnable coordinate-wise scale and shift through $(\gamma,\beta)$.
- ### 4.2 RMSNorm
- `[DIRECT]` RMSNorm omits mean subtraction:

  $$
  \operatorname{RMSNorm}_\gamma(x)
  =\gamma\odot
  \frac{x}{\sqrt{d^{-1}\sum_{r=1}^d x_r^2+\epsilon_{\mathrm{norm}}}}.
  $$
- `[DERIVATION]` Ignoring $\epsilon_{\mathrm{norm}}$ for the moment and taking $a>0$,

  $$
  \frac{ax}{\sqrt{d^{-1}\sum_r (ax_r)^2}}
  =\frac{ax}{a\sqrt{d^{-1}\sum_r x_r^2}}
  =\frac{x}{\sqrt{d^{-1}\sum_r x_r^2}}.
  $$
- `[CHECK]` The stabilizer makes the invariance approximate rather than exact when the input norm is comparable to $\sqrt{\epsilon_{\mathrm{norm}}}$. Also, unlike LayerNorm, RMSNorm is not invariant to adding the same constant to every feature.
- ### 4.3 Position-wise MLPs
- `[DIRECT]` The original Transformer applies the same two-layer MLP independently at every position:

  $$
  \operatorname{MLP}_{\mathrm{ReLU}}(x)=W_2\operatorname{ReLU}(W_1x),
  $$
- `[EXPLANATION]` Here $W_1\in\mathbb{R}^{d_{\mathrm{ff}}\times d}$ expands into the feed-forward width, and $W_2\in\mathbb{R}^{d\times d_{\mathrm{ff}}}$ returns to model dimension.
- `[DIRECT]` The lecture also gives the gated SwiGLU form. With $\operatorname{SiLU}(a)=a/(1+e^{-a})$ applied coordinate-wise,

  $$
  \operatorname{MLP}_{\mathrm{SwiGLU}}(x)
  =W_2\left(\operatorname{SiLU}(W_gx)\odot W_ux\right).
  $$
- `[SYNTHESIS]` $W_gx$ produces a nonlinear gate, while $W_ux$ produces candidate features. Their element-wise product lets one branch modulate the other before $W_2$ returns to model dimension.
- `[SYNTHESIS]` Attention is the cross-position operation; the MLP is the per-position operation. Repeating both lets the model alternate routing information and computing with it.
- ## 5. Returning to tokens
- ### 5.1 Unembedding
- `[DIRECT]` The unembedding matrix

  $$
  W_U\in\mathbb{R}^{N_V\times d}
  $$

  maps a contextual representation $x\in\mathbb{R}^d$ to one logit per vocabulary item:

  $$
  z=W_Ux,
  \qquad
  p=\operatorname{softmax}(z).
  $$
- `[DIRECT]` The model may learn $W_U$ independently or tie it to the embedding matrix by setting $W_U=W_E^\top$.
- `[SYNTHESIS]` Embedding and unembedding face opposite directions: $W_E$ maps a discrete vocabulary coordinate into model space, while $W_U$ compares a model-space state with vocabulary directions to produce token scores.
- ## 6. The complete decoder-only Transformer
- ### 6.1 One pre-norm residual block
- `[DIRECT]` Let $X^{(0)}\in\mathbb{R}^{d\times T}$ contain the initial token representations. For layer $r\in[L]$, the attention update is

  $$
  U^{(r)}
  =X^{(r-1)}
  +\operatorname{MHA}_r\!\left(
  \operatorname{Norm}^{(r)}_{\mathrm{attn}}(X^{(r-1)});M_{\mathrm{causal}}
  \right).
  $$
- `[DIRECT]` The MLP update is

  $$
  X^{(r)}
  =U^{(r)}
  +\operatorname{MLP}_r\!\left(
  \operatorname{Norm}^{(r)}_{\mathrm{mlp}}(U^{(r)})
  \right).
  $$
- `[DERIVATION]` These are **pre-norm** equations because normalization occurs inside each residual branch, before attention or the MLP. The residual stream itself passes forward through the identity paths $X^{(r-1)}\to U^{(r)}$ and $U^{(r)}\to X^{(r)}$.
- `[SYNTHESIS]` One block can be read as:
  1. normalize the current state;
  2. gather prefix information with attention and add it to the state;
  3. normalize the result;
  4. transform features locally with the MLP and add that update.
- ### 6.2 Final prediction
- `[DIRECT]` After $L$ layers,

  $$
  H=\operatorname{Norm}_{\mathrm{final}}(X^{(L)}),
  \qquad
  Z=W_UH,
  \qquad
  P=\operatorname{softmax}(Z),
  $$
- `[EXPLANATION]` Softmax is applied independently to each column of $Z$.
- `[DERIVATION]` Column $H[:,t]$ depends only on $v_1,\ldots,v_t$ by the causal-dependence proof above. Consequently,

  $$
  P[:,t]
  =p_\theta(\,\cdot\mid v_1,\ldots,v_t)
  $$
- `[CONCLUSION]` This column is aligned with the target token $v_{t+1}$.
- `[SYNTHESIS]` Training can score all next-token targets in parallel because the triangular causal mask creates the correct prefix restriction simultaneously at every column. Generation remains sequential because the input token at the next position does not exist until it has been sampled or selected.
- ### 6.3 End-to-end shape ledger

  | Symbol | Shape | Meaning |
  | --- | --- | --- |
  | $v_{1:T}$ | $T$ discrete IDs | tokenized sequence |
  | $X^{(r)}$ | $d\times T$ | residual stream after layer $r$ |
  | $Q,K$ | $d_k\times T$ | per-head queries and keys |
  | $V$ | $d_v\times T$ | per-head values |
  | $S,A$ | $T\times T$ | attention scores and weights |
  | $Y^{(h)}$ | $d_v\times T$ | output of head $h$ |
  | $H$ | $d\times T$ | final normalized states |
  | $Z$ | $N_V\times T$ | vocabulary logits |
  | $P$ | $N_V\times T$ | next-token distributions |
- ## 7. Minimal takeaways
- Tokenization chooses the discrete units on which the model operates; embedding turns those IDs into learned vectors.
- Attention performs content-dependent routing across positions, while an MLP transforms each position independently.
- The causal mask is a structural guarantee: output position $i$ can depend only on the prefix through $i$.
- Positional information is necessary because content-only attention is permutation equivariant.
- RoPE places relative displacement inside the query-key dot product through the identity $R_i^\top R_j=R_{j-i}$.
- Normalization controls feature scale; residual connections maintain a persistent stream to which each sublayer adds an update.
- Unembedding and softmax turn the final contextual state at position $t$ into a distribution for token $t+1$.
- ## 8. Questions for manual reconstruction
- [ ] Starting only from $Q=W_QX$, $K=W_KZ$, and $V=W_VZ$, recover the shape of $S$, $A$, and $Y$ without looking above.
- [ ] Explain why the softmax is column-wise under the lecture's column-token convention.
- [ ] Prove the causal-dependence claim by induction over layers in my own words.
- [ ] Multiply two rotation matrices to recover $R(a)^\top R(b)=R(b-a)$.
- [ ] Explain exactly what LayerNorm removes that RMSNorm keeps.
- [ ] Draw the two residual paths in one pre-norm block and label every normalization point.
- [ ] Explain why training is parallel across positions but autoregressive generation is not.
- ## 9. Open questions and research parking lot
- `[CHECK]` How do implementation conventions change when tokens are stored as rows rather than columns? Rewrite every matrix product in batch-major notation.
- `[CHECK]` How does key-value caching change the computation performed during autoregressive generation without changing the model distribution?
- `[HYPOTHESIS]` RoPE extrapolation failures may be easier to understand by inspecting how unseen relative displacements change rotation phases at different frequencies. This is only a research direction, not a conclusion from the lecture.
- `[HYPOTHESIS]` The attention/MLP alternation resembles a separation between communication and local computation. A later note could test where this analogy is mathematically useful and where it breaks.
- ## References
- `[DIRECT]` Noah Golowich. *CS 395T: Foundations of Modern Generative AI — Lecture 3: Transformer Architecture.* Fall 2026. <https://noahgol.github.io/teaching/cs395t-f26/lecture3.pdf>
- `[DIRECT]` The lecture attributes its formal Transformer presentation to Mary Phuong and Marcus Hutter, *Formal Algorithms for Transformers*, arXiv:2207.09238. Listed here from the lecture bibliography; not yet independently reconstructed in this note.
- `[DIRECT]` Other works listed in the lecture bibliography include BPE by Sennrich, Haddow, and Birch; the original Transformer by Vaswani et al.; RoPE by Su et al.; LayerNorm by Ba, Kiros, and Hinton; RMSNorm by Zhang and Sennrich; and SwiGLU by Shazeer. These papers are reading pointers, not claims that they were independently read for this first draft.

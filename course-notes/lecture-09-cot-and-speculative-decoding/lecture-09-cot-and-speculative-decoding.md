- ---
  title: "Lecture 09 — Chain-of-Thought and Speculative Decoding"
  slug: "lecture-09"
  date: "2026-09-23"
  updated: "2026-09-24"
  status: "draft"
  type: "learning-note"
  course: "CS 395T"
  source: "https://noahgol.github.io/teaching/cs395t-f26/lecture9.pdf"
  summary: "A detailed guided reconstruction of chain-of-thought, STaR, function-composition lower bounds, and speculative decoding, with proofs explained step by step."
  public: true
  tags:
    - chain-of-thought
    - STaR
    - function composition
    - speculative decoding
  related_research_notes: []
  authorship: "ai-assisted"
  ---
- > **AI-assisted guided reconstruction.** This note follows the mathematical route of the Lecture 09 PDF but rewrites the exposition and explains the proofs step by step. It is not the author's manual reconstruction; it should be checked, revised, and extended in Logseq during learning.
- ## Lecture context
- `[SOURCE]`：<https://noahgol.github.io/teaching/cs395t-f26/lecture9.pdf>
- How chain-of-thought supports sequential computation and how sepculative decoding speeds up generation?
- `[SYNTHESIS]` The two halves study different uses of inference-time computation. Chain-of-thought adds sequential state to expand what a model can compute. Speculative decoding reorganizes generation so expensive target-model work is used more efficiently.
- ## Reading map

  | Part | Central question | Main idea |
  | --- | --- | --- |
  | CoT prompting | Why can intermediate tokens help? | A generated rationale becomes context for later predictions. |
  | STaR | How can a model learn rationales from answer supervision? | Successful generated rationales become a fine-tuning set. |
  | Function composition | What is difficult for one attention layer? | Dependent lookups create an information bottleneck. |
  | Speculative decoding | How can generation become faster without changing the target distribution? | Draft cheaply, verify in parallel, and correct exactly. |
- ## 1. Chain-of-thought as sequential computation
- ### 1.1 Prompting with intermediate steps
- `[SOURCE]` Let $x$ be a question, $r$ a generated rationale, and $y$ the answer. The joint continuation factorizes as $p_\theta(r,y\mid x)=p_\theta(r\mid x)\,p_\theta(y\mid x,r)$.
- `[EXPLANATION]` Each factor is itself a product of next-token probabilities. The model first assigns probability to every token of $r$, then assigns probability to every answer token conditioned on both $x$ and the generated $r$.
- `[SOURCE]` Few-shot CoT prompting places several question-rationale-answer demonstrations before a new question. The model parameters remain fixed; only the context changes. Standard few-shot prompting typically supplies question-answer pairs without the intermediate rationale.
- `[INTUITION]` The rationale is not merely an explanation attached after the answer. Once emitted, it becomes part of the context, so later tokens can reuse intermediate values produced earlier.
- `[SYNTHESIS]` The autoregressive sequence acts like a small external state: write an intermediate result, attend to it, and continue the computation with the same model.
- `[LIMIT]` More tokens do not guarantee valid reasoning. A fluent rationale can contain a wrong step, and a correct final answer does not certify the path that produced it.
- #### A small computation example
- `[EXAMPLE]` Suppose a problem requires computing $(8+4)/3$ and then subtracting $1$. A direct answer token must somehow encode the whole composition at once. A rationale can instead write $8+4=12$, place $12$ in the context, then compute $12/3=4$, and finally output $4-1=3$.
- `[EXPLANATION]` The important point is not the arithmetic difficulty. The example isolates a dependency: the second operation needs the value created by the first. Autoregressive generation turns that newly created value into an input for the next step.
- ### 1.2 STaR: learning from successful rationales
- `[SOURCE]` STaR begins with questions and known answers, plus a small set of rationale demonstrations. One round follows three steps:
- Generate a rationale and an answer for each question.
- Retain a generated sequence when its final answer matches the known answer.
- Fine-tune the original pretrained model on the retained rationale-answer sequences, then repeat generation.
- `[INTUITION]` The model manufactures candidate reasoning traces, while the answer key supplies a cheap success signal. This turns answer-only supervision into a filtered rationale dataset.
- `[SOURCE]` Rationalization covers some initially failed questions: reveal the correct answer as a hint, ask the model for a compatible rationale, then remove the hint from the fine-tuning input.
- `[LIMIT]` The filter checks only the final answer. It can preserve a lucky or internally inconsistent rationale. Rationalized samples are also generated under an answer hint, not under the unhinted policy used at test time.
- #### Rationalization, step by step
- `[SOURCE]` If the first generation for $x_i$ ends with the wrong answer, ordinary filtering produces no example for that question.
- `[SOURCE]` STaR then appends the known answer $y_i$ as a hint and asks the model to generate another rationale-answer sequence.
- `[SOURCE]` If the hinted generation now ends at $y_i$, its rationale $r_i^{\mathrm{rat}}$ is retained.
- `[EXPLANATION]` During fine-tuning, the answer hint is removed from the input. The training pair is therefore question $x_i \to$ rationale $r_i^{\mathrm{rat}}$ followed by answer $y_i$.
- `[LIMIT]` The generated rationale depended on information that will not be available at test time. Rationalization can expand coverage, but it introduces a distribution mismatch that the clean policy-gradient story does not include.
- #### Policy-gradient view
- `[SOURCE]` Treat the generated rationale and answer as a sampled action sequence and reward exact answer correctness. For examples $(x_i,y_i)$, write $J(\theta)=\sum_i\mathbb{E}[\mathbf{1}\{\hat y_i=y_i\}]$.
- `[DERIVATION · Step 1]` For one training question, expand the expectation over every possible generated rationale $r$ and answer $y$:

  $$
  J_i(\theta)=\sum_{r,y}p_\theta(r,y\mid x_i)\,\mathbf{1}\{y=y_i\}.
  $$
- `[EXPLANATION]` The indicator is the reward. It is one only when the sampled final answer equals the label.
- `[DERIVATION · Step 2]` Differentiate term by term:

  $$
  \nabla_\theta J_i(\theta)
  =\sum_{r,y}\mathbf{1}\{y=y_i\}\,\nabla_\theta p_\theta(r,y\mid x_i).
  $$
- `[DERIVATION · Step 3]` Use the log-derivative identity $\nabla_\theta p_\theta=p_\theta\nabla_\theta\log p_\theta$:

  $$
  \nabla_\theta J_i(\theta)
  =\sum_{r,y}p_\theta(r,y\mid x_i)\,\mathbf{1}\{y=y_i\}
  \nabla_\theta\log p_\theta(r,y\mid x_i).
  $$
- `[DERIVATION · Step 4]` Recognize the weighted sum as an expectation under the current model:

  $$
  \nabla_\theta J(\theta)
  =\sum_i\mathbb{E}_{(\hat r,\hat y)\sim p_\theta(\cdot\mid x_i)}
  \left[\mathbf{1}\{\hat y=y_i\}\,\nabla_\theta\log p_\theta(\hat r,\hat y\mid x_i)\right].
  $$
- `[EXPLANATION]` A failed sample receives reward zero and contributes no update. A successful sample contributes the log-likelihood gradient of its full rationale and answer. Filtering successful samples and training on them therefore points in the same qualitative direction as this estimator.
- `[LIMIT]` This is an interpretation rather than an exact equivalence. The clean estimator assumes on-policy samples, whereas STaR uses greedy generations and several supervised updates on a retained dataset.
- ### 1.3 Why dependent lookups are difficult in one layer
- `[SOURCE]` Consider arbitrary tables $g,f:[n]\to[n]$ and a query $a$. Computing $f(g(a))$ needs two dependent lookups: first discover $g(a)$, then use that result as the key for $f$.
- `[INTUITION]` In one attention layer, all heads form their queries from the original representations in parallel. A head cannot receive $g(a)$ from another head and then change its own query within that same layer.
- #### Transformer-layer model
- `[SOURCE]` Put the input representations into $X=[x_1,\ldots,x_N]$, with every $x_j\in\mathbb{R}^d$. For attention head $h$, define

  $$
  \begin{aligned}
  q_i^{(h)}&=W_Q^{(h)}x_i,\\
  k_j^{(h)}&=W_K^{(h)}x_j,\\
  v_j^{(h)}&=W_V^{(h)}x_j.
  \end{aligned}
  $$
- `[SOURCE]` At output position $i$, head $h$ forms

  $$
  \begin{aligned}
  y_i^{(h)}&=\sum_j A_{ji}^{(h)}v_j^{(h)},\\[4pt]
  A_{ji}^{(h)}&=
  \frac{\exp\!\left(\langle q_i^{(h)},k_j^{(h)}\rangle/\sqrt d\right)}
  {\sum_u\exp\!\left(\langle q_i^{(h)},k_u^{(h)}\rangle/\sqrt d\right)}.
  \end{aligned}
  $$
- `[EXPLANATION]` The softmax numerator scores one token against the query. The denominator normalizes across every input position. The head output is therefore a weighted average of value vectors.
- `[SOURCE]` A position-wise map $\Phi$ combines the $H$ head outputs, and a local decoder maps the result to the predicted answer.
- `[SOURCE]` The lower-bound model is generous about local computation: $\Phi$ may include an output projection, MLP, normalization, or even more general position-wise processing. A residual connection can be represented by also giving $\Phi$ the local input.
- `[SOURCE]` Embeddings, position information, and other preprocessing may be placed inside each $x_j$, provided that preprocessing is local to position $j$. What is excluded is the result of an earlier attention layer.
- `[SOURCE]` Every scalar has finite precision $b$ bits. This lets us count how many bits a head can communicate.
- #### Function-composition input
- `[SOURCE]` The prompt uses a fixed layout: some positions encode the table of $f$, some encode $g$, and a query block encodes $a$. The designated output position must return $f(g(a))$.
- `[EXPLANATION]` The layout and model parameters do not adapt to the sampled tables. The $f$ tokens depend only on $f$, the $g$ tokens only on $g$, and the query tokens only on $a$.
- `[EXAMPLE]` If $g(\mathrm{person})$ returns that person's parent and $f(\mathrm{person})$ returns that person's occupation, a query about the occupation of Alice's parent first needs the identity $g(\mathrm{Alice})$ and then the value $f(g(\mathrm{Alice}))$.
- #### Single-layer composition lower bound (Theorem 1.1)
- `[THEOREM]` With $H$ heads, hidden dimension $d$, and $b$ bits per scalar, define $B=H(d+1)b$. When $B<n\log_2 n$, set $R=n\log_2 n-B>0$. For independent uniformly random tables $f$ and $g$, every fixed query $a$ has error probability at least $R/(3n\log_2 n)$. Averaging gives the same bound for an independent uniform query.
- #### Entropy notation needed for the proof
- `[SOURCE]` All logarithms in this lower-bound argument use base two, so entropy is measured in bits.
- `[DEFINITION]` For a discrete random variable $U$, $H(U)=-\sum_u\Pr(U=u)\log\Pr(U=u)$.
- `[DEFINITION]` Conditional mutual information is $I(U;V\mid W)=H(U\mid W)-H(U\mid V,W)$.
- `[DEFINITION]` Binary entropy is $h_2(\delta)=-\delta\log\delta-(1-\delta)\log(1-\delta)$.
- #### Fano's inequality (Lemma 1.2)
- `[THEOREM]` Let $Y$ take values in $[n]$, let $Z$ be an observation, and let $\widehat Y=\psi(Z)$. If $\delta=\Pr(\widehat Y\ne Y)$, then

  $$
  H(Y\mid Z)\le h_2(\delta)+\delta\log(n-1).
  $$
- `[DERIVATION · Step 1]` Define the error flag $E=\mathbf{1}\{\widehat Y\ne Y\}$. Once $(Y,Z)$ is known, $\widehat Y=\psi(Z)$ is known, so $E$ is determined. Therefore $H(E\mid Y,Z)=0$.
- `[DERIVATION · Step 2]` Add $E$ without changing the conditional uncertainty:

  $$
  H(Y\mid Z)=H(E,Y\mid Z).
  $$
- `[EXPLANATION]` The chain rule normally gives $H(E,Y\mid Z)=H(Y\mid Z)+H(E\mid Y,Z)$. The final term is zero by Step 1.
- `[DERIVATION · Step 3]` Apply the chain rule in the other order:

  $$
  H(E,Y\mid Z)=H(E\mid Z)+H(Y\mid E,Z).
  $$
- `[DERIVATION · Step 4]` Conditioning cannot increase entropy, so $H(E\mid Z)\le H(E)=h_2(\delta)$.
- `[DERIVATION · Step 5]` When $E=0$, the observation determines $Y$ because $Y=\widehat Y$. When $E=1$, $Y$ can be any label except $\widehat Y$, leaving at most $n-1$ possibilities. Hence

  $$
  H(Y\mid E,Z)\le\delta\log(n-1).
  $$
- `[CONCLUSION]` Adding the two bounds proves Fano's inequality.
- #### Indexing communication lemma (Lemma 1.3)
- `[SETUP]` Faye knows a uniformly random table $F=(F_1,\ldots,F_n)\in[n]^n$. She sends a $B$-bit message $M$ before Xavier receives an independent uniform index $J$. Xavier must estimate $F_J$ from $(M,J)$.
- `[THEOREM]` If $B<n\log n$ and $R=n\log n-B$, every such protocol has error at least $R/(3n\log n)$.
- `[DERIVATION · Step 1]` Because $J$ is uniform and independent,

  $$
  I(M;F_J\mid J)=\frac1n\sum_{j=1}^n I(M;F_j).
  $$
- `[DERIVATION · Step 2]` The coordinates of $F$ are independent, so $H(F_j)=H(F_j\mid F_{<j})$. Also, adding $F_{<j}$ to the conditioning can only reduce the remaining entropy of $F_j$. Thus

  $$
  I(M;F_j)\le I(M;F_j\mid F_{<j}).
  $$
- `[DERIVATION · Step 3]` Sum the preceding terms and use the mutual-information chain rule:

  $$
  \begin{aligned}
  I(M;F_J\mid J)
  &\le \frac1n\sum_j I(M;F_j\mid F_{<j})\\
  &=\frac1n I(M;F)\\
  &\le \frac{H(M)}n\\
  &\le \frac Bn.
  \end{aligned}
  $$
- `[EXPLANATION]` A $B$-bit message has at most $2^B$ values, so its entropy cannot exceed $B$.
- `[DERIVATION · Step 4]` Before seeing $M$, the requested value is uniform on $[n]$, hence $H(F_J\mid J)=\log n$. After seeing the message,

  $$
  \begin{aligned}
  H(F_J\mid M,J)
  &=H(F_J\mid J)-I(M;F_J\mid J)\\
  &\ge \log n-\frac Bn\\
  &=\frac Rn.
  \end{aligned}
  $$
- `[DERIVATION · Step 5]` Apply Fano to Xavier's estimate:

  $$
  \frac Rn\le h_2(\delta)+\delta\log(n-1).
  $$
- #### Turning Fano's bound into the stated linear bound
- `[CASE 1]` Suppose $\delta\ge 3/(4n)$. Use $h_2(u)\le u\log(e/u)$:

  $$
  \frac Rn\le\delta\log\!\left(\frac{e(n-1)}\delta\right)\le3\delta\log n.
  $$
- `[EXPLANATION]` The last comparison follows from $e(n-1)/\delta\le(4e/3)n(n-1)\le n^3$ for $n\ge2$. Rearranging yields $\delta\ge R/(3n\log n)$.
- `[CASE 2]` Suppose $\delta<3/(4n)$. Each message $m$ determines a full predicted table $\widehat F(m)$ by collecting Xavier's answers for all possible indices. Since there are at most $2^B$ messages, at most $2^B$ of the $n^n$ possible tables can be recovered perfectly.
- `[DERIVATION]` A wrong table differs in at least one coordinate. Averaging over the uniform requested coordinate gives

  $$
  \begin{aligned}
  n\delta
  &\ge \Pr(F\ne\widehat F(M))\\
  &\ge 1-\frac{2^B}{n^n}\\
  &=1-2^{-R}.
  \end{aligned}
  $$
- `[DERIVATION]` The assumption $n\delta<3/4$ forces $R<2$. On $0\le R\le2$, concavity gives $1-2^{-R}\ge3R/8$. Therefore

  $$
  \delta\ge\frac{3R}{8n}\ge\frac{R}{3n\log n}.
  $$
- `[SOURCE]` Extra side information does not help if it is independent of $F$ after conditioning on $J$. Randomized protocols obey the same bound by fixing the random coins, applying the deterministic result, and averaging.
- #### Reducing one attention layer to indexing
- `[DERIVATION · Step 1]` Partition the prompt positions into $I_f$, $I_g$, and $I_a$: the table of $f$, the table of $g$, and the query block. Let $t$ be the output position.
- `[DERIVATION · Step 2]` At position $t$, every query vector $q_t^{(h)}$ depends only on the fixed query template and $a$. It does not depend on the table values or on the unknown intermediate index $g(a)$.
- `[DERIVATION · Step 3]` For head $h$, define unnormalized attention weight $w_j^{(h)}=\exp(\langle q_t^{(h)},k_j^{(h)}\rangle/\sqrt d)$. The holder of $f$ computes

  $$
  \begin{aligned}
  S_f^{(h)}&=\sum_{j\in I_f}w_j^{(h)}v_j^{(h)}\in\mathbb{R}^d,\\
  Z_f^{(h)}&=\sum_{j\in I_f}w_j^{(h)}\in\mathbb{R}.
  \end{aligned}
  $$
- `[EXPLANATION]` $S_f^{(h)}$ is the part of the softmax numerator contributed by $f$ tokens; $Z_f^{(h)}$ is their part of the denominator.
- `[DERIVATION · Step 4]` Sending one $d$-dimensional vector and one scalar for each of $H$ heads costs $H(d+1)b=B$ bits at $b$-bit precision.
- `[DERIVATION · Step 5]` The receiver already knows $g$, $a$, the layout, and the model. It computes corresponding pairs $(S_g,Z_g)$ and $(S_a,Z_a)$ and reconstructs every head exactly:

  $$
  y_t^{(h)}=
  \frac{S_f^{(h)}+S_g^{(h)}+S_a^{(h)}}
       {Z_f^{(h)}+Z_g^{(h)}+Z_a^{(h)}}.
  $$
- `[DERIVATION · Step 6]` The receiver applies the same local map $\Phi$ and answer decoder as the original transformer, so the communication protocol reproduces the layer's prediction.
- `[DERIVATION · Step 7]` For uniform $g$, $J=g(a)$ is uniform and independent of $f$. The rest of $g$ remains independent of $f$ after conditioning on $J$, so the indexing lemma applies with message length $B$.
- `[CONCLUSION]` The layer's error is at least $R/(3n\log n)$. The core obstruction is simultaneity: all heads query the original state, so no head can use another head's newly discovered $g(a)$ to form a second lookup in the same layer.
- #### Iterated composition and CoT (Theorem 1.4)
- `[SOURCE]` For functions $f_1,\ldots,f_K$, set $z_0=a$ and recursively define $z_j=f_j(z_{j-1})$. The desired answer is $z_K$.
- `[SOURCE]` The lecture states, without proving, that sufficiently general iterated composition has a worst-case lower bound of $\Omega\!\left(\sqrt{n/(Hdb)}\right)$ CoT steps in this finite-precision model. The underlying argument uses pointer-chasing communication complexity.
- `[LIMIT]` The number of composed functions is allowed to grow with problem size. The theorem does not assert this many steps for every fixed value of $K$.
- `[DERIVATION]` A constructive upper bound is more direct: write $z_1=f_1(a)$, append it, then compute and append $z_2=f_2(z_1)$, continuing until $z_K$. This uses one sequential lookup per function.
- `[SOURCE]` One way to realize an exact lookup is to encode every table key by a distinct binary sign vector. The matching key has a strictly larger inner product with the query than every nonmatching key.
- `[DERIVATION]` Scale the attention scores until the matching row receives weight greater than $1/2$. If the table values are binary encoded, thresholding each output coordinate at $1/2$ recovers the selected value exactly.
- `[EXPLANATION]` A token that records both the current value and the current stage tells the local map which function table to query next. This makes $K$ sequential steps sufficient under enough precision.
- `[SYNTHESIS]` This result does not claim that every real reasoning task needs a long visible rationale. It supplies a clean case in which sequentially generated state expands the available computation beyond one parallel layer.
- ## 2. Speculative decoding
- ### 2.1 Draft, verify, accept, correct
- `[SOURCE]` Let $p$ be the expensive target model and $q$ a cheaper draft model. From prefix $s$, the draft model proposes $\gamma$ tokens sequentially. The target model then evaluates the corresponding draft prefixes together in one causally masked forward pass.
- `[DEFINITION]` If the draft tokens are $\tilde v_1,\ldots,\tilde v_\gamma$, define

  $$
  \begin{aligned}
  q_i(\cdot)&=q(\cdot\mid s\tilde v_1\cdots\tilde v_{i-1}),\\
  p_i(\cdot)&=p(\cdot\mid s\tilde v_1\cdots\tilde v_{i-1}).
  \end{aligned}
  $$
- `[SOURCE]` The target also evaluates $p_{\gamma+1}$ at the prefix containing the entire draft. A causal mask lets one target forward pass produce all $\gamma+1$ next-token distributions once the proposed block is known.
- #### One complete speculative iteration
- `[ALGORITHM · Draft]` For $i=1,\ldots,\gamma$, sample $\tilde v_i\sim q_i$. These draft steps remain sequential because proposal $i$ conditions on earlier proposals.
- `[ALGORITHM · Verify]` Evaluate $p_1,\ldots,p_{\gamma+1}$ together with the target model.
- `[ALGORITHM · Test]` At position $i$, draw a fresh $U_i\sim\operatorname{Uniform}[0,1]$. Accept $\tilde v_i$ when

  $$
  U_i\le\min\!\left(1,\frac{p_i(\tilde v_i)}{q_i(\tilde v_i)}\right).
  $$
- `[ALGORITHM · Correct]` At the first rejection, form

  $$
  r_i(v)=\frac{[p_i(v)-q_i(v)]_+}{\sum_u[p_i(u)-q_i(u)]_+}.
  $$
- `[ALGORITHM · Return after rejection]` Sample $w\sim r_i$, emit the earlier accepted drafts followed by $w$, and discard every later draft token.
- `[ALGORITHM]` If all $\gamma$ proposals are accepted, sample one bonus token directly from the target distribution. An iteration therefore emits between one and $\gamma+1$ tokens.
- `[INTUITION]` The draft model guesses a short future and the target checks that future in parallel. Agreement creates progress; disagreement invokes an exact correction rather than an approximate fallback.
- ### 2.2 Why the sampler is exact
- #### One-token correctness lemma (Lemma 2.1)
- `[SETUP]` Ignore prefixes temporarily and consider two distributions $p$ and $q$ on the same finite vocabulary. Propose $V\sim q$, accept with probability $\min(1,p(V)/q(V))$, and on rejection draw from the residual distribution.
- `[DERIVATION · Step 1]` The probability that value $v$ is both proposed and accepted is

  $$
  q(v)\min\!\left(1,\frac{p(v)}{q(v)}\right)=\min(p(v),q(v)).
  $$
- `[EXPLANATION]` If $p(v)\ge q(v)$, every proposal of $v$ is accepted and contributes $q(v)$. If $p(v)<q(v)$, only the fraction $p(v)/q(v)$ is accepted and contributes $p(v)$. If $q(v)=0$, $v$ is never proposed, so the ratio is never evaluated on that event.
- `[DERIVATION · Step 2]` Sum the accepted mass:

  $$
  \beta=\sum_v\min(p(v),q(v)).
  $$
- `[DERIVATION · Step 3]` Use $\min(a,b)=(a+b-|a-b|)/2$ and the fact that both distributions sum to one:

  $$
  \begin{aligned}
  \beta&=1-\frac12\sum_v|p(v)-q(v)|\\
  &=1-\operatorname{TV}(p,q).
  \end{aligned}
  $$
- `[EXPLANATION]` The rejection probability is therefore $1-\beta=\operatorname{TV}(p,q)$. A better draft model overlaps more strongly with the target and is rejected less often.
- `[DERIVATION · Step 4]` The total positive difference equals the missing target mass:

  $$
  \sum_v[p(v)-q(v)]_+=1-\beta.
  $$
- `[EXPLANATION]` The positive and negative differences between two normalized distributions have equal total magnitude. The positive part is precisely the mass target $p$ has that was not supplied by accepted proposals.
- `[DERIVATION · Step 5]` Normalize that missing mass:

  $$
  r(v)=\frac{[p(v)-q(v)]_+}{1-\beta}.
  $$
- `[DERIVATION · Step 6]` Add the two mutually exclusive ways to output $v$:

  $$
  \begin{aligned}
  \Pr(\text{output}=v)
  &=\Pr(\text{accept }v)+\Pr(\text{reject})\Pr(\text{correction}=v\mid\text{reject})\\
  &=\min(p(v),q(v))+(1-\beta)r(v)\\
  &=\min(p(v),q(v))+[p(v)-q(v)]_+\\
  &=p(v).
  \end{aligned}
  $$
- `[EDGE CASE]` If $\beta=1$, then $p=q$. Rejection has probability zero, so the residual distribution is never needed. Whenever rejection can occur, $1-\beta>0$, which makes the denominator valid.
- #### Sequence-level correctness (Proposition 2.2)
- `[THEOREM]` Repeated speculative decoding produces exactly the target autoregressive law. For target length $T$, the probability of tokens $(v_1,\ldots,v_T)$ after prompt $s_0$ is

  $$
  \prod_{t=1}^T p(v_t\mid s_0v_1\cdots v_{t-1}).
  $$
- `[DERIVATION · Step 1]` Condition on any emitted prefix and on any internal state compatible with that prefix: which drafts were sampled, which earlier proposals passed, and whether the next position is a correction or bonus position.
- `[DERIVATION · Step 2]` At a reached draft position, the next proposal is still distributed as the local $q$, and its acceptance uniform is fresh. The one-token lemma says the next emitted token has the local target distribution $p$.
- `[DERIVATION · Step 3]` At a bonus position, the algorithm samples directly from the same local target distribution.
- `[DERIVATION · Step 4]` Every compatible internal state therefore gives the same conditional law $p$ for the next emitted token. Averaging over internal states does not change that law.
- `[DERIVATION · Step 5]` Induct over output positions. Each conditional probability is the appropriate target next-token probability; multiplying them gives the target sequence probability.
- `[LIMIT]` Draft tokens after the first rejection must be discarded because their precomputed target distributions condition on the rejected token, which is absent from the corrected output prefix.
- ### 2.3 Expected progress and speedup
- #### Prefix-dependent acceptance
- `[DEFINITION]` At prefix $s$, define the overlap

  $$
  \beta(s)=\sum_v\min\bigl(p(v\mid s),q(v\mid s)\bigr).
  $$
- `[EXPLANATION]` By the one-token lemma, $\beta(s)$ is exactly the probability that a draft proposal at prefix $s$ is accepted.
- `[DEFINITION]` Let $A_i$ be the event that proposal $i$ passes. Let $R_i=A_1\cap\cdots\cap A_{i-1}$ be the event that position $i$ is reached, with $R_1$ always true. Let $S_i$ be the random draft prefix present at that position.
- `[ASSUMPTION]` For the runtime calculation, suppose every reached position has the same conditional mean overlap:

  $$
  \mathbb{E}[\beta(S_i)\mid R_i]=\alpha.
  $$
- `[DERIVATION]` Conditional on $S_i$, the probability of passing is $\beta(S_i)$. Taking the conditional expectation over reached prefixes gives $\Pr(A_i\mid R_i)=\alpha$.
- #### Expected tokens per iteration
- `[DEFINITION]` Let $L$ be the number of consecutive accepted drafts, so $L$ ranges from $0$ to $\gamma$. Let $N=L+1$ be the number of emitted tokens, including either a correction token or the all-accepted bonus token.
- `[DERIVATION · Step 1]` Reaching at least $j$ acceptances means that the first $j$ acceptance events all occurred:

  $$
  \Pr(L\ge j)=\Pr(A_1\cap\cdots\cap A_j).
  $$
- `[DERIVATION · Step 2]` Repeated conditioning uses $\Pr(A_i\mid R_i)=\alpha$ at every reached position:

  $$
  \Pr(L\ge j)=\alpha^j,\qquad 0\le j\le\gamma.
  $$
- `[EXPLANATION]` This does not require the acceptance events to be independent. The constant conditional mean assumption is sufficient.
- `[DERIVATION · Step 3]` For an integer-valued variable $N\in\{1,\ldots,\gamma+1\}$, use the tail-sum identity

  $$
  N=\sum_{k=1}^{\gamma+1}\mathbf{1}\{N\ge k\}.
  $$
- `[DERIVATION · Step 4]` Take expectations and substitute the tail probabilities:

  $$
  \begin{aligned}
  \mathbb{E}[N]
  &=\sum_{j=0}^{\gamma}\Pr(L\ge j)\\
  &=\sum_{j=0}^{\gamma}\alpha^j\\
  &=\frac{1-\alpha^{\gamma+1}}{1-\alpha},\qquad \alpha<1.
  \end{aligned}
  $$
- `[EDGE CASE]` If $\alpha=1$, every proposal is accepted and $\mathbb{E}[N]=\gamma+1$.
- #### Cost model
- `[DEFINITION]` Let $T_p$ be the wall-clock time for one ordinary target-model step and $T_q$ the time for one draft-model step. Define relative draft cost $c=T_q/T_p$.
- `[ASSUMPTION]` With sufficient parallel hardware, verifying the whole draft and computing the bonus distribution takes approximately one target-step time $T_p$.
- `[DERIVATION]` The draft still requires $\gamma$ sequential calls. Hence one iteration costs

  $$
  \gamma T_q+T_p=(1+\gamma c)T_p.
  $$
- #### Wall-clock speedup theorem (Theorem 2.3)
- `[DERIVATION · Step 1]` Baseline decoding spends $T_p$ per output token.
- `[DERIVATION · Step 2]` Speculative decoding spends $(1+\gamma c)T_p$ per iteration and emits $\mathbb{E}[N]$ tokens on average, so its long-run time per token is

  $$
  \frac{(1+\gamma c)T_p}{\mathbb{E}[N]}.
  $$
- `[DERIVATION · Step 3]` Divide baseline time per token by speculative time per token:

  $$
  \operatorname{Speedup}
  =\frac{\mathbb{E}[N]}{1+\gamma c}
  =\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(1+\gamma c)}.
  $$
- `[EDGE CASE]` At $\alpha=1$, the expression is understood by continuity and becomes $(\gamma+1)/(1+\gamma c)$.
- `[EXPLANATION]` The theorem compares total time with total progress over many complete iterations. It does not average the random per-iteration ratio $(1+\gamma c)T_p/N$, which would answer a different question.
- `[EXAMPLE]` If $\alpha=0.8$, $\gamma=4$, and $c=0.1$, then

  $$
  \begin{aligned}
  \mathbb{E}[N]&=1+0.8+0.8^2+0.8^3+0.8^4=3.3616,\\
  \operatorname{Speedup}&=\frac{3.3616}{1.4}\approx2.40\times.
  \end{aligned}
  $$
- `[LIMIT]` This is not a hardware-independent promise. The model neglects acceptance-test overhead, memory traffic, batching effects, and a final truncated iteration. Verification performs arithmetic at every drafted position even when wall-clock time is parallelized.
- ## Appendix A. A second proof of the indexing bound
- `[SOURCE]` The lecture includes an alternative argument that replaces Fano's inequality with a bound involving Hamming distance.
- #### Hamming-distance entropy lemma (Lemma A.1)
- `[THEOREM]` Let $U$ be uniform on a nonempty set $S\subseteq[n]^n$. For any fixed vector $z\in[n]^n$,

  $$
  \log|S|\le3\log n\,\mathbb{E}[d_H(U,z)].
  $$
- `[DEFINITION]` $d_H(U,z)$ counts the coordinates where $U$ and $z$ differ. Write $s=|S|$ and $D=\mathbb{E}[d_H(U,z)]$.
- `[CASE]` When $s=1$, the left side is zero, so the inequality follows from $D\ge0$.
- `[CASE]` When $s=2$ or $s=3$, at most one element of $S$ can equal $z$; therefore $D\ge(s-1)/s$. Directly checking $s=2,3$ gives $\log s\le3(s-1)/s\le3D\log n$.
- `[CASE · Step 1]` For $s\ge4$, the same observation gives $D\ge(s-1)/s\ge3/4$.
- `[CASE · Step 2]` Continue with $s\ge4$. Let $e_j=\Pr(U_j\ne z_j)$. To describe coordinate $U_j$, first record whether it equals $z_j$; if not, record one of at most $n-1$ alternatives. Therefore

  $$
  H(U_j)\le h_2(e_j)+e_j\log(n-1).
  $$
- `[CASE · Step 3]` Because $U$ is uniform on $S$, $H(U)=\log s$. Entropy subadditivity and concavity of $h_2$ yield

  $$
  \begin{aligned}
  \log s=H(U)
  &\le\sum_jH(U_j)\\
  &\le n\,h_2(D/n)+D\log(n-1).
  \end{aligned}
  $$
- `[EXPLANATION]` The equality $\sum_j e_j=D$ converts coordinate-wise error probabilities into expected Hamming distance. Jensen's inequality moves their average inside the concave binary entropy function.
- `[CASE · Step 4]` Apply $h_2(u)\le u\log(e/u)$:

  $$
  \log s\le D\log\!\left(\frac{en(n-1)}D\right).
  $$
- `[CASE · Step 5]` Since $D\ge3/4$ and $n\ge2$, the logarithm's argument is at most $n^3$, giving $\log s\le3D\log n$.
- #### Apply the lemma to the message classes
- `[DERIVATION · Step 1]` For a fixed message value $m$, let $S_m$ be the set of tables producing that message. Conditional on $M=m$, the random table $F$ is uniform on $S_m$.
- `[DERIVATION · Step 2]` Let $z_m(j)$ be a most likely value of coordinate $F_j$ under this conditional distribution. This coordinate-wise choice minimizes the probability of error for a uniformly requested index.
- `[DERIVATION · Step 3]` The conditional error is

  $$
  \delta_m=\frac1n\mathbb{E}[d_H(F,z_m)\mid M=m].
  $$
- `[DERIVATION · Step 4]` Apply the Hamming-distance lemma inside each message class:

  $$
  H(F\mid M=m)=\log|S_m|\le3n\log n\,\delta_m.
  $$
- `[DERIVATION · Step 5]` Because $M$ is determined by $F$,

  $$
  \begin{aligned}
  H(F\mid M)&=H(F)-H(M)\\
  &\ge n\log n-B\\
  &=R.
  \end{aligned}
  $$
- `[CONCLUSION]` Average the per-message upper bound and compare it with the lower bound:

  $$
  R\le H(F\mid M)\le3n\log n\,\delta.
  $$
- `[CONCLUSION]` Therefore $\delta\ge R/(3n\log n)$.
- `[SOURCE]` As in the first proof, conditionally independent side information does not change the distribution of $F$ given $(M,J)$, and randomized protocols are handled by fixing and averaging over their coins.
- ## 3. Connecting the two halves
- `[SYNTHESIS]` Chain-of-thought spends sequential tokens to make dependent computation possible. Speculative decoding tries to recover wall-clock efficiency by using a cheap model for proposals and parallel target verification.
- `[SYNTHESIS]` Both methods expose the dependency structure of autoregressive generation. CoT emphasizes that later computation can depend on newly written state; speculative decoding exploits the fact that several conditional distributions can be evaluated together once a candidate prefix is known.
- `[HYPOTHESIS]` Reasoning-heavy outputs may have uneven acceptance rates: routine connective text may be easy for the draft model, while decisive reasoning steps may cause rejection. Measuring acceptance by semantic role could show where speculative decoding saves time on reasoning tasks.
- ## 4. Minimal takeaways
- A rationale can function as sequential working state, not only as an explanation.
- STaR converts final-answer supervision into filtered rationale training, but answer correctness does not validate every reasoning step.
- Function composition isolates a concrete limitation of parallel single-layer lookup.
- Speculative decoding is distributionally exact because rejected probability mass is restored through the residual distribution.
- Speedup depends jointly on draft-target agreement, draft length, draft cost, and the hardware's ability to verify a block in parallel.
- ## 5. Questions for manual reconstruction
- [ ] Derive the STaR score-function gradient without looking at the lecture PDF.
- [ ] Explain why one attention head cannot use another head's newly discovered $g(a)$ in the same layer.
- [ ] Reconstruct the one-token speculative sampling proof from the overlap mass $\min(p,q)$.
- [ ] Re-derive the tail-sum formula for $\mathbb{E}[N]$ and identify every runtime assumption.
- [ ] Test how predicted speedup changes as $\gamma$, $\alpha$, and $c$ vary.
- ## 6. Open questions
- `[QUESTION]` How often do answer-correct STaR rationales contain invalid intermediate claims?
- `[QUESTION]` Which task properties determine whether visible CoT is genuinely computational or merely explanatory?
- `[QUESTION]` How stable is the constant-mean-acceptance approximation across different positions in a reasoning trace?
- `[QUESTION]` Can a draft model be trained specifically for low-entropy reasoning transitions without copying the full target model?
- ## References
- Noah Golowich. “Lecture 9: Chain-of-Thought and Speculative Decoding.” CS 395T, Fall 2026. <https://noahgol.github.io/teaching/cs395t-f26/lecture9.pdf>
- Jason Wei et al. “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.” NeurIPS, 2022. <https://arxiv.org/abs/2201.11903>
- Eric Zelikman et al. “STaR: Bootstrapping Reasoning With Reasoning.” NeurIPS, 2022. <https://arxiv.org/abs/2203.14465>
- Binghui Peng, Srini Narayanan, and Christos Papadimitriou. “On Limitations of the Transformer Architecture.” 2024. <https://arxiv.org/abs/2402.08164>
- Yaniv Leviathan, Matan Kalman, and Yossi Matias. “Fast Inference from Transformers via Speculative Decoding.” ICML, 2023. <https://arxiv.org/abs/2211.17192>
- `[SOURCE NOTE]` The lecture PDF includes an AI-use disclosure and states that the lecturer takes responsibility for its contents.

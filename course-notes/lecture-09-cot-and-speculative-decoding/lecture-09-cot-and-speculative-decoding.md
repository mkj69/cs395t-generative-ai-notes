- ---
  title: "Lecture 09 — Chain-of-Thought and Speculative Decoding"
  slug: "lecture-09"
  date: "2026-09-23"
  updated: "2026-09-24"
  status: "draft"
  type: "learning-note"
  course: "CS 395T"
  source: "https://noahgol.github.io/teaching/cs395t-f26/lecture9.pdf"
  summary: "A first-pass study note on sequential computation through chain-of-thought and distribution-preserving acceleration through speculative decoding."
  public: true
  tags:
    - chain-of-thought
    - STaR
    - function composition
    - speculative decoding
  related_research_notes: []
  authorship: "ai-assisted"
  ---
- > **AI-assisted draft.** This baseline was synthesized from the Lecture 09 PDF so that the main ideas, derivations, and questions are available for study. It is not the author's manual reconstruction; it should be revised, corrected, and extended in Logseq during learning.
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
- `[SOURCE]` Let `x` be a question, `r` a generated rationale, and `y` the answer. The joint continuation factorizes as `p_theta(r, y | x) = p_theta(r | x) p_theta(y | x, r)`.
- `[INTUITION]` The rationale is not merely an explanation attached after the answer. Once emitted, it becomes part of the context, so later tokens can reuse intermediate values produced earlier.
- `[SYNTHESIS]` The autoregressive sequence acts like a small external state: write an intermediate result, attend to it, and continue the computation with the same model.
- `[LIMIT]` More tokens do not guarantee valid reasoning. A fluent rationale can contain a wrong step, and a correct final answer does not certify the path that produced it.
- ### 1.2 STaR: learning from successful rationales
- `[SOURCE]` STaR begins with questions and known answers, plus a small set of rationale demonstrations. One round follows three steps:
- Generate a rationale and an answer for each question.
- Retain a generated sequence when its final answer matches the known answer.
- Fine-tune the original pretrained model on the retained rationale-answer sequences, then repeat generation.
- `[INTUITION]` The model manufactures candidate reasoning traces, while the answer key supplies a cheap success signal. This turns answer-only supervision into a filtered rationale dataset.
- `[SOURCE]` Rationalization covers some initially failed questions: reveal the correct answer as a hint, ask the model for a compatible rationale, then remove the hint from the fine-tuning input.
- `[LIMIT]` The filter checks only the final answer. It can preserve a lucky or internally inconsistent rationale. Rationalized samples are also generated under an answer hint, not under the unhinted policy used at test time.
- #### Policy-gradient view
- `[SOURCE]` Treat the generated rationale and answer as a sampled action sequence and reward exact answer correctness. For examples `(x_i, y_i)`, write `J(theta) = sum_i E[1{y_hat_i = y_i}]`.
- `[DERIVATION]` The score-function identity yields a term of the form `E[1{correct} grad_theta log p_theta(r_hat, y_hat | x)]`. Incorrect samples contribute zero; correct samples increase the likelihood of their complete generated sequence.
- `[LIMIT]` This is an interpretation rather than an exact equivalence. The clean estimator assumes on-policy samples, whereas STaR uses greedy generations and several supervised updates on a retained dataset.
- ### 1.3 Why dependent lookups are difficult in one layer
- `[SOURCE]` Consider arbitrary tables `g, f : [n] -> [n]` and a query `a`. Computing `f(g(a))` needs two dependent lookups: first discover `g(a)`, then use that result as the key for `f`.
- `[INTUITION]` In one attention layer, all heads form their queries from the original representations in parallel. A head cannot receive `g(a)` from another head and then change its own query within that same layer.
- `[SOURCE]` With `H` heads, hidden dimension `d`, and `b` bits per scalar, define `B = H(d + 1)b`. When `B < n log_2 n`, set `R = n log_2 n - B`; for uniformly random tables the error probability is at least `R / (3 n log_2 n)`.
- #### Proof skeleton
- `[DERIVATION]` The lower bound reduces one attention layer to a one-way communication protocol.
- The party holding `f` summarizes its contribution to every attention head using partial attention numerators and denominators.
- This message contains at most `B` bits and is fixed before the requested index `J = g(a)` is known.
- Recovering `f(J)` from that short message is an indexing problem.
- Fano's inequality turns the remaining uncertainty about `f(J)` into a lower bound on prediction error.
- `[INTUITION]` Attention can perform a lookup, but the second lookup depends on the result of the first while a single layer performs its lookups simultaneously.
- #### Iterated composition and CoT
- `[SOURCE]` For `z_0 = a` and `z_j = f_j(z_{j-1})`, each generated CoT token can store the current state `z_j`, allowing the next autoregressive step to perform the next lookup.
- `[SOURCE]` The lecture states a worst-case lower bound of `Omega(sqrt(n / (H d b)))` CoT steps for sufficiently general iterated composition in its finite-precision model. It also explains why `K` explicit sequential lookups suffice for a composition of length `K` under a suitable encoding.
- `[SYNTHESIS]` This result does not claim that every real reasoning task needs a long visible rationale. It supplies a clean case in which sequentially generated state expands the available computation beyond one parallel layer.
- ## 2. Speculative decoding
- ### 2.1 Draft, verify, accept, correct
- `[SOURCE]` Let `p` be the expensive target model and `q` a cheaper draft model. From prefix `s`, the draft model proposes `gamma` tokens sequentially. The target model then evaluates the corresponding draft prefixes together in one causally masked forward pass.
- `[ALGORITHM]` At draft position `i`, accept proposed token `v` with probability `min(1, p_i(v) / q_i(v))`.
- `[ALGORITHM]` On the first rejection, keep the previously accepted tokens, sample a correction from the normalized positive part `[p_i - q_i]_+`, and discard the rest of the draft.
- `[ALGORITHM]` If all `gamma` proposals are accepted, sample one bonus token directly from the target distribution. An iteration therefore emits between one and `gamma + 1` tokens.
- `[INTUITION]` The draft model guesses a short future and the target checks that future in parallel. Agreement creates progress; disagreement invokes an exact correction rather than an approximate fallback.
- ### 2.2 Why the sampler is exact
- `[SOURCE]` For one token, the probability of proposing and accepting value `v` is `q(v) min(1, p(v)/q(v)) = min(p(v), q(v))`.
- `[DERIVATION]` The total acceptance probability is `beta = sum_v min(p(v), q(v)) = 1 - TV(p, q)`, so rejection has probability `1 - beta`.
- `[DERIVATION]` Define the residual distribution as `r(v) = [p(v) - q(v)]_+ / (1 - beta)`. Accepted and corrected mass combine to `min(p(v), q(v)) + (1 - beta)r(v) = p(v)`.
- `[SYNTHESIS]` Exactness holds locally at every reached prefix. Repeating the one-token argument shows that the emitted sequence has the same autoregressive distribution as ordinary sampling from the target model.
- `[LIMIT]` Draft tokens after the first rejection must be discarded because their target probabilities were evaluated under a prefix containing a token that was not emitted.
- ### 2.3 Expected progress and speedup
- `[SOURCE]` Assume the conditional mean acceptance probability at each reached draft position is a constant `alpha`. If `L` is the number of consecutive accepted drafts, then `P(L >= j) = alpha^j` for `0 <= j <= gamma`.
- `[DERIVATION]` With `N = L + 1` emitted tokens per iteration, the tail-sum identity gives `E[N] = sum_{j=0}^{gamma} alpha^j = (1 - alpha^(gamma+1)) / (1 - alpha)` for `alpha < 1`.
- `[SOURCE]` Let a target step cost `T_p`, a draft step cost `T_q`, and `c = T_q / T_p`. Under the parallel verification model, one iteration costs `(1 + gamma c)T_p`.
- `[DERIVATION]` The idealized wall-clock speedup is `E[N] / (1 + gamma c)`.
- `[EXAMPLE]` If `alpha = 0.8`, `gamma = 4`, and `c = 0.1`, then `E[N] = 3.3616` and the predicted speedup is approximately `2.40x`.
- `[LIMIT]` This formula is not a hardware-independent promise. It assumes block verification takes roughly one target-step time and neglects acceptance overhead, memory traffic, batching effects, and the final truncated iteration.
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
- [ ] Explain why one attention head cannot use another head's newly discovered `g(a)` in the same layer.
- [ ] Reconstruct the one-token speculative sampling proof from the overlap mass `min(p, q)`.
- [ ] Re-derive the tail-sum formula for `E[N]` and identify every runtime assumption.
- [ ] Test how predicted speedup changes as `gamma`, `alpha`, and `c` vary.
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

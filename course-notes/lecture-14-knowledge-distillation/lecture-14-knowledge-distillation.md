---
title: "Lecture 14 — Knowledge Distillation for Language Models"
slug: "lecture-14"
date: "2026-10-12"
updated: "2026-10-12"
status: "draft"
type: "learning-note"
course: "CS 395T"
source: "https://noahgol.github.io/teaching/cs395t-f26/"
summary: "A detailed AI-assisted study scaffold for knowledge distillation, from temperature-scaled soft targets to sequence-level and on-policy distillation for autoregressive language models."
public: true
tags:
  - knowledge distillation
  - model compression
  - soft targets
  - temperature scaling
  - sequence-level distillation
  - forward KL
  - reverse KL
  - on-policy distillation
related_research_notes: []
authorship: "ai-assisted"
---

> **AI-assisted study scaffold, not a lecture transcript.** The official course schedule lists Lecture 14 as a student-presentation session on distillation, but it does not publish a unified handout or slide deck for this lecture. This note therefore does not claim to reconstruct what the presenters said. It develops the named topic from the primary papers listed below and marks derivations and synthesis explicitly. It has not yet been manually verified or rewritten by the author.

## Source boundary

`[DIRECT · COURSE SCHEDULE]` The official schedule names the October 12 session **“Post-training III (Student Presentations #2): Distillation.”** It does not link lecture notes or slides for this session: <https://noahgol.github.io/teaching/cs395t-f26/>.

`[DIRECT · PRIMARY SOURCES]` The main technical sources used in this scaffold are:

1. Hinton, Vinyals, and Dean (2015), [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).
2. Kim and Rush (2016), [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947).
3. Gu et al. (2023), [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543).
4. Agarwal et al. (2023), [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649).

`[SOURCE BOUNDARY]` Statements labeled `[DIRECT]` are attributable to these papers or to the course schedule. `[DERIVATION]` lines work through the mathematics. `[EXPLANATION]` and `[SYNTHESIS]` organize the ideas but are not reported classroom claims.

`[SYNTHESIS]` Distillation is best understood as **distribution transfer under constraints**. A teacher defines behavior that we would like to reproduce; a student has limited capacity, a different deployment budget, and access to only some form of teacher signal. Every distillation method must answer four questions:

1. Which inputs or prompts will be used?
2. What part of the teacher is observable: labels, generated sequences, logits, or hidden states?
3. Under whose prefixes is the student trained: data, teacher, or student prefixes?
4. Which discrepancy between teacher and student is minimized?

## Reading map

| Part | Teacher signal | Student training context | Main question |
| --- | --- | --- | --- |
| Classical logit distillation | Full class probabilities | Fixed dataset inputs | Why are soft targets richer than one-hot labels? |
| Token-level language-model KD | Next-token distribution | Dataset or teacher prefixes | How does a classifier loss extend to an autoregressive model? |
| Sequence-level KD | Complete teacher-generated response | Teacher-generated prefixes | Can teacher outputs be converted into an ordinary SFT dataset? |
| Reverse-KL distillation | Teacher probabilities | Student-generated prefixes | What happens when the student chooses where it is corrected? |
| Generalized on-policy KD | Teacher distribution or score | Mixture of student/data prefixes | How can train-time prefixes resemble deployment prefixes? |

## 1. What is being distilled?

### 1.1 Teacher, student, and transfer distribution

Let the teacher have parameters $\phi$ and the student have parameters $\theta$. For an input $x$, they define conditional distributions

$$
p_\phi(y\mid x)
\qquad\text{and}\qquad
q_\theta(y\mid x).
$$

The transfer inputs are drawn from a distribution $x\sim\mathcal D_{\mathrm{tr}}$.

`[EXPLANATION]` $\mathcal D_{\mathrm{tr}}$ need not contain labels. If the teacher can provide a distribution or generate a response, unlabeled inputs can still carry supervision.

`[SYNTHESIS]` A distillation result is therefore never specified by “teacher” and “student” alone. It also depends on the transfer distribution. A student can closely imitate the teacher on $\mathcal D_{\mathrm{tr}}$ and disagree elsewhere.

### 1.2 Why use a student at all?

`[DIRECT · HINTON ET AL.]` The original motivation is deployment: an ensemble or large model can be accurate but expensive, while a single smaller model can be faster and easier to serve.

`[SYNTHESIS]` For language models, the relevant constraints may include:

- parameter memory;
- key-value-cache memory;
- latency to first token;
- decoding throughput;
- energy or hardware limits;
- ability to run locally;
- licensing or API-access constraints.

`[CHECK]` Distillation does not automatically improve every one of these quantities. A smaller parameter count may not produce lower latency if the architecture, sequence length, batching regime, or hardware utilization changes unfavorably.

### 1.3 White-box and black-box access

`[DEFINITION]` In this note:

- **white-box distillation** means that the student trainer can inspect teacher probabilities, logits, or internal representations;
- **black-box distillation** means that only teacher-generated outputs or API responses are available.

`[SYNTHESIS]` The distinction changes the learning problem. White-box distillation can transfer the entire next-token distribution. Black-box imitation only observes samples from that distribution, often after decoding has further altered it.

## 2. Classical soft-target distillation

### 2.1 Hard labels discard similarity information

Suppose a $K$-class problem has a hard label $c$. Ordinary supervised learning minimizes

$$
\mathcal L_{\mathrm{hard}}
=-\log q_\theta(c\mid x).
$$

`[EXPLANATION]` This target says that class $c$ is correct, but it says nothing about how the remaining classes compare with one another.

If the teacher assigns

$$
p_\phi(\cdot\mid x)
=(0.70,0.20,0.08,0.02),
$$

then the second class is a much more plausible alternative than the fourth. A one-hot target $(1,0,0,0)$ erases that relation.

`[SYNTHESIS]` The non-maximal probabilities are often called **dark knowledge**: they encode the teacher's view of similarities and confusions among alternatives.

### 2.2 Temperature-scaled distributions

Let $z^T\in\mathbb R^K$ and $z^S\in\mathbb R^K$ be teacher and student logits. For temperature $\tau>0$, define

$$
p_i^{(\tau)}
=\frac{\exp(z_i^T/\tau)}
{\sum_j\exp(z_j^T/\tau)},
\qquad
q_i^{(\tau)}
=\frac{\exp(z_i^S/\tau)}
{\sum_j\exp(z_j^S/\tau)}.
$$

`[EXPLANATION]`

- $\tau=1$ gives the ordinary softmax;
- $\tau>1$ flattens the distribution and exposes more relative information;
- $0<\tau<1$ sharpens the distribution.

`[CHECK]` Temperature changes the target used during training. At deployment, the student does not have to decode with that same temperature.

### 2.3 The distillation loss

The teacher-to-student forward KL is

$$
\begin{aligned}
\mathcal L_{\mathrm{KD}}
&=D_{\mathrm{KL}}\!\left(
p^{(\tau)}\,\|\,q^{(\tau)}
\right)\\
&=\sum_{i=1}^{K}
p_i^{(\tau)}
\log\frac{p_i^{(\tau)}}{q_i^{(\tau)}}.
\end{aligned}
$$

The entropy of the teacher does not depend on $\theta$, so minimizing this KL is equivalent to minimizing soft-target cross-entropy:

$$
\boxed{
\mathcal L_{\mathrm{soft}}
=-\sum_i p_i^{(\tau)}\log q_i^{(\tau)}.
}
$$

### 2.4 Gradient with respect to student logits

`[DERIVATION · Step 1]` For temperature-scaled softmax,

$$
\frac{\partial\log q_i^{(\tau)}}{\partial z_k^S}
=\frac1\tau
\left(\mathbf 1\{i=k\}-q_k^{(\tau)}\right).
$$

`[DERIVATION · Step 2]` Differentiate the cross-entropy:

$$
\begin{aligned}
\frac{\partial\mathcal L_{\mathrm{soft}}}{\partial z_k^S}
&=-\sum_i p_i^{(\tau)}
\frac{\partial\log q_i^{(\tau)}}{\partial z_k^S}\\
&=\frac1\tau
\left(q_k^{(\tau)}-p_k^{(\tau)}\right).
\end{aligned}
$$

`[CONCLUSION]` Every class receives a signed correction. The student logit increases when it assigns less probability than the teacher and decreases when it assigns more.

### 2.5 Why use the squared-temperature factor?

At a larger temperature, both the explicit $1/\tau$ in the gradient and the shrinking difference $q^{(\tau)}-p^{(\tau)}$ reduce the gradient magnitude. Hinton et al. multiply the soft loss by $\tau^2$:

$$
\mathcal L
=(1-\alpha)\mathcal L_{\mathrm{hard}}
+\alpha\tau^2\mathcal L_{\mathrm{soft}}.
$$

`[DERIVATION · HIGH-TEMPERATURE INTUITION]` Center the logits so that $\sum_i z_i=0$. When $\tau$ is large,

$$
\exp(z_i/\tau)\approx 1+z_i/\tau,
$$

and therefore

$$
\operatorname{softmax}(z/\tau)_i
\approx \frac1K+\frac{z_i}{K\tau}.
$$

Thus

$$
q_i^{(\tau)}-p_i^{(\tau)}
\approx
\frac{z_i^S-z_i^T}{K\tau},
$$

and the unscaled logit gradient is approximately proportional to $1/\tau^2$. Multiplication by $\tau^2$ keeps the soft-target term on a comparable scale.

`[LIMIT]` This approximation explains the classical scaling rule, but it is not an identity for arbitrary temperatures or logits.

## 3. From classification to autoregressive language models

### 3.1 A response is a product of token conditionals

For a prompt $x$ and response $y=(y_1,\ldots,y_T)$,

$$
p_\phi(y\mid x)
=\prod_{t=1}^{T}
p_\phi(y_t\mid x,y_{<t}),
$$

and similarly

$$
q_\theta(y\mid x)
=\prod_{t=1}^{T}
q_\theta(y_t\mid x,y_{<t}).
$$

`[EXPLANATION]` The vocabulary is now the class set, but the classification input changes at every token because the prefix changes.

### 3.2 Token-level forward-KL distillation

Given a collection of prefixes $h_t=(x,y_{<t})$, a common objective is

$$
\boxed{
\mathcal L_{\mathrm{token\text{-}KD}}
=\mathbb E_{h_t\sim\mu}
\left[
D_{\mathrm{KL}}
\left(
p_\phi(\cdot\mid h_t)
\,\|\,
q_\theta(\cdot\mid h_t)
\right)
\right],
}
$$

where $\mu$ is a distribution over prefixes.

`[SYNTHESIS]` The loss is incomplete until $\mu$ is specified. Prefixes may come from:

- human demonstrations;
- a fixed text corpus;
- teacher-generated responses;
- student-generated responses;
- a mixture of these sources.

### 3.3 Sequence KL decomposes into token KL

`[DERIVATION · Step 1]` Start with forward KL over complete responses:

$$
D_{\mathrm{KL}}(p_\phi\|q_\theta)
=\mathbb E_{y\sim p_\phi}
\left[
\log\frac{p_\phi(y\mid x)}{q_\theta(y\mid x)}
\right].
$$

`[DERIVATION · Step 2]` Apply the autoregressive factorization:

$$
\log\frac{p_\phi(y\mid x)}{q_\theta(y\mid x)}
=\sum_{t=1}^{T}
\log\frac{
p_\phi(y_t\mid x,y_{<t})
}{
q_\theta(y_t\mid x,y_{<t})
}.
$$

`[DERIVATION · Step 3]` Condition on each teacher-generated prefix:

$$
\boxed{
D_{\mathrm{KL}}(p_\phi\|q_\theta)
=
\mathbb E_{y\sim p_\phi}
\left[
\sum_{t=1}^{T}
D_{\mathrm{KL}}
\left(
p_\phi(\cdot\mid x,y_{<t})
\,\|\,
q_\theta(\cdot\mid x,y_{<t})
\right)
\right].
}
$$

`[CONCLUSION]` Exact sequence-level forward KL can be estimated by sampling responses from the teacher and matching the teacher's full next-token distribution along those teacher trajectories.

### 3.4 Token averaging changes the objective

For a batch of variable-length responses, consider

$$
\mathcal L_{\mathrm{all\ tokens}}
=\frac{
\sum_i\sum_t m_{i,t}\,\ell_{i,t}
}{
\sum_i\sum_t m_{i,t}
},
$$

versus

$$
\mathcal L_{\mathrm{per\ response}}
=\frac1B\sum_i
\frac{
\sum_t m_{i,t}\,\ell_{i,t}
}{
\sum_t m_{i,t}
}.
$$

`[SYNTHESIS]` The first gives longer responses more total weight. The second gives every response equal weight. Distillation code should state which normalization it uses.

## 4. Three kinds of targets for language-model distillation

### 4.1 Hard ground-truth targets

With a demonstrated token $y_t^*$,

$$
\ell_{\mathrm{NLL},t}
=-\log q_\theta(y_t^*\mid x,y_{<t}^*).
$$

This is ordinary supervised fine-tuning.

### 4.2 Soft token targets

With teacher probabilities,

$$
\ell_{\mathrm{soft},t}
=-\sum_{v\in\mathcal V}
p_\phi^{(\tau)}(v\mid h_t)
\log q_\theta^{(\tau)}(v\mid h_t).
$$

`[EXPLANATION]` One forward pass of the teacher supplies a target over the entire vocabulary, not merely the sampled token.

### 4.3 Hard teacher-generated sequences

If only samples are available, draw or decode

$$
\hat y\sim p_\phi(\cdot\mid x)
$$

and train with

$$
\ell_{\mathrm{SeqKD}}
=-\sum_t
\log q_\theta(\hat y_t\mid x,\hat y_{<t}).
$$

`[DIRECT · KIM AND RUSH]` Sequence-level knowledge distillation trains a student on sequences produced by a teacher, converting teacher behavior into a standard supervised dataset.

`[SYNTHESIS]` Hard sequence targets are convenient and work with black-box teachers, but they discard the teacher's uncertainty at every token. The decoding algorithm also becomes part of the target-generating process.

## 5. Sequence-level knowledge distillation

### 5.1 Basic algorithm

For every transfer input $x_i$:

1. run teacher decoding to produce $\hat y_i$;
2. store $(x_i,\hat y_i)$;
3. train the student by maximum likelihood on the resulting dataset.

In pseudocode:

```python
distilled_examples = []

for prompt in transfer_prompts:
    teacher_response = teacher.generate(
        prompt,
        decoding="beam"  # or greedy / sampling
    )
    distilled_examples.append((prompt, teacher_response))

for prompt, response in batches(distilled_examples):
    loss = response_nll(student, prompt, response)
    loss.backward()
    optimizer.step()
```

### 5.2 Decoding is a lossy projection

Suppose the teacher distribution is multimodal:

$$
p_\phi(y\mid x)
=0.45\,\delta_{y_A}
+0.40\,\delta_{y_B}
+0.15\,\delta_{y_C}.
$$

Greedy or beam decoding may always choose $y_A$. A dataset containing only $y_A$ makes the effective target approximately $\delta_{y_A}$ rather than the full teacher distribution.

`[SYNTHESIS]` Sequence KD may simplify the output distribution in a way that helps a limited-capacity student, but it can also erase diversity and calibrated uncertainty.

### 5.3 Sampling versus deterministic decoding

`[EXPLANATION]`

- deterministic decoding produces cleaner and less diverse targets;
- stochastic sampling better represents teacher diversity but can include low-quality outputs;
- generating several candidates and filtering them spends more teacher compute in exchange for a curated training set.

`[CHECK]` “Train on teacher data” is not a complete recipe. The number of samples, decoding temperature, top-$p$, filtering rule, and prompt distribution all shape the student.

## 6. The prefix-distribution problem

### 6.1 Teacher forcing trains on prefixes chosen elsewhere

In fixed-data training, the student sees prefixes $y_{<t}$ from a dataset or the teacher. At deployment, it sees prefixes sampled from itself.

Let

$$
\mu_{\mathrm{train}}(h)
\neq
\mu_{q_\theta}(h)
$$

denote this mismatch.

`[EXPLANATION]` If the student makes an early mistake, the resulting prefix may be rare or absent in the training data. The student then receives no direct training on how to recover from that state.

### 6.2 A two-step example

Suppose the teacher almost always starts with token $A$, while the student sometimes starts with $B$.

- Offline teacher-forced KD trains the second token mostly after prefix $A$.
- Deployment asks the student to predict the second token after both $A$ and $B$.
- If prefix $B$ was never queried during training, teacher knowledge about how to continue from $B$ was not transferred.

`[SYNTHESIS]` This is not merely “error accumulation.” It is a support problem: the training process may never visit the states at which the deployed student needs guidance.

## 7. Forward KL and reverse KL

For a fixed prompt, abbreviate the teacher sequence distribution by $p(y)$ and the student by $q_\theta(y)$.

### 7.1 Forward KL

$$
D_{\mathrm{KL}}(p\|q_\theta)
=\mathbb E_{y\sim p}
\left[
\log p(y)-\log q_\theta(y)
\right].
$$

Only the second term depends on $\theta$, so

$$
\nabla_\theta D_{\mathrm{KL}}(p\|q_\theta)
=-
\mathbb E_{y\sim p}
[\nabla_\theta\log q_\theta(y)].
$$

`[EXPLANATION]` The teacher chooses which sequences appear in the expectation. Missing a sequence with substantial teacher probability is costly.

### 7.2 Reverse KL

$$
D_{\mathrm{KL}}(q_\theta\|p)
=\mathbb E_{y\sim q_\theta}
\left[
\log q_\theta(y)-\log p(y)
\right].
$$

`[EXPLANATION]` Now the student chooses which sequences appear in the expectation. Probability placed where the teacher assigns very little mass is costly.

### 7.3 Deriving the reverse-KL gradient

Define

$$
f_\theta(y)=\log q_\theta(y)-\log p(y).
$$

`[DERIVATION · Step 1]` Apply the score-function identity to an expectation whose distribution and integrand both depend on $\theta$:

$$
\nabla_\theta\mathbb E_{y\sim q_\theta}[f_\theta(y)]
=\mathbb E_{q_\theta}
\left[
f_\theta(y)\nabla_\theta\log q_\theta(y)
+\nabla_\theta f_\theta(y)
\right].
$$

`[DERIVATION · Step 2]` Since the teacher is fixed,

$$
\nabla_\theta f_\theta(y)
=\nabla_\theta\log q_\theta(y).
$$

`[DERIVATION · Step 3]` The expected score is zero:

$$
\mathbb E_{q_\theta}
[\nabla_\theta\log q_\theta(y)]
=\nabla_\theta\sum_y q_\theta(y)=0.
$$

Therefore

$$
\boxed{
\nabla_\theta D_{\mathrm{KL}}(q_\theta\|p)
=\mathbb E_{y\sim q_\theta}
\left[
\log\frac{q_\theta(y)}{p(y)}
\nabla_\theta\log q_\theta(y)
\right].
}
$$

`[CONNECTION]` This has the form of a policy gradient. The log-ratio acts like a sequence-level cost, and a baseline can be subtracted to reduce variance without changing the expected gradient.

### 7.4 Autoregressive log-ratio

For a sampled student response,

$$
\log\frac{q_\theta(y\mid x)}{p_\phi(y\mid x)}
=\sum_{t=1}^{T}
\log\frac{
q_\theta(y_t\mid x,y_{<t})
}{
p_\phi(y_t\mid x,y_{<t})
}.
$$

`[EXPLANATION]` The teacher can score the same student-generated prefix and token. This supplies a dense token-level discrepancy even though the response was sampled on-policy from the student.

### 7.5 “Mode covering” and “mode seeking” need care

`[COMMON INTUITION]` Forward KL is often described as mode covering because it penalizes the student for missing teacher-supported outcomes. Reverse KL is often described as mode seeking because it strongly penalizes student mass in teacher-low-probability regions.

`[LIMIT]` These phrases are qualitative intuitions, not universal theorems about a finite-capacity neural language model trained with stochastic gradients. The observed behavior also depends on parameterization, sampling, prefix distribution, optimization, and finite data.

## 8. On-policy distillation

### 8.1 Correct the student on its own prefixes

`[DIRECT · AGARWAL ET AL.]` Generalized Knowledge Distillation trains on student-generated sequences so the teacher can provide feedback at prefixes that the student itself visits.

A schematic objective is

$$
\mathcal L_{\mathrm{on\text{-}policy}}
=\mathbb E_{x\sim\mathcal D}
\mathbb E_{y\sim q_\theta(\cdot\mid x)}
\left[
\sum_t
d\!\left(
p_\phi(\cdot\mid x,y_{<t}),
q_\theta(\cdot\mid x,y_{<t})
\right)
\right],
$$

where $d$ may be a forward KL, reverse KL, Jensen–Shannon-style divergence, or another discrepancy.

### 8.2 Mixed-policy training

Pure on-policy sampling can be noisy early in training. A mixture can generate a prefix from the teacher or student:

$$
\pi_{\mathrm{mix}}
=(1-\lambda)p_\phi+\lambda q_\theta,
\qquad 0\le\lambda\le1.
$$

Then

$$
\mathcal L_{\mathrm{GKD}}
=\mathbb E_{y\sim\pi_{\mathrm{mix}}}
\left[
\sum_t d(p_{\phi,t},q_{\theta,t})
\right].
$$

`[SYNTHESIS]` $\lambda$ controls a curriculum over prefix distributions:

- $\lambda=0$: teacher-prefix distillation;
- $\lambda=1$: fully student-prefix distillation;
- intermediate $\lambda$: retain teacher-quality trajectories while exposing the student to its own mistakes.

### 8.3 A minimal on-policy loop

```python
for prompts in loader:
    with no_grad():
        student_tokens = student.generate(
            prompts,
            do_sample=True,
            temperature=1.0,
        )

        # Teacher is evaluated on the student's prefixes.
        teacher_logits = teacher(prompts, student_tokens).logits

    student_logits = student(prompts, student_tokens).logits

    loss = masked_token_kl(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        response_mask=response_mask(student_tokens),
    )

    loss.backward()
    optimizer.step()
```

`[CHECK]` The generated token IDs should be treated as sampled data. Accidentally backpropagating through an assumed differentiable sampling operation changes neither the discrete samples nor the intended gradient estimator.

## 9. Implementing token-level soft distillation

### 9.1 Stable loss computation

```python
def token_kd_loss(student_logits, teacher_logits, mask, tau=2.0):
    # [batch, time, vocabulary]
    teacher_prob = softmax(teacher_logits / tau, dim=-1)
    student_logprob = log_softmax(student_logits / tau, dim=-1)

    # Cross-entropy differs from forward KL only by teacher entropy.
    token_loss = -(teacher_prob * student_logprob).sum(dim=-1)

    masked_mean = (token_loss * mask).sum() / mask.sum().clamp_min(1)
    return (tau ** 2) * masked_mean
```

`[IMPLEMENTATION NOTE]` Computing `log_softmax` directly is more numerically stable than taking a logarithm after `softmax`.

### 9.2 Combining hard and soft supervision

```python
loss = (
    (1.0 - alpha) * hard_response_nll(student_logits, labels)
    + alpha * token_kd_loss(
        student_logits,
        teacher_logits,
        response_mask,
        tau=temperature,
    )
)
```

`[SYNTHESIS]` Hard targets anchor the student to observed desired outputs. Soft targets communicate how the teacher distributes probability among alternatives. The mixing coefficient $\alpha$ is a modeling choice, not a harmless scale factor.

### 9.3 Vocabulary alignment

The token-KL formula assumes teacher and student distributions share the same vocabulary and tokenization.

`[LIMIT]` If tokenizers differ, token $v$ in one model need not correspond to a single token in the other. Options include:

- sequence-level distillation;
- mapping distributions into a shared vocabulary;
- distilling representations after learned projections;
- matching probabilities over strings rather than raw token IDs.

None is automatically equivalent to same-vocabulary token KL.

## 10. Numerical example: one soft-target update

Suppose the teacher and student next-token probabilities are

$$
p=(0.70,0.20,0.10),
\qquad
q=(0.40,0.30,0.30),
$$

at $\tau=1$.

The gradient of the soft cross-entropy with respect to the student logits is

$$
q-p=(-0.30,0.10,0.20).
$$

Under gradient descent:

- the first logit increases because the student underestimates it by $0.30$;
- the second logit decreases slightly;
- the third decreases more strongly because the student assigns three times the teacher probability.

The forward KL is

$$
\begin{aligned}
D_{\mathrm{KL}}(p\|q)
&=0.70\log\frac{0.70}{0.40}
+0.20\log\frac{0.20}{0.30}
+0.10\log\frac{0.10}{0.30}\\
&\approx 0.201.
\end{aligned}
$$

The reverse KL is

$$
\begin{aligned}
D_{\mathrm{KL}}(q\|p)
&=0.40\log\frac{0.40}{0.70}
+0.30\log\frac{0.30}{0.20}
+0.30\log\frac{0.30}{0.10}\\
&\approx 0.227.
\end{aligned}
$$

`[EXPLANATION]` KL is asymmetric because the expectation weights differ. Forward KL weights the log-ratio by teacher probability; reverse KL weights it by student probability.

## 11. Capacity gap and the projection viewpoint

Let $\mathcal Q$ be the family of distributions the student architecture can represent. Distillation approximately solves

$$
q^*
=\arg\min_{q\in\mathcal Q}
D(p_\phi,q).
$$

`[SYNTHESIS]` If $p_\phi\notin\mathcal Q$, the student cannot exactly copy the teacher. The divergence determines which approximation is preferred.

### 11.1 Why a smaller student may need a simpler target

The teacher may allocate probability across many valid styles, reasoning paths, and lengths. A limited student may struggle to represent all of them simultaneously.

Possible responses include:

- sharpen the teacher distribution;
- use deterministic sequence targets;
- restrict the prompt domain;
- introduce an intermediate-size assistant teacher;
- distill one capability at a time;
- increase student capacity.

`[HYPOTHESIS]` The success of a narrower, cleaner teacher-generated dataset can partly reflect an easier projection problem rather than perfect transfer of the teacher's complete behavior.

### 11.2 The student can sometimes exceed the teacher on a metric

`[EXPLANATION]` Distillation is not a theorem that the student must score below the teacher. A student may benefit from regularization, a different architecture, additional hard labels, filtered data, or an evaluation distribution closer to the transfer set.

`[CHECK]` “Student beats teacher” does not imply that all teacher knowledge was transferred. It is a statement about a particular metric and distribution.

## 12. Distillation versus nearby post-training methods

| Method | Supervision | Primary objective | Main limitation |
| --- | --- | --- | --- |
| SFT | Demonstrated response | Increase likelihood of one target | Does not expose alternatives |
| Soft KD | Teacher token distribution | Match dense teacher probabilities | Requires logit access and aligned vocabularies |
| Sequence KD | Teacher-generated response | Imitate sampled/decoded outputs | Loses uncertainty; depends on decoding |
| DPO | Preferred and rejected responses | Increase relative preference under a reference | Requires comparison data |
| RLHF/RLVR | Scalar learned or verified reward | Increase expected reward | Sampling and optimization are more complex |

`[CONNECTION TO LECTURE 12]` SFT uses a delta-like target concentrated on the demonstrated token. Soft distillation replaces that target with the teacher's full distribution. DPO does not try to reproduce a teacher distribution; it fits relative preferences.

`[CONNECTION TO LECTURE 13]` On-policy distillation and policy gradient both sample from the current student/policy. Their signals differ: distillation queries a teacher distribution, whereas RLVR queries a verifier. Reverse-KL gradients can nevertheless take a policy-gradient form.

## 13. Failure modes and diagnostics

### 13.1 Poor transfer prompts

If $\mathcal D_{\mathrm{tr}}$ omits a capability, the teacher provides no supervision for it.

**Diagnostic:** stratify evaluation by prompt domain and compare coverage with the transfer set.

### 13.2 Teacher errors become labels

A generated response can be fluent and wrong. Sequence KD may turn that error into a hard target.

**Diagnostic:** retain teacher confidence, verifier scores, human checks, or multiple candidates rather than assuming every teacher generation is correct.

### 13.3 Exposure bias

The student performs well on teacher prefixes but degrades on self-generated prefixes.

**Diagnostic:** compare token KL on dataset prefixes, teacher prefixes, and student prefixes. A large gap indicates a prefix-distribution problem.

### 13.4 Temperature is miscalibrated

Very high temperature can overemphasize a noisy tail; very low temperature approaches hard-label imitation.

**Diagnostic:** inspect probability mass and gradient contribution by teacher-probability rank, not only the final loss.

### 13.5 Loss decreases while deployment quality stalls

The student may match teacher probabilities on the training prefixes without improving end-to-end generations.

**Diagnostic:** track both teacher-agreement metrics and task metrics under actual decoding settings.

### 13.6 Efficiency is not measured end to end

A smaller model may have worse throughput because of kernels, batching, or memory movement.

**Diagnostic:** report latency, throughput, peak memory, energy when available, and quality at the deployment sequence lengths and hardware.

## 14. A practical experiment design

### 14.1 Controlled comparison

Hold constant:

- student initialization and architecture;
- transfer prompts;
- number of response tokens processed;
- optimizer and learning-rate budget;
- evaluation prompts and decoding.

Compare:

1. SFT on original responses;
2. sequence KD on teacher responses;
3. offline token-level forward KL;
4. on-policy token-level KD;
5. a hard-plus-soft mixture.

### 14.2 Metrics

Measure at least three layers:

| Layer | Example metrics | What it tests |
| --- | --- | --- |
| Agreement | token KL, teacher win rate | Does the student imitate the teacher? |
| Task quality | accuracy, pass rate, human preference | Does imitation help the target task? |
| Deployment | latency, throughput, memory | Did compression achieve its practical goal? |

`[SYNTHESIS]` No single metric establishes successful distillation. Low KL can coexist with poor generations, and strong benchmark quality can coexist with little deployment benefit.

## 15. A compact decision procedure

1. **Only teacher text is available:** begin with sequence KD and carefully document decoding and filtering.
2. **Teacher logits and a shared tokenizer are available:** add token-level soft targets.
3. **Student generations drift after early errors:** query the teacher on student prefixes and try on-policy or mixed-policy KD.
4. **The student is much smaller:** test temperature, target simplification, and intermediate teachers instead of assuming exact imitation is possible.
5. **Serving cost is the goal:** measure real deployment cost, not just parameter count.

## 16. Open questions

1. Which prefix mixture gives the best tradeoff between teacher-quality trajectories and correction of student mistakes?
2. How should temperature depend on student capacity or token position?
3. When does preserving teacher uncertainty improve generation, and when does it overwhelm a smaller student?
4. Can a black-box student recover calibrated uncertainty from multiple teacher samples without logits?
5. How should distillation handle different tokenizers without reducing everything to hard sequence imitation?
6. Which teacher capabilities are absent because the transfer prompt distribution never elicits them?
7. How much apparent compression comes from narrowing the task rather than transferring general capability?
8. Can verifier-guided filtering remove teacher errors without collapsing response diversity?

## 17. What to verify manually

- Re-read the four primary papers and confirm which variants were actually emphasized in the student presentations.
- Check whether the course presenters used “distillation” mainly for compression, synthetic-data imitation, or self-distillation.
- Reproduce the $\tau^2$ scaling derivation with centered logits.
- Implement the same-vocabulary token-KL loss and verify its gradient numerically.
- Compare teacher-prefix and student-prefix losses on one small autoregressive model.
- Replace broad synthesis claims with the author's own examples and conclusions after studying the papers.

## Source trail

- CS 395T course schedule: <https://noahgol.github.io/teaching/cs395t-f26/>
- Hinton, Vinyals, and Dean, *Distilling the Knowledge in a Neural Network*: <https://arxiv.org/abs/1503.02531>
- Kim and Rush, *Sequence-Level Knowledge Distillation*: <https://arxiv.org/abs/1606.07947>
- Gu et al., *MiniLLM: Knowledge Distillation of Large Language Models*: <https://arxiv.org/abs/2306.08543>
- Agarwal et al., *On-Policy Distillation of Language Models*: <https://arxiv.org/abs/2306.13649>

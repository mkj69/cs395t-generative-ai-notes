---
title: "Lecture 15 — LoRA, Continual Learning, and Test-Time Training"
slug: "lecture-15"
date: "2026-10-14"
updated: "2026-10-12"
status: "draft"
type: "learning-note"
course: "CS 395T"
source: "https://noahgol.github.io/teaching/cs395t-f26/"
summary: "A detailed AI-assisted study scaffold connecting low-rank adaptation, catastrophic forgetting, replay and regularization, and test-time parameter updates under distribution shift."
public: true
tags:
  - LoRA
  - parameter-efficient fine-tuning
  - continual learning
  - catastrophic forgetting
  - experience replay
  - elastic weight consolidation
  - test-time training
  - test-time adaptation
related_research_notes: []
authorship: "ai-assisted"
---

> **AI-assisted study scaffold, not a lecture transcript.** The official course schedule lists Lecture 15 as a student-presentation session on LoRA, continual learning, and test-time training, but it does not publish a unified handout or slide deck. This note therefore does not claim to reconstruct what the presenters said. It develops the named topics from the primary papers listed below and marks derivations and synthesis explicitly. It has not yet been manually verified or rewritten by the author.

## Source boundary

`[DIRECT · COURSE SCHEDULE]` The official schedule names the October 14 session **“Post-training IV (Student Presentations #3): LoRA, continual learning, & test-time training.”** It does not link lecture notes or slides for this session: <https://noahgol.github.io/teaching/cs395t-f26/>.

`[DIRECT · PRIMARY SOURCES]` The main technical sources used here are:

1. Hu et al. (2021), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
2. Kirkpatrick et al. (2017), [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796).
3. Lopez-Paz and Ranzato (2017), [Gradient Episodic Memory for Continual Learning](https://arxiv.org/abs/1706.08840).
4. Li and Hoiem (2016), [Learning without Forgetting](https://arxiv.org/abs/1606.09282).
5. Sun et al. (2019), [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](https://arxiv.org/abs/1909.13231).
6. Wang et al. (2020), [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726).

`[SOURCE BOUNDARY]` Statements labeled `[DIRECT]` are attributable to these papers or to the course schedule. `[DERIVATION]` lines work through the mathematics. `[EXPLANATION]` and `[SYNTHESIS]` organize the material but are not reported classroom claims.

`[SYNTHESIS]` The three topics share one question:

> When the task or input distribution changes, which parts of a pretrained model should be allowed to move, what signal should move them, and what prevents useful old behavior from being destroyed?

- **LoRA** restricts the *parameterization* of the update.
- **Continual learning** restricts the *damage to earlier tasks* while learning a stream of new ones.
- **Test-time training** changes parameters *during deployment* using information available on test inputs.

These are related tools, but they are not interchangeable.

## Reading map

| Part | What changes? | Available signal | Main risk |
| --- | --- | --- | --- |
| Full fine-tuning | All selected parameters | Labeled or preference data | Cost and interference |
| LoRA | Low-rank update matrices | Ordinary fine-tuning loss | Rank or placement may be insufficient |
| Continual learning | Shared parameters across a task stream | New data plus memory/constraints | Catastrophic forgetting |
| Test-time training | Some parameters at deployment | Unlabeled auxiliary signal | Drift, collapse, and latency |

## 1. A common mathematical frame

Let a pretrained model have parameters $\theta_0$. Adaptation produces

$$
\theta' = \theta_0 + \Delta\theta.
$$

Every adaptation method implicitly chooses:

1. an allowed update set $\Delta\theta\in\mathcal U$;
2. an adaptation loss $\mathcal L_{\mathrm{adapt}}$;
3. data or observations on which that loss is computed;
4. a rule for preserving previous behavior;
5. a moment at which the update occurs.

`[SYNTHESIS]` LoRA primarily changes $\mathcal U$. Continual learning changes the objective and data retained across time. Test-time training changes when and from which distribution the update is computed.

### 1.1 Local quadratic view

Near $\theta_0$, a loss can be approximated by

$$
\mathcal L(\theta_0+\Delta\theta)
\approx
\mathcal L(\theta_0)
+g^\top\Delta\theta
+\frac12\Delta\theta^\top H\Delta\theta,
$$

where $g=\nabla\mathcal L(\theta_0)$ and $H$ is the Hessian.

`[EXPLANATION]` Adaptation asks for a direction that reduces the new loss. Preservation asks whether that same direction increases an old loss. Parameter-efficient methods restrict which directions are available.

## 2. LoRA: low-rank parameter updates

### 2.1 From full fine-tuning to a structured update

Consider a pretrained linear transformation

$$
h = W_0x,
\qquad
W_0\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}.
$$

Full fine-tuning learns an unrestricted update $\Delta W$:

$$
h=(W_0+\Delta W)x.
$$

`[DIRECT · LORA]` LoRA freezes $W_0$ and parameterizes the update as a product of two thin matrices:

$$
\boxed{
\Delta W = \frac{\alpha}{r}BA,
}
$$

with

$$
A\in\mathbb R^{r\times d_{\mathrm{in}}},
\qquad
B\in\mathbb R^{d_{\mathrm{out}}\times r},
\qquad
r\ll \min(d_{\mathrm{in}},d_{\mathrm{out}}).
$$

The adapted layer is

$$
h=W_0x+\frac{\alpha}{r}B(Ax).
$$

### 2.2 Why the update is low rank

By the rank inequality,

$$
\operatorname{rank}(BA)
\le
\min(\operatorname{rank}(B),\operatorname{rank}(A))
\le r.
$$

`[EXPLANATION]` LoRA does not make the pretrained weight matrix low rank. It makes the *change to that matrix* have rank at most $r$.

### 2.3 Parameter count

Full fine-tuning stores

$$
d_{\mathrm{out}}d_{\mathrm{in}}
$$

trainable parameters for this matrix. LoRA stores

$$
r(d_{\mathrm{in}}+d_{\mathrm{out}}).
$$

The ratio is

$$
\frac{r(d_{\mathrm{in}}+d_{\mathrm{out}})}
{d_{\mathrm{out}}d_{\mathrm{in}}}.
$$

For a square $4096\times4096$ matrix with $r=8$:

$$
4096^2=16{,}777{,}216
$$

full-update parameters, versus

$$
8(4096+4096)=65{,}536
$$

LoRA parameters—a factor of $256$ fewer for that matrix.

`[CHECK]` This is a trainable-parameter comparison, not automatically the total training-memory ratio. Activations, frozen weights, gradients for other modules, optimizer choices, and sequence length still matter.

### 2.4 Scaling and initialization

The factor $\alpha/r$ separates the nominal rank from the update scale:

$$
\Delta W = sBA,
\qquad s=\alpha/r.
$$

`[DIRECT · LORA]` A common initialization makes one factor random and the other zero, so that

$$
BA=0
$$

at initialization and the adapted model initially equals the base model.

`[DERIVATION]` If $B=0$ initially, then for a scalar loss $L$ and upstream gradient $g_h=\partial L/\partial h$,

$$
\frac{\partial L}{\partial B}
=s\,g_h(Ax)^\top,
$$

while

$$
\frac{\partial L}{\partial A}
=s\,B^\top g_hx^\top=0
$$

on the first step. $B$ moves first; after $B$ becomes nonzero, $A$ also receives gradients.

### 2.5 LoRA as a constrained update space

If the best unrestricted update has singular-value decomposition

$$
\Delta W^*=U\Sigma V^\top,
$$

the best rank-$r$ approximation in Frobenius norm keeps the largest $r$ singular components:

$$
\Delta W_r^*=U_r\Sigma_rV_r^\top.
$$

`[SYNTHESIS]` This does not prove that learned LoRA recovers the truncated SVD of a full fine-tuning update. It explains the inductive bias: LoRA assumes the useful update lies near a low-dimensional matrix subspace.

### 2.6 Merging at inference

After training, define

$$
W_{\mathrm{merged}}
=W_0+\frac{\alpha}{r}BA.
$$

Then inference uses

$$
h=W_{\mathrm{merged}}x
$$

with no separate adapter branch.

`[EXPLANATION]` This is why LoRA can add no extra inference-layer depth after merging. Multiple task-specific adapters can instead remain separate and be selected dynamically.

`[LIMIT]` Merging several adapters by simply adding their updates is not guaranteed to preserve each adapter's behavior; update directions can interfere.

### 2.7 Minimal implementation

```python
class LoRALinear(Module):
    def __init__(self, frozen_weight, rank, alpha):
        self.weight = frozen_weight          # requires_grad = False
        self.A = Parameter(randn(rank, frozen_weight.in_features) * 0.01)
        self.B = Parameter(zeros(frozen_weight.out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        base = linear(x, self.weight)
        update = (x @ self.A.T) @ self.B.T
        return base + self.scale * update
```

`[CHECK]` Applying LoRA to “the model” is underspecified. One must state which projections receive adapters—for example, attention query/value matrices, all attention projections, or feed-forward matrices.

## 3. What LoRA does not solve by itself

`[SYNTHESIS]` LoRA reduces the trainable update space and the number of optimizer states, but it does not itself specify:

- which data should be remembered;
- how to detect task boundaries;
- how to prevent forgetting if the same adapter is trained sequentially;
- how to select or compose task-specific adapters;
- what to do under unlabeled test-time shift.

If one LoRA adapter is updated on tasks $1,2,\ldots,K$, it can still catastrophically forget earlier tasks. If one adapter is stored per task, forgetting may be reduced by isolation, but task identification and storage grow with the number of tasks.

## 4. Continual learning: learning from a stream

### 4.1 Problem setup

Suppose data arrive from a sequence of distributions

$$
\mathcal D_1,\mathcal D_2,\ldots,\mathcal D_K.
$$

After training through task $k$, the model should perform well not only on $\mathcal D_k$ but also on earlier tasks.

Let $R_{k,j}$ be performance on task $j$ after training through task $k$.

One simple average final performance is

$$
A_K=\frac1K\sum_{j=1}^{K}R_{K,j}.
$$

A forgetting measure for task $j$ is

$$
F_j
=\max_{\ell\in\{j,\ldots,K-1\}}R_{\ell,j}
-R_{K,j}.
$$

`[EXPLANATION]` A method can have strong accuracy on the newest task while having large forgetting on old tasks. Reporting only current-task accuracy hides the central continual-learning failure.

### 4.2 Task settings matter

`[DEFINITION]`

- **Task-incremental:** the task identity is known at inference and may select a head or adapter.
- **Domain-incremental:** the output space stays fixed while the input distribution changes.
- **Class-incremental:** new classes arrive and inference must distinguish all classes without a task label.

`[CHECK]` Results from these settings are not directly interchangeable. Task identity can make routing substantially easier.

## 5. Why catastrophic forgetting occurs

### 5.1 Gradient interference

Let $L_A(\theta)$ be an old-task loss and $L_B(\theta)$ a new-task loss. A new-task gradient step is

$$
\theta' = \theta-\eta g_B,
\qquad
g_B=\nabla L_B(\theta).
$$

`[DERIVATION]` First-order Taylor expansion of the old loss gives

$$
\begin{aligned}
L_A(\theta')
&\approx L_A(\theta)
+\nabla L_A(\theta)^\top(\theta'-\theta)\\
&=L_A(\theta)-\eta g_A^\top g_B.
\end{aligned}
$$

Therefore:

- if $g_A^\top g_B>0$, the new step also decreases the old loss locally;
- if $g_A^\top g_B<0$, the new step increases the old loss locally;
- if the dot product is near zero, the tasks are locally non-interfering.

`[SYNTHESIS]` Catastrophic forgetting is not only about parameter distance. It is about movement in directions important for old behavior.

### 5.2 Stability–plasticity tradeoff

- **Plasticity:** change enough to learn the new task.
- **Stability:** avoid changing behavior needed by old tasks.

Too much stability prevents learning; too much plasticity destroys prior behavior. Continual-learning algorithms choose different mechanisms for navigating this tradeoff.

## 6. Replay: keep some old evidence

### 6.1 Basic replay objective

Maintain a memory buffer $\mathcal M$ containing old examples. On task $k$, optimize

$$
\mathcal L_{\mathrm{replay}}(\theta)
=
\mathbb E_{z\sim\mathcal D_k}[\ell(\theta;z)]
+\lambda
\mathbb E_{z\sim\mathcal M}[\ell(\theta;z)].
$$

`[EXPLANATION]` Replay turns preservation into ordinary training on a mixture of new and remembered data.

### 6.2 Reservoir sampling for a bounded buffer

Suppose the stream has seen $t$ examples and the buffer holds $M$ items.

For the $t$-th example:

1. if $t\le M$, insert it;
2. otherwise, retain it with probability $M/t$;
3. if retained, replace a uniformly chosen buffer item.

`[DERIVATION]` After processing $t$ items, each past item has probability $M/t$ of being in the buffer. Thus the buffer is a uniform sample of the observed stream.

### 6.3 What replay buys and costs

`[SYNTHESIS]`

- It directly represents old data and often gives a strong baseline.
- It consumes storage and may create privacy or licensing constraints.
- A small buffer may underrepresent rare behaviors.
- Replaying model-generated approximations avoids raw storage but introduces generator error.

## 7. Functional regularization: preserve outputs

### 7.1 Learning without Forgetting

`[DIRECT · LWF]` Learning without Forgetting uses the old model's outputs on new-task inputs as targets for preserving old capabilities when old data are unavailable.

Let $f_{\theta^-}$ be the frozen pre-update model and $f_\theta$ the model being adapted. A schematic objective is

$$
\mathcal L(\theta)
=\mathcal L_{\mathrm{new}}(\theta)
+\lambda T^2
D_{\mathrm{KL}}
\left(
p_{\theta^-}^{(T)}(\cdot\mid x)
\,\|\,
p_\theta^{(T)}(\cdot\mid x)
\right).
$$

`[CONNECTION TO LECTURE 14]` This is distillation used as an anti-forgetting constraint: the previous model is the teacher and the updated model is the student.

`[LIMIT]` If new-task inputs do not activate old-task behavior, matching outputs only on those inputs may not preserve the old function where it matters.

## 8. Parameter regularization: Elastic Weight Consolidation

### 8.1 Bayesian starting point

After task $A$ and before task $B$, Bayes' rule gives

$$
p(\theta\mid D_A,D_B)
\propto
p(D_B\mid\theta)p(\theta\mid D_A).
$$

Taking negative logs:

$$
-\log p(\theta\mid D_A,D_B)
=
-\log p(D_B\mid\theta)
-\log p(\theta\mid D_A)
+C.
$$

The first term is the new-task loss. The second is a prior supplied by the old-task posterior.

### 8.2 Laplace approximation

Approximate the old posterior near its optimum $\theta_A^*$ by a Gaussian:

$$
-\log p(\theta\mid D_A)
\approx
\frac12
(\theta-\theta_A^*)^\top
F_A
(\theta-\theta_A^*),
$$

where $F_A$ is approximated by a Fisher-information matrix.

Using a diagonal approximation yields the EWC objective:

$$
\boxed{
\mathcal L_{\mathrm{EWC}}(\theta)
=\mathcal L_B(\theta)
+\frac{\lambda}{2}
\sum_i F_{A,i}
(\theta_i-\theta_{A,i}^*)^2.
}
$$

`[EXPLANATION]` A parameter with large $F_{A,i}$ is treated as important for task $A$ and is strongly pulled toward its old value. A low-importance parameter is freer to move.

### 8.3 Two-parameter example

Suppose

$$
\theta_A^*=(0,0),
\qquad
F_A=(10,1).
$$

Moving to $\theta=(1,1)$ incurs

$$
\frac{\lambda}{2}
\left(10\cdot1^2+1\cdot1^2\right)
=\frac{11\lambda}{2}.
$$

Moving to $\theta=(0,1)$ costs only

$$
\frac{\lambda}{2}.
$$

`[EXPLANATION]` EWC prefers adapting through the second coordinate because the first was estimated to be more important to the old task.

### 8.4 Limits of the approximation

`[LIMIT]`

- A diagonal Fisher ignores interactions between parameters.
- The old posterior may be far from Gaussian.
- Importance estimates depend on the data used to compute them.
- Repeated tasks require combining or updating multiple constraints.
- A very large $\lambda$ protects the past by preventing useful new learning.

## 9. Gradient constraints: GEM

`[DIRECT · GEM]` Gradient Episodic Memory stores examples from previous tasks and constrains the new update so it does not increase their losses locally.

Let $g$ be the gradient on current data and $g_j$ the gradient on memory from old task $j$. We seek an adjusted gradient $\tilde g$ satisfying

$$
\tilde g^\top g_j\ge0
\qquad\text{for every old task }j.
$$

Why this sign? A step $\theta' = \theta-\eta\tilde g$ changes old loss $L_j$ by

$$
L_j(\theta')-L_j(\theta)
\approx
-\eta g_j^\top\tilde g.
$$

Thus $g_j^\top\tilde g\ge0$ implies a non-increasing old loss to first order.

If the raw gradient violates a constraint, solve

$$
\boxed{
\min_{\tilde g}
\frac12\|\tilde g-g\|_2^2
\quad\text{subject to}\quad
\tilde g^\top g_j\ge0
\ \forall j.
}
$$

`[SYNTHESIS]` Replay adds old examples to the objective. GEM additionally reasons about the geometry of old and new gradients.

## 10. Three families of continual-learning defenses

| Family | Core mechanism | Representative example | Main cost |
| --- | --- | --- | --- |
| Replay | Train again on remembered old evidence | Experience replay | Storage and extra compute |
| Regularization | Penalize changes important to old behavior | EWC, output distillation | Approximate importance or coverage |
| Isolation/expansion | Give tasks separate parameters or routes | Task-specific LoRA adapters | Routing and growing storage |

`[SYNTHESIS]` Hybrid methods are natural: replay can be combined with LoRA; EWC can regularize adapter parameters; task-specific adapters can share a common base.

## 11. Test-time training and adaptation

### 11.1 The deployment setting

Training data follow $P_{\mathrm{train}}(x,y)$, while deployment data may follow

$$
P_{\mathrm{test}}(x,y)
\neq
P_{\mathrm{train}}(x,y).
$$

At test time, labels $y$ are usually unavailable. Adaptation therefore needs an auxiliary signal computed from $x$ or from the model's predictions.

`[DEFINITION]`

- **Test-time training (TTT):** train the model so that an auxiliary self-supervised update on a test example improves the main prediction.
- **Test-time adaptation (TTA):** a broader family of methods that updates a pretrained model during testing, possibly without joint training of the auxiliary task.

The literature sometimes uses these names less strictly; the update signal and protocol should always be stated.

### 11.2 One-step test-time update

Let $L_{\mathrm{aux}}(x;\theta)$ be a label-free auxiliary loss. For a test input $x$:

$$
\theta'(x)
=\theta-\eta
\nabla_\theta L_{\mathrm{aux}}(x;\theta).
$$

Then predict

$$
\hat y
=f_{\theta'(x)}(x).
$$

`[EXPLANATION]` The model does not merely run inference. It first performs learning on the test input, then predicts with the adapted parameters.

### 11.3 Episodic and online protocols

- **Episodic:** reset to $\theta_0$ before every example or batch.
- **Online:** carry the adapted parameters into the next batch.

`[SYNTHESIS]` Episodic adaptation isolates examples but repeats work. Online adaptation can exploit persistent shifts but can accumulate errors and become order-dependent.

## 12. Test-Time Training with self-supervision

`[DIRECT · SUN ET AL.]` Test-Time Training uses a self-supervised auxiliary task on an unlabeled test example, updates shared features, and then makes the main prediction.

During source training, one can optimize

$$
L_{\mathrm{train}}(\theta)
=L_{\mathrm{main}}(x,y;\theta)
+\lambda L_{\mathrm{aux}}(x;\theta).
$$

At test time, only $L_{\mathrm{aux}}$ is available.

### 12.1 One-step meta-gradient intuition

An even more explicit objective asks whether the auxiliary update helps the main task:

$$
\theta'(x)=\theta-\eta\nabla_\theta L_{\mathrm{aux}}(x;\theta),
$$

then minimizes

$$
L_{\mathrm{main}}(x,y;\theta'(x)).
$$

`[DERIVATION]` Differentiating through the update gives

$$
\nabla_\theta L_{\mathrm{main}}(\theta')
=
\left(I-\eta H_{\mathrm{aux}}(\theta)\right)^\top
\nabla_{\theta'}L_{\mathrm{main}}(\theta'),
$$

where

$$
H_{\mathrm{aux}}(\theta)
=\nabla_\theta^2L_{\mathrm{aux}}(x;\theta).
$$

`[EXPLANATION]` The training process can shape the representation so that an auxiliary gradient at test time points toward parameters useful for the main task.

`[CHECK]` Jointly training an auxiliary loss is not enough to guarantee that its test-time gradient improves the main task under every shift.

## 13. Tent: entropy-minimization adaptation

`[DIRECT · TENT]` Tent adapts at test time by minimizing prediction entropy and updating normalization statistics plus selected affine normalization parameters.

For class probabilities $p_\theta(c\mid x)$, predictive entropy is

$$
H_\theta(Y\mid x)
=-
\sum_{c=1}^{C}
p_\theta(c\mid x)
\log p_\theta(c\mid x).
$$

The batch objective is

$$
L_{\mathrm{Tent}}(\theta)
=\frac1B\sum_{i=1}^{B}
H_\theta(Y\mid x_i).
$$

### 13.1 Why entropy supplies a gradient without labels

If a prediction is uncertain, probability is spread across classes and entropy is high. Gradient descent on entropy pushes the distribution toward greater confidence.

`[SYNTHESIS]` This can help when the model's decision boundary is approximately correct but predictions become uncertain under a covariate shift.

`[LIMIT]` Confidence is not correctness. If the initial prediction is wrong, entropy minimization can make it confidently wrong. Unconstrained minimization can also encourage collapse toward a narrow set of outputs.

### 13.2 Restricting which parameters move

Tent updates a small selected subset rather than every model parameter.

`[SYNTHESIS]` This is the same structural instinct as parameter-efficient fine-tuning: adaptation becomes safer and cheaper when the update space is deliberately restricted. The exact parameter subset is nevertheless part of the algorithm, not an implementation footnote.

### 13.3 Minimal adaptation loop

```python
adapt_params = select_adaptation_parameters(model)
optimizer = Adam(adapt_params, lr=test_lr)

for test_batch in stream:
    probabilities = softmax(model(test_batch), dim=-1)
    entropy = -(probabilities * log(probabilities + 1e-8)).sum(dim=-1)
    loss = entropy.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with no_grad():
        predictions = model(test_batch).argmax(dim=-1)
```

`[CHECK]` This code predicts after the update. Some protocols predict before updating or reuse the same forward pass; these choices change evaluation and latency.

## 14. LoRA at test time

One can freeze the base model and update only LoRA parameters during deployment:

$$
\theta_{\mathrm{base}}'=\theta_{\mathrm{base}},
\qquad
(A',B')=(A,B)-\eta\nabla_{A,B}L_{\mathrm{aux}}.
$$

Potential advantages:

- fewer trainable parameters and optimizer states;
- the base model remains recoverable;
- episodic reset means restoring a small adapter;
- separate adapters can represent different environments.

Potential limitations:

- the useful test-time correction may not lie in the chosen low-rank subspace;
- online adapters can still drift or forget;
- selecting the correct adapter may require domain detection;
- generation-time auxiliary objectives for LLMs may be expensive or poorly aligned.

`[SYNTHESIS]` “LoRA + test-time training” is a composition: LoRA specifies the update coordinates, while test-time training specifies the signal and timing.

## 15. A toy gradient-conflict example

Let

$$
g_A=(1,0),
\qquad
g_B=(-1,1).
$$

Their dot product is

$$
g_A^\top g_B=-1.
$$

A gradient step on task $B$ therefore increases task $A$'s loss to first order.

GEM seeks the nearest $\tilde g$ to $g_B$ satisfying

$$
g_A^\top\tilde g\ge0.
$$

The violated component is the negative first coordinate. Projecting onto the boundary gives

$$
\tilde g=(0,1).
$$

Now

$$
g_A^\top\tilde g=0,
$$

so the step is locally neutral for task $A$ while retaining the non-conflicting part of the task-$B$ gradient.

## 16. Comparing the mechanisms

| Method | Update restriction | Preservation mechanism | Requires old data? | Test-time updates? |
| --- | --- | --- | --- | --- |
| LoRA | Low-rank matrix updates | None by itself | No | Optional |
| Replay | Usually unrestricted | Rehearse remembered examples | Yes, or generated replay | Usually no |
| LwF | Usually unrestricted | Match old model outputs | No old labels/data, but uses old model | Usually no |
| EWC | Soft parameter penalty | Protect high-importance weights | No raw old data | Usually no |
| GEM | Gradient projection | Prevent local old-loss increase | Yes, episodic memory | Usually no |
| TTT/Tent | Selected deployment parameters | Reset, restriction, or stream protocol | No old training data | Yes |

## 17. Failure modes and diagnostics

### 17.1 LoRA rank is too small or placed poorly

**Symptom:** training loss plateaus above full fine-tuning.

**Diagnostic:** vary rank and target modules separately; compare trainable parameter counts and validation quality.

### 17.2 LoRA overfits despite few parameters

**Symptom:** training quality rises while held-out behavior degrades.

**Diagnostic:** remember that fewer parameters are not an automatic generalization guarantee; inspect data size, learning rate, adapter scaling, and regularization.

### 17.3 Continual learner protects the past by refusing the future

**Symptom:** old-task retention is high but new-task acquisition is weak.

**Diagnostic:** report both final average accuracy and current-task learning curves; sweep the replay or regularization strength.

### 17.4 Replay buffer is unrepresentative

**Symptom:** common old cases survive while rare skills disappear.

**Diagnostic:** evaluate per domain/class and compare random reservoir sampling with stratified or importance-aware selection.

### 17.5 Test-time objective is misaligned

**Symptom:** auxiliary loss decreases but task accuracy falls.

**Diagnostic:** on source validation data, measure the correlation between one-step auxiliary-loss improvement and main-task improvement under simulated shifts.

### 17.6 Online adaptation drifts

**Symptom:** results depend strongly on test-example order or collapse late in the stream.

**Diagnostic:** compare episodic reset, periodic reset, sliding-window adaptation, and frozen-source anchors.

### 17.7 Evaluation leaks labels or future data

**Symptom:** adaptation uses information unavailable at the moment a prediction should be made.

**Diagnostic:** write the temporal protocol explicitly: what is observed, when the update occurs, and when the prediction is scored.

## 18. A practical experiment matrix

Hold constant:

- the pretrained checkpoint;
- task order and data budget;
- optimizer step budget;
- evaluation prompts/examples;
- model-selection information allowed at each time.

Compare:

1. sequential full fine-tuning;
2. one shared sequential LoRA adapter;
3. one LoRA adapter per task;
4. shared LoRA plus replay;
5. shared LoRA plus EWC;
6. episodic test-time adaptation;
7. online test-time adaptation.

Measure:

| Axis | Metrics |
| --- | --- |
| New learning | current-task accuracy/loss, adaptation speed |
| Retention | old-task accuracy, average forgetting |
| Transfer | forward and backward transfer |
| Efficiency | trainable parameters, optimizer memory, update latency |
| Deployment stability | order sensitivity, reset frequency, collapse rate |

## 19. Connections to earlier lectures

`[CONNECTION TO LECTURE 3]` LoRA is inserted into Transformer linear maps such as attention and feed-forward projections. Its effect depends on where those matrices participate in the architecture.

`[CONNECTION TO LECTURE 7]` Low-rank parameterization changes optimization geometry. The gradients of $A$ and $B$ are not the same as projecting a full $\Delta W$ gradient after every step.

`[CONNECTION TO LECTURE 12]` SFT, DPO, and RLHF can all optimize LoRA parameters instead of the full model. LoRA changes the parameterization, not the semantic meaning of those objectives.

`[CONNECTION TO LECTURE 14]` Learning without Forgetting uses distillation from the previous model. Continual learning converts teacher–student matching into a preservation mechanism over time.

## 20. Open questions

1. When does the useful adaptation update actually have low rank, and how should rank vary by layer?
2. Can multiple LoRA adapters be composed without destructive interference?
3. How should a system detect that a new domain deserves a new adapter rather than an update to an existing one?
4. What old examples are most valuable per byte of replay memory?
5. Can parameter importance be estimated reliably for generative behavior rather than classification accuracy?
6. Which label-free test-time objectives correlate with correctness for language generation?
7. How can online test-time adaptation distinguish persistent shift from a transient or adversarial batch?
8. Should test-time updates be rolled back after each user, session, domain, or confidence failure?
9. How do privacy constraints alter the choice between replay, output distillation, and task-specific adapters?
10. Can a model retain plasticity over a long task stream without continually expanding memory or parameters?

## 21. What to verify manually

- Confirm which LoRA target modules and continual-learning methods the student presenters actually emphasized.
- Reproduce the LoRA parameter-count calculation for the dimensions of a model you use.
- Check the initial gradients when one LoRA factor is zero.
- Derive the EWC objective from a diagonal Laplace approximation in your own notation.
- Implement the two-dimensional GEM projection example.
- Compare episodic and online test-time adaptation on a controlled distribution shift.
- Replace broad synthesis statements with your own conclusions after reading the primary papers.

## Source trail

- CS 395T course schedule: <https://noahgol.github.io/teaching/cs395t-f26/>
- Hu et al., *LoRA*: <https://arxiv.org/abs/2106.09685>
- Kirkpatrick et al., *Overcoming catastrophic forgetting*: <https://arxiv.org/abs/1612.00796>
- Lopez-Paz and Ranzato, *Gradient Episodic Memory*: <https://arxiv.org/abs/1706.08840>
- Li and Hoiem, *Learning without Forgetting*: <https://arxiv.org/abs/1606.09282>
- Sun et al., *Test-Time Training with Self-Supervision*: <https://arxiv.org/abs/1909.13231>
- Wang et al., *Tent*: <https://arxiv.org/abs/2006.10726>

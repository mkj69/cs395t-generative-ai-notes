---
title: "Lecture 12 — SFT, RLHF, and DPO"
slug: "lecture-12"
date: "2026-10-05"
updated: "2026-10-12"
status: "draft"
type: "learning-note"
course: "CS 395T"
source: "https://noahgol.github.io/teaching/cs395t-f26/"
summary: "A detailed AI-assisted study scaffold for supervised fine-tuning, reward modeling, PPO-based RLHF, and the derivation and implementation of direct preference optimization."
public: true
tags:
  - supervised fine-tuning
  - preference learning
  - reward modeling
  - RLHF
  - DPO
  - KL regularization
  - post-training
related_research_notes: []
authorship: "ai-assisted"
---

> **AI-assisted study scaffold, not a lecture transcript.** The course schedule lists Lecture 12 as a student-presentation session on SFT, RLHF, and DPO, but it does not publish a unified handout for this lecture. This note therefore does not claim to reconstruct what the presenters said. It develops the named topics from the primary papers listed below and marks derivations and synthesis explicitly. It has not yet been manually verified or rewritten by the author.

## Source boundary

`[DIRECT · COURSE SCHEDULE]` The official schedule names the October 5 session **“Post-training I (Student Presentations #1): SFT, RLHF, and DPO.”** It does not link lecture notes or slides for that session: <https://noahgol.github.io/teaching/cs395t-f26/>.

`[DIRECT · PRIMARY SOURCES]` The main technical sources used here are:

1. Ouyang et al. (2022), [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155).
2. Christiano et al. (2017), [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741).
3. Rafailov et al. (2023), [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).

`[SOURCE BOUNDARY]` Statements labeled `[DIRECT]` are attributable to these sources or to the course schedule. `[DERIVATION]` lines work through the mathematics. `[EXPLANATION]` and `[SYNTHESIS]` organize the ideas but are not reported classroom claims.

`[SYNTHESIS]` The central problem is that next-token pretraining and desired assistant behavior are different objectives. Post-training supplies a narrower behavioral signal:

- demonstrations say **what response to imitate**;
- comparisons say **which of two responses is preferred**;
- RLHF turns those comparisons into a reward and optimizes it;
- DPO eliminates the explicit reward-model-plus-RL loop by fitting a policy directly to the comparisons under a specific model of preferences and KL-regularized optimization.

## Reading map

| Part | Data | Learned object | Main question |
| --- | --- | --- | --- |
| SFT | Prompt-response demonstrations $(x,y^*)$ | Policy $\pi_\theta(y\mid x)$ | Can the model imitate desired behavior? |
| Reward modeling | Preference triples $(x,y_w,y_l)$ | Scalar score $r_\phi(x,y)$ | Can pairwise judgments be generalized to unseen responses? |
| PPO-based RLHF | Prompts and policy samples | Policy plus value model | Can expected learned reward be increased without drifting too far? |
| DPO | Static preference triples | Policy relative to a reference | Can the KL-regularized optimum be fitted without explicit reward modeling or online RL? |

## 1. Post-training changes the behavioral objective

### 1.1 Pretraining is not instruction following

`[DIRECT · INSTRUCTGPT]` The InstructGPT paper starts from a mismatch: predicting the next token on internet text is not the same objective as following a user's instruction helpfully and safely.

`[EXPLANATION]` A pretrained model can contain useful knowledge and capabilities while assigning substantial probability to responses that are evasive, badly formatted, unhelpful, or inconsistent with the intended role. Post-training does not create a clean separation between “knowledge” and “behavior,” but it supplies examples and comparisons that directly concern behavior on prompts of interest.

### 1.2 Three distributions must be distinguished

Let

- $\mathcal D_{mathrm{pre}}$ be the pretraining text distribution;
- $\mathcal D_{mathrm{prompt}}$ be the prompts on which the assistant will be trained or deployed;
- $\mathcal D_{mathrm{pref}}$ be the distribution of collected human comparisons.

`[SYNTHESIS]` These distributions need not match. A method can fit its preference dataset well and still fail on deployment prompts. Data collection is therefore not an incidental step: it defines which behaviors receive supervision.

## 2. Supervised fine-tuning

### 2.1 Demonstration likelihood

Suppose the demonstration dataset is

$$
\mathcal D_{mathrm{SFT}}
=\{(x^{(i)},y^{(i)})\}_{i=1}^{N},
$$

where $x$ is a prompt and $y=(y_1,\ldots,y_T)$ is a desired response.

The autoregressive policy factorizes as

$$
\pi_\theta(y\mid x)
=\prod_{t=1}^{T}
\pi_\theta(y_t\mid x,y_{<t}).
$$

Maximum likelihood minimizes the response negative log-likelihood

$$
\boxed{
\mathcal L_{mathrm{SFT}}(\theta)
=
-\mathbb E_{(x,y)\sim\mathcal D_{mathrm{SFT}}}
\left[
\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x,y_{<t})
\right].
}
$$

`[EXPLANATION]` Training uses teacher forcing: token $y_t$ is predicted from the demonstrated prefix $y_{<t}$, not from a prefix sampled from the current model.

### 2.2 Which tokens contribute to the loss?

In instruction tuning, the prompt and response are often concatenated into one sequence. A response mask $m_t\in\{0,1\}$ can exclude prompt tokens:

$$
\mathcal L_{mathrm{SFT}}
=
-\frac{
\sum_t m_t\log\pi_\theta(z_t\mid z_{<t})
}{
\sum_t m_t
},
$$

where $z$ denotes the full formatted sequence.

`[SYNTHESIS]` Including or excluding prompt-token loss changes the training objective. “SFT” is not fully specified until the serialization template, masks, normalization, and example weights are known.

### 2.3 Token averaging versus example averaging

Two common batch objectives are

$$
\mathcal L_{mathrm{token}}
=
-\frac{
\sum_i\sum_t m_{i,t}\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})
}{
\sum_i\sum_t m_{i,t}
},
$$

and

$$
\mathcal L_{mathrm{example}}
=
-\frac1B
\sum_{i=1}^{B}
\frac{
\sum_t m_{i,t}\log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})
}{
\sum_t m_{i,t}
}.
$$

`[DERIVATION]` In the first objective, a response with twice as many unmasked tokens contributes roughly twice as much total weight. In the second, every response receives equal total weight regardless of length.

`[SYNTHESIS]` Length normalization is not merely an implementation detail; it changes the empirical distribution being optimized.

### 2.4 Minimal SFT loss

```python
# logits: [batch, sequence, vocabulary]
# labels: next-token ids, with -100 on prompt/padding positions
shifted_logits = logits[:, :-1]
shifted_labels = labels[:, 1:]

loss = cross_entropy(
    shifted_logits.reshape(-1, vocab_size),
    shifted_labels.reshape(-1),
    ignore_index=-100,
)
```

`[CHECK]` A common silent error is an off-by-one mask: the mask must select the labels being predicted after the causal shift, not the input positions before the shift.

### 2.5 What SFT can and cannot infer

`[EXPLANATION]` A demonstration tells the model that one response should have high likelihood. It does not directly specify how that response compares with the many other plausible responses.

`[SYNTHESIS]` SFT is especially natural when a target response can be written. Preference learning becomes attractive when producing a perfect answer is difficult but comparing two candidate answers is easier.

`[LIMIT]` SFT only observes demonstrated prefixes during training. During generation, the model conditions on its own sampled prefixes. Errors can therefore move generation into regions absent from the demonstration data.

## 3. From comparisons to a reward model

### 3.1 Preference data

Let

$$
(x,y_w,y_l)\sim\mathcal D_{mathrm{pref}},
$$

where $y_w$ is preferred to $y_l$ for prompt $x$.

`[DIRECT · INSTRUCTGPT]` The InstructGPT pipeline first collected demonstrations for SFT, then collected rankings of model responses and trained a reward model, and finally optimized the policy against that reward model using PPO.

### 3.2 Bradley-Terry preference model

Assign each response a scalar latent reward $r^*(x,y)$. The Bradley-Terry model assumes

$$
\Pr(y_1\succ y_2\mid x)
=
\frac{e^{r^*(x,y_1)}}
{e^{r^*(x,y_1)}+e^{r^*(x,y_2)}}
=
\sigma\bigl(r^*(x,y_1)-r^*(x,y_2)\bigr),
$$

where

$$
\sigma(z)=\frac1{1+e^{-z}}.
$$

`[EXPLANATION]` The model converts a difference in latent rewards into a preference probability. Equal rewards predict a $50\%$ preference probability; a large positive difference predicts near-certain preference for the first response.

### 3.3 Reward-model likelihood

Parameterize a scalar reward $r_\phi(x,y)$. Maximum likelihood gives

$$
\boxed{
\mathcal L_{mathrm{RM}}(\phi)
=
-\mathbb E_{(x,y_w,y_l)\sim\mathcal D_{mathrm{pref}}}
\left[
\log\sigma
\bigl(r_\phi(x,y_w)-r_\phi(x,y_l)\bigr)
\right].
}
$$

`[DERIVATION]` Let $d=r_\phi(x,y_w)-r_\phi(x,y_l)$. For one pair,

$$
\ell(d)=-\log\sigma(d)=\log(1+e^{-d}).
$$

Its derivative is

$$
\frac{d\ell}{dd}=-\sigma(-d).
$$

When the model incorrectly gives the loser a larger reward, $d<0$ and $\sigma(-d)$ is large, so the update strongly increases the reward difference. Already-correct, high-margin pairs receive smaller gradients.

### 3.4 Reward is identifiable only up to a prompt-dependent shift

For any function $c(x)$, define

$$
r'(x,y)=r(x,y)+c(x).
$$

Then

$$
r'(x,y_w)-r'(x,y_l)
=
r(x,y_w)-r(x,y_l).
$$

`[CONCLUSION]` Pairwise comparisons cannot identify the absolute reward level for a prompt. They only constrain differences among responses to the same prompt.

`[DIRECT · INSTRUCTGPT]` The InstructGPT implementation normalized the learned reward with a bias before RL. Such a convention selects an origin but does not create new preference information.

### 3.5 Minimal reward-model loss

```python
chosen_reward = reward_model(prompt, chosen_response)
rejected_reward = reward_model(prompt, rejected_response)

reward_margin = chosen_reward - rejected_reward
loss = -logsigmoid(reward_margin).mean()
```

`[CHECK]` Swapping chosen and rejected examples changes the sign of the margin. A reversed convention can train a perfectly consistent model in exactly the wrong direction.

## 4. PPO-based RLHF

### 4.1 The KL-regularized objective

Let $\pi_{\mathrm{ref}}$ be a fixed reference policy, commonly the SFT model. A standard population objective is

$$
\boxed{
\max_{\pi}
\mathbb E_{x\sim\mathcal D_{\mathrm{prompt}}}
\left[
\mathbb E_{y\sim\pi(\cdot\mid x)}[r_\phi(x,y)]
-\beta
D_{\mathrm{KL}}
\bigl(\pi(\cdot\mid x)\|\pi_{\mathrm{ref}}(\cdot\mid x)\bigr)
\right].
}
$$

`[EXPLANATION]` The reward model encourages preferred behavior. The KL term discourages the policy from moving far from a distribution on which the reward model and language model are better understood.

### 4.2 Why the sampled log-ratio is a KL penalty

For a fixed prompt,

$$
D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})
=
\sum_y\pi(y\mid x)
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
$$

Therefore

$$
D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})
=
\mathbb E_{y\sim\pi(\cdot\mid x)}
\left[
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right].
$$

One sampled response can use the regularized scalar return

$$
R(x,y)
=
r_\phi(x,y)
-\beta
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
$$

For an autoregressive model,

$$
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
=
\sum_{t=1}^{T}
\left[
\log\pi(y_t\mid x,y_{<t})
-\log\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})
\right].
$$

`[DIRECT · INSTRUCTGPT]` The InstructGPT implementation used a per-token KL penalty from the SFT model to mitigate over-optimization of the reward model.

### 4.3 Reference policy versus old policy

Lecture 11 introduced two distinct policies:

- $\pi_{\mathrm{old}}$ collected the current PPO batch and changes after each outer iteration;
- $\pi_{\mathrm{ref}}$ anchors the entire post-training run and usually remains fixed.

`[SYNTHESIS]` PPO clipping controls one update relative to $\pi_{\mathrm{old}}$. The reference KL controls cumulative behavioral drift relative to $\pi_{\mathrm{ref}}$. Confusing them obscures both the derivation and implementation.

### 4.4 Why not maximize learned reward without constraint?

`[SYNTHESIS]` The reward model is fitted on a finite comparison distribution. If policy optimization finds responses far outside that distribution, a high predicted reward may reflect model error rather than genuine preference.

The KL term does not prove that reward exploitation is impossible. It limits how aggressively the optimizer searches unfamiliar regions.

### 4.5 Where PPO enters

The regularized response reward is not differentiable through sampled discrete tokens in the ordinary pathwise sense. Policy gradients estimate how changing token probabilities changes expected reward. PPO then reuses each on-policy batch for multiple bounded surrogate updates, as derived in Lecture 11.

`[SYNTHESIS]` Classical language-model RLHF contains several learned components:

1. an SFT policy;
2. a reward model;
3. an RL policy;
4. usually a value model or value head;
5. a fixed reference policy;
6. a temporarily frozen old policy or stored old log-probabilities.

This complexity is one motivation for direct preference methods.

## 5. DPO starts from the same regularized control problem

### 5.1 Solve for the optimal policy at one prompt

Fix prompt $x$ and assume a general reward $r(x,y)$. Consider

$$
\max_{\pi}
\sum_y\pi(y\mid x)r(x,y)
-\beta
\sum_y\pi(y\mid x)
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)},
$$

subject to

$$
\sum_y\pi(y\mid x)=1.
$$

Introduce multiplier $\lambda$:

$$
\begin{aligned}
\mathcal J(\pi,\lambda)
=
&\sum_y\pi(y\mid x)r(x,y)\\
&-\beta\sum_y\pi(y\mid x)
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\\
&+\lambda\left(\sum_y\pi(y\mid x)-1\right).
\end{aligned}
$$

### 5.2 Stationarity calculation

`[DERIVATION · Step 1]` Differentiate with respect to one probability $\pi(y\mid x)$:

$$
\frac{\partial\mathcal J}{\partial\pi(y\mid x)}
=
r(x,y)
-\beta\left(
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+1
\right)
+\lambda.
$$

`[DERIVATION · Step 2]` Set the derivative to zero and rearrange:

$$
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
=
\frac{r(x,y)}{\beta}
+\frac{\lambda}{\beta}-1.
$$

`[DERIVATION · Step 3]` Exponentiate:

$$
\pi(y\mid x)
=
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right)
\exp\left(\frac{\lambda}{\beta}-1\right).
$$

The final exponential is constant across responses for the same prompt. Normalization gives

$$
\boxed{
\pi_r(y\mid x)
=
\frac1{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right),
}
$$

with

$$
Z(x)
=
\sum_y
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right).
$$

`[EXPLANATION]` The optimum is an exponentially tilted reference policy. Larger reward increases probability; the reference supplies the base measure; $\beta$ controls how strongly reward can tilt it.

### 5.3 Invert the policy-reward relation

Take logs:

$$
\log\pi_r(y\mid x)
=
\log\pi_{\mathrm{ref}}(y\mid x)
+\frac{r(x,y)}{\beta}
-\log Z(x).
$$

Therefore

$$
\boxed{
r(x,y)
=
\beta
\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
}
$$

`[EXPLANATION]` This expresses a reward through the policy it induces. The unknown partition term depends on $x$ but not on which response $y$ is compared.

## 6. The DPO loss

### 6.1 The partition function cancels in comparisons

For two responses to the same prompt,

$$
\begin{aligned}
r(x,y_w)-r(x,y_l)
=
\beta\Biggl[
&\log\frac{\pi_r(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}\\
&-\log\frac{\pi_r(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\Biggr].
\end{aligned}
$$

The two $\beta\log Z(x)$ terms cancel.

`[SYNTHESIS]` This cancellation is the key move. DPO does not need to compute the partition function over all possible responses because preference data only depends on reward differences.

### 6.2 Parameterize the optimal policy directly

Define the policy-relative score

$$
s_\theta(x,y)
:=
\beta
\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
$$

The preference logit for one pair is

$$
z_\theta(x,y_w,y_l)
=
s_\theta(x,y_w)-s_\theta(x,y_l).
$$

Substitute this logit into the Bradley-Terry likelihood:

$$
\boxed{
\mathcal L_{\mathrm{DPO}}(\theta)
=
-\mathbb E_{(x,y_w,y_l)\sim\mathcal D_{mathrm{pref}}}
\left[
\log\sigma\bigl(z_\theta(x,y_w,y_l)\bigr)
\right].
}
$$

Expanded,

$$
\begin{aligned}
z_\theta
=\beta\Biggl[
&\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}\\
&-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\Biggr].
\end{aligned}
$$

`[CONCLUSION]` DPO uses a binary classification loss over response pairs, but the classifier logit is constrained to be a difference of policy-versus-reference log-probability ratios.

### 6.3 Why the reference policy does not cancel

The prompt-dependent partition term cancels, but the response-dependent reference probabilities generally do not:

$$
z_\theta
=
\beta
\left[
\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x)
\right]
-\beta
\left[
\log\pi_{\mathrm{ref}}(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_l\mid x)
\right].
$$

`[EXPLANATION]` DPO is not simply “make the winner more likely than the loser.” It learns how much more the new policy should favor the winner **relative to the reference policy's existing preference**.

## 7. Understanding the DPO gradient

For one pair,

$$
\ell(\theta)=-\log\sigma(z_\theta).
$$

Since

$$
\frac{d}{dz}[-\log\sigma(z)]= -\sigma(-z),
$$

and the reference policy is fixed,

$$
\nabla_\theta z_\theta
=
\beta
\left[
\nabla_\theta\log\pi_\theta(y_w\mid x)
-\nabla_\theta\log\pi_\theta(y_l\mid x)
\right].
$$

Therefore

$$
\boxed{
\nabla_\theta\ell(\theta)
=
-\beta\sigma(-z_\theta)
\left[
\nabla_\theta\log\pi_\theta(y_w\mid x)
-\nabla_\theta\log\pi_\theta(y_l\mid x)
\right].
}
$$

Gradient descent follows the negative gradient, so it:

- increases the chosen response log-probability;
- decreases the rejected response log-probability;
- applies a large weight $\sigma(-z_\theta)$ when the current relative policy score ranks the pair incorrectly or with a small margin;
- applies a small weight when the chosen response already has a large relative margin.

`[SYNTHESIS]` DPO behaves like a reference-corrected, adaptively weighted contrast between two sequence log-likelihoods.

## 8. A small numerical example

Suppose, for one prompt,

$$
\pi_{\mathrm{ref}}(y_w\mid x)=0.30,
\qquad
\pi_{\mathrm{ref}}(y_l\mid x)=0.20,
$$

while the trainable policy has

$$
\pi_\theta(y_w\mid x)=0.36,
\qquad
\pi_\theta(y_l\mid x)=0.18.
$$

The winner and loser log-ratios are

$$
\log\frac{0.36}{0.30}=\log 1.2\approx0.182,
$$

$$
\log\frac{0.18}{0.20}=\log 0.9\approx-0.105.
$$

Their difference is about $0.287$. If $\beta=0.1$, then

$$
z\approx0.0287,
\qquad
\sigma(z)\approx0.507,
$$

and the pair loss is

$$
-\log\sigma(z)\approx0.679.
$$

`[EXPLANATION]` The trainable policy has moved in the preferred direction relative to the reference, but the scaled margin is still small, so the pair still receives a substantial gradient.

## 9. Minimal DPO implementation

### 9.1 Sequence log-probabilities

For a response mask $m_t$,

$$
\log\pi_\theta(y\mid x)
=
\sum_t
m_t\log\pi_\theta(y_t\mid x,y_{<t}).
$$

Use the same serialization, causal shift, response mask, and length convention for the policy and reference model.

### 9.2 Pairwise loss

```python
def masked_sequence_logp(logits, labels, response_mask):
    token_logp = log_softmax(logits[:, :-1], dim=-1)
    target = labels[:, 1:]
    mask = response_mask[:, 1:]
    selected = gather(token_logp, index=target)
    return (selected * mask).sum(dim=-1)


chosen_logp = masked_sequence_logp(
    policy_chosen_logits, chosen_labels, chosen_mask
)
rejected_logp = masked_sequence_logp(
    policy_rejected_logits, rejected_labels, rejected_mask
)

with no_grad():
    ref_chosen_logp = masked_sequence_logp(
        reference_chosen_logits, chosen_labels, chosen_mask
    )
    ref_rejected_logp = masked_sequence_logp(
        reference_rejected_logits, rejected_labels, rejected_mask
    )

policy_margin = chosen_logp - rejected_logp
reference_margin = ref_chosen_logp - ref_rejected_logp
dpo_logit = beta * (policy_margin - reference_margin)

loss = -logsigmoid(dpo_logit).mean()
```

`[CHECK]` The reference model is frozen. Gradients through its log-probabilities waste memory at best and change the intended objective at worst.

`[CHECK]` Both responses must share the same prompt formatting. If system messages or separators differ, the loss may learn formatting artifacts instead of the labeled preference.

## 10. SFT, reward-model RLHF, and DPO compared

| Property | SFT | Reward model + PPO | DPO |
| --- | --- | --- | --- |
| Training signal | One desired response | Pairwise preferences converted to scalar reward | Pairwise preferences directly |
| Sampling during optimization | No | Yes, from the current policy | No, for the basic offline algorithm |
| Explicit reward model | No | Yes | No standalone model |
| Learned value model | No | Usually | No |
| Fixed reference policy | Optional | Usually | Yes in the standard loss |
| On-policy RL loop | No | Yes | No |
| Core objective | Demonstration likelihood | Expected reward minus KL | Preference likelihood under policy-ratio reward parameterization |
| Main convenience | Simple and stable | Can optimize a learned scalar objective on fresh samples | Simple supervised-style optimization on preference pairs |
| Main limitation | Demonstrations are expensive and specify only one target | Complex and vulnerable to reward-model exploitation | Depends on static preference coverage and modeling assumptions |

`[SYNTHESIS]` DPO removes components from the training pipeline, but it does not remove assumptions. It relies on the preference model, reference policy, offline pair distribution, and the relationship between KL-regularized reward maximization and the fitted policy.

## 11. Important subtleties

### 11.1 DPO is offline with respect to its pair dataset

Basic DPO does not sample fresh responses during each gradient update. If the optimized policy moves into regions poorly represented by the preference pairs, the dataset does not automatically expand to cover them.

### 11.2 The meaning of beta

In the regularized control objective, larger $\beta$ imposes a stronger KL penalty. The closed-form optimum

$$
\pi_r\propto\pi_{\mathrm{ref}}e^{r/\beta}
$$

then stays closer to the reference for the same reward scale.

In the DPO likelihood, $\beta$ multiplies the policy log-ratio margin. Its effect cannot be interpreted independently of reward scale, optimization, data, and model capacity.

### 11.3 Sequence length changes log-probability scale

A sequence log-probability is a sum over tokens. Longer responses often have more negative raw log-probability simply because they contain more factors.

`[SYNTHESIS]` Replacing sums with per-token averages changes the DPO objective. Either convention can be implemented, but it should not be changed silently.

### 11.4 Reference support matters

The derivation assumes the relevant response has nonzero reference probability. Softmax language models assign positive mathematical probability to every vocabulary token, but finite precision, masking, truncation, and incompatible tokenization or templates can still make practical ratios invalid.

### 11.5 Preference labels need not define a universal reward

Human judgments may be noisy, context-dependent, or mutually inconsistent. Bradley-Terry is a statistical model of comparisons, not a proof that every preference dataset arises from one globally coherent scalar reward.

### 11.6 Low training loss is not sufficient evaluation

A policy can separate training pairs without improving responses drawn from the deployment distribution. Evaluation should sample from the trained policy and assess held-out prompts, not only re-score stored pairs.

## 12. Relationship to Lectures 11 and 13

### 12.1 From Lecture 11

Lecture 11 supplies the policy-gradient and PPO machinery used in classical RLHF. Its distinction between $\pi_{\mathrm{old}}$ and $\pi_{\mathrm{ref}}$ prevents a common conceptual error in post-training implementations.

### 12.2 Toward Lecture 13

Lecture 13 studies verifiable rewards and group-based estimators. Compared with the preference pipeline here:

- RLHF learns a reward from human comparisons;
- DPO trains directly on offline comparisons;
- RLVR receives a programmatic or otherwise verifiable outcome reward;
- GRPO-like methods estimate relative advantages from several fresh responses to the same prompt.

`[SYNTHESIS]` These methods differ along two independent axes:

1. **Where does the supervision come from?** Demonstration, human preference, learned reward, or verifier.
2. **How is the policy updated?** Maximum likelihood, policy gradient, clipped surrogate, or direct preference likelihood.

## 13. A diagnostic checklist

### SFT

- Are only intended response tokens included in the loss?
- Is the causal shift aligned with the mask?
- Are examples weighted per token or per response?
- Does evaluation use generated responses rather than teacher-forced loss alone?

### Reward modeling

- Are chosen and rejected responses consistently ordered?
- Are train/test splits grouped by prompt to avoid correlated leakage?
- Is calibration checked, not only pairwise accuracy?
- Does the reward model generalize to samples from later policies?

### PPO-based RLHF

- Are old-policy and reference-policy log-probabilities stored separately?
- Is the KL measured with the same token masks and serialization as the policy loss?
- Are reward, KL, value error, entropy, and sample quality monitored together?
- Is the policy being optimized outside the reward model's data distribution?

### DPO

- Is the reference model frozen?
- Are policy and reference sequence log-probabilities computed identically?
- Is the winner-loser sign verified with a tiny synthetic batch?
- Is the chosen length normalization explicitly documented?
- Are trained-policy generations evaluated on held-out prompts?

## 14. Compact derivation chain

1. **SFT imitates demonstrations**

   $$
   \mathcal L_{\mathrm{SFT}}
   =-\mathbb E\log\pi_\theta(y^*\mid x).
   $$

2. **Reward modeling fits preference differences**

   $$
   \mathcal L_{\mathrm{RM}}
   =-\mathbb E\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)).
   $$

3. **RLHF optimizes learned reward under a KL anchor**

   $$
   \max_\pi
   \mathbb E[r_\phi]
   -\beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}).
   $$

4. **The exact regularized optimum is an exponential tilt**

   $$
   \pi_r(y\mid x)
   =\frac1{Z(x)}
   \pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}.
   $$

5. **Invert the relation**

   $$
   r(x,y)
   =\beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
   +\beta\log Z(x).
   $$

6. **Pairwise differences cancel the partition function**

   $$
   r_w-r_l
   =\beta\left[
   \log\frac{\pi_r(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
   -\log\frac{\pi_r(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
   \right].
   $$

7. **DPO fits the policy directly**

   $$
   \mathcal L_{\mathrm{DPO}}
   =-\mathbb E\log\sigma(z_\theta).
   $$

## 15. Open questions for manual study

- `[OPEN]` Which behaviors are easier to demonstrate than to compare, and which are easier to compare than to demonstrate?
- `[OPEN]` When does a scalar reward model fail to represent heterogeneous or intransitive preferences?
- `[OPEN]` How quickly does an offline DPO dataset become stale as the policy moves away from the model that generated its responses?
- `[OPEN]` Can active data collection choose comparison pairs that are maximally informative rather than merely high-scoring?
- `[OPEN]` How should response length enter preference objectives when labelers have length preferences of their own?
- `[OPEN]` What diagnostics distinguish genuine preference improvement from exploitation of annotator or judge artifacts?
- `[OPEN]` Under what assumptions does lower DPO loss imply improvement in expected human preference for fresh policy samples?
- `[OPEN]` How should uncertainty or disagreement among labelers be represented instead of collapsed into one binary label?

## 16. Source trail

- `[DIRECT]` [CS 395T schedule](https://noahgol.github.io/teaching/cs395t-f26/) — identifies the session topic and presentation format.
- `[DIRECT]` Ouyang et al. (2022), [InstructGPT](https://arxiv.org/abs/2203.02155) — demonstration SFT, comparison-based reward modeling, and PPO with an SFT-reference KL penalty.
- `[DIRECT]` Christiano et al. (2017), [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) — learning behavioral rewards from human comparisons.
- `[DIRECT]` Rafailov et al. (2023), [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — closed-form KL-regularized optimum, reward-policy reparameterization, and the DPO loss.

## Revision log

- **2026-10-12 — AI-assisted first draft.** Added a source-bounded study scaffold; derived SFT, reward-model, RLHF, and DPO objectives; included the DPO gradient, a numerical example, implementation skeletons, diagnostics, and links to Lectures 11 and 13. Manual verification remains pending.

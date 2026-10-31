---
title: 'Lecture 21: Diffusion V: Discrete Diffusion Models'
slug: lecture-21
date: '2026-11-04'
updated: '2026-10-31'
status: draft
type: learning-note
course: CS 395T
source: https://noahgol.github.io/teaching/cs395t-f26/
summary: Transition matrices make discrete noising tractable. Small examples derive
  reverse posteriors, absorbing masks, training losses, and the dependence errors
  of parallel token generation.
public: true
language: en
tags:
- Diffusion
- discrete diffusion
- D3PM
- masked language modeling
related_research_notes: []
authorship: ai-assisted
---

> **AI-assisted study note awaiting classroom materials.** As of October 31, 2026, the course schedule lists the topic without a public slide deck or handout. This note develops the topic from public primary sources and independent derivations. It does not reconstruct the student presentation. References are supplementary sources, not a confirmed assigned reading list. The page date is the scheduled class date.

`[DIRECT]` identifies a source-supported statement. `[DERIVATION]` identifies mathematical reasoning. `[SYNTHESIS]` identifies an explanatory connection. Constructed numerical examples are not reported experiments.

[Diffusion reading guide](diffusion-unit.html) · [Separate bilingual source readings](diffusion-unit.html#bilingual-source-readings)

## Why tokens need different probability objects

`[SYNTHESIS]` A continuous image state admits small real-valued perturbations. A token belongs to a finite set of categories, and subtracting token IDs has no intrinsic semantic meaning. Discrete diffusion retains the idea of a controlled corruption process followed by learned recovery, but replaces Gaussian perturbations with random transitions between categories.

Discrete states can evolve in continuous time. Those are separate modeling choices. This note starts with time steps $t=0,1,\ldots,T$ and later introduces continuous-time rate matrices. Time $t=0$ always denotes clean data.

## A transition matrix specifies forward corruption

`[DERIVATION]` Define $Q_t[i,j]=q(X_t=j\mid X_{t-1}=i)$. A row identifies the starting state and a column the destination. Each row is nonnegative and sums to 1. Using row vectors for probabilities,

$$\overline Q_t=Q_1Q_2\cdots Q_t,\qquad q(X_t=j\mid X_0=i)=\overline Q_t[i,j].$$

Matrix multiplication sums the probabilities of possible intermediate paths. If $\overline Q_t$ is efficient to compute, training can sample the state at any time directly instead of simulating every forward step. Dense matrices over large vocabularies can be expensive; structured transitions change that computational cost.

`[DIRECT]` D3PM represents discrete corruption with transition matrices and studies structures including uniform, local, and absorbing transitions. It combines a variational objective with an auxiliary denoising cross-entropy loss. Source: [Austin et al., v1, Section 3](https://arxiv.org/html/2107.03006v1). The small matrices below are constructed examples rather than reproduced experiments.

## A two-state example that can be checked by hand

`[DERIVATION]` Let the states be A and B, with

$$Q=\begin{pmatrix}0.8&0.2\\0.3&0.7\end{pmatrix}.$$

Starting at A, the probability of returning to A after two steps is $0.8\times0.8+0.2\times0.3=0.70$. The probability of reaching B is $0.30$. Matrix multiplication already sums over the unknown intermediate state.

Now suppose $X_0=A$ and $X_2=B$. The two possible paths have unnormalized probabilities

$$P(A\to A\to B)=0.8\times0.2=0.16,$$

$$P(A\to B\to B)=0.2\times0.7=0.14.$$

Thus $q(X_1=A\mid X_2=B,X_0=A)=8/15$, and the probability of $X_1=B$ is $7/15$. Transposing $Q$ would not produce this reverse posterior. Reverse inference must account for the probability mass at the preceding time.

## Bayes' rule gives the reverse posterior

`[DERIVATION]` Whenever the denominator is positive,

$$q(X_{t-1}=k\mid X_t=j,X_0=i)=\frac{\overline Q_{t-1}[i,k]Q_t[k,j]}{\overline Q_t[i,j]}.$$

The numerator is the probability of first reaching $k$ and then moving to $j$. The denominator sums that quantity over every possible $k$. During training, $X_0$ is known, so the posterior is computable. During generation, $X_0$ is unknown. A model must supply reverse probabilities or predict a distribution over clean categories.

The exact marginalization identity is

$$q(X_{t-1}=k\mid X_t=j)=\sum_i q(X_{t-1}=k\mid X_t=j,X_0=i)q(X_0=i\mid X_t=j).$$

It explains why clean-category prediction can help. Particular models can use different parameterizations or normalization rules, so this identity should not automatically be read as the literal implementation of every D3PM model.

For text, forward noise may be applied independently at each position. A reverse prediction at one position nevertheless usually depends on the entire noisy sequence. Independent forward corruption does not imply independent language tokens.

## An absorbing mask remains masked in the forward process

`[DERIVATION]` Add a special state M. At each step, a normal token $a$ changes to M with probability $\beta_t$ and otherwise remains $a$. A state already equal to M stays there. The cumulative survival probability is

$$\overline\alpha_t=\prod_{r=1}^t(1-\beta_r).$$

Given $X_0=a$, the probabilities are $q(X_t=a\mid a)=\overline\alpha_t$ and $q(X_t=M\mid a)=1-\overline\alpha_t$. A forward token can be masked, but it does not directly turn into a different ordinary token.

For $s<t$, conditional on $X_t=M$ and $X_0=a$,

$$q(X_s=a\mid X_t=M,X_0=a)=\frac{\overline\alpha_s-\overline\alpha_t}{1-\overline\alpha_t},$$

$$q(X_s=M\mid X_t=M,X_0=a)=\frac{1-\overline\alpha_s}{1-\overline\alpha_t}.$$

For $\overline\alpha_s=0.8$ and $\overline\alpha_t=0.5$, the probability of revealing the token in this reverse step is 0.6. The probability of retaining the mask is 0.4. If $X_t$ is already an ordinary token, its earlier state in the exact reverse of this particular process must be the same token.

This explains why revealed tokens are retained in the basic absorbing-mask reverse process. Samplers that remask, replace, or edit already revealed tokens change that process and require their own interpretation.

## What the training objective measures

`[DERIVATION]` For a discrete-time latent Markov chain, a variational upper bound on negative log likelihood contains three parts:

$$\mathcal L=\mathbb E_{X_0}\mathrm{KL}(q(X_T\mid X_0)\|p(X_T))$$

$$+\sum_{t=2}^T\mathbb E_{X_0,X_t}\mathrm{KL}(q(X_{t-1}\mid X_t,X_0)\|p_\theta(X_{t-1}\mid X_t))$$

$$-\mathbb E_{X_0,X_1}\log p_\theta(X_0\mid X_1).$$

The first term measures the mismatch between the terminal corruption distribution and the sampling prior. The middle terms compare the learned reverse transitions with computable posteriors. The final term reconstructs data from the nearest noisy level. Parameterization and time sampling affect the estimator and its weights. Describing the loss only as cross entropy leaves those choices unexplained.

`[DIRECT]` MDLM simplifies an absorbing-mask objective through structured output constraints and develops corresponding sampling methods. Source: [Sahoo et al., v1, Sections 3 and 4](https://arxiv.org/html/2406.07524v1).

`[DERIVATION]` In the single-token absorbing example, an already revealed token has no remaining category uncertainty. For a masked token, the reverse-step category loss can be written as the reveal probability times the negative log probability assigned to the original category:

$$\frac{\overline\alpha_s-\overline\alpha_t}{1-\overline\alpha_t}\left[-\log r_\theta(a\mid X_t)\right].$$

This contribution is conditional on the current state being masked. Averaging over current states also includes the masking probability. The expression explains how language-modeling losses across masking levels arise. It also explains why an arbitrary uniformly masked cross-entropy objective need not equal the same likelihood bound.

## Generating from masks requires joint dependencies

`[SYNTHESIS]` If the terminal corruption distribution is fully masked, reverse generation starts from an all-MASK sequence. The model predicts category distributions from the current context, a sampling rule reveals some positions, and predictions are then recomputed. Later rounds can use previously revealed tokens. Predicting several positions in parallel reduces sequential rounds, but does not automatically preserve their joint relationships.

Consider a constructed dataset containing only AA and BB, each with probability one half. From an all-MASK input, the correct marginal at each position assigns equal probability to A and B. Independent simultaneous sampling generates AA, AB, BA, and BB with probability one quarter each. Half the samples lie outside the data support. Revealing one position and conditioning the other prediction on it can preserve the dependency.

The example shows that accurate marginals do not guarantee accurate joint sampling. It also explains why large reverse steps can lose dependence information.

For a sequence of length $L$ and $K$ sampling rounds, the inequality $K<L$ alone does not establish faster generation than an autoregressive model. Per-round computation, caching, batch size, and hardware affect runtime. Quality comparisons also depend on token counts and compute budgets.

## Continuous time can still have discrete states

`[DERIVATION]` A continuous-time finite-state process uses a rate matrix $R(t)$. Off-diagonal entries are nonnegative and each row sums to zero. For a small interval,

$$P(X_{t+h}=j\mid X_t=i)=\delta_{ij}+hR_{ij}(t)+o(h).$$

The states remain discrete categories. When $p_t(j)>0$, the reverse-time jump rate from $j$ to $i$ at the corresponding forward time involves $p_t(i)R_{ij}(t)/p_t(j)$. This compares incoming probability flow with current probability mass. It parallels the need for density information in continuous diffusion without taking an ordinary spatial derivative of token IDs.

## Checks and terminology

- In the example matrix, the two-step probability from A to B is 0.30.
- A transposed forward matrix generally does not define the reverse chain.
- Forward absorption determines the retention rule of its exact reverse process.
- Correct single-position marginals do not replace a correct joint sequence distribution.

A **categorical distribution** assigns probabilities to finitely many categories. A **transition matrix** specifies jump probabilities. An **absorbing state** cannot be left once entered. A **posterior** is a distribution conditioned on an observation. **Masking** replaces a visible token with a special hidden state. A **rate matrix** specifies instantaneous transition intensities. **Ancestral sampling** follows a model's probabilistic factorization to draw successive states.

`[SYNTHESIS]` Continuous and discrete diffusion both use a controlled forward corruption to construct a learning signal for recovery. Their state spaces, probability objects, and sampling approximations differ.

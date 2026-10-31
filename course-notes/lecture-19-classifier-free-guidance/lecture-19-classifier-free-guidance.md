---
title: 'Lecture 19: Diffusion III: Conditional Generation and Classifier-Free Guidance'
slug: lecture-19
date: '2026-10-28'
updated: '2026-10-31'
status: draft
type: learning-note
course: CS 395T
source: https://noahgol.github.io/teaching/cs395t-f26/
summary: Bayes’ rule explains the conditional score direction. Worked examples clarify
  guidance conventions, condition dropout, and amplification of prediction errors.
public: true
language: en
tags:
- Diffusion
- conditional generation
- classifier-free guidance
related_research_notes: []
authorship: ai-assisted
---

> **AI-assisted study note awaiting classroom materials.** As of October 31, 2026, the course schedule lists the topic without a public slide deck or handout. This note develops the topic from public primary sources and independent derivations. It does not reconstruct the student presentation. References are supplementary sources, not a confirmed assigned reading list. The page date is the scheduled class date.

`[DIRECT]` identifies a source-supported statement. `[DERIVATION]` identifies mathematical reasoning. `[SYNTHESIS]` identifies an explanatory connection. Constructed numerical examples are not reported experiments.

[Diffusion reading guide](diffusion-unit.html) · [Separate bilingual source readings](diffusion-unit.html#bilingual-source-readings)

## From unconditional generation to generation with a condition

`[SYNTHESIS]` An unconditional model generates samples from an overall data distribution. A conditional model also receives information $c$, such as a class label or text. With the same initial noise, changing the condition should change the semantic region reached by the sample. The target distribution becomes $p_t(x\mid c)$.

The central questions concern the difference between two denoising predictions. Why does that difference encode condition-specific information? Why does amplifying it change generation? What errors can amplification introduce?

## Bayes' rule identifies a conditional direction

`[DERIVATION]` Where the densities are positive and differentiable,

$$p_t(x\mid c)=\frac{p_t(c\mid x)p_t(x)}{p(c)}.$$

Taking logarithms and differentiating with respect to $x$ removes $p(c)$, which does not depend on $x$:

$$s_c(x,t)=s_u(x,t)+\nabla_x\log p_t(c\mid x),$$

where $s_c=\nabla_x\log p_t(x\mid c)$ and $s_u=\nabla_x\log p_t(x)$. Therefore,

$$s_c-s_u=\nabla_x\log p_t(c\mid x).$$

The right-hand side points toward increasing confidence in condition $c$ at the current noisy state. Classifier guidance estimates that direction with a classifier. Classifier-free guidance approximates it through the difference between conditional and unconditional generative predictions.

A classifier used this way must account for the noise level $t$. A classifier trained only on clean images may provide unsuitable gradients on highly perturbed states. Bayes' identity is exact, while the difference between neural predictions still contains estimation error.

## Two guidance conventions differ by one

`[DIRECT]` Ho and Salimans jointly train conditional and unconditional denoising predictions and combine them during sampling. Their notation uses $(1+w)\epsilon_c-w\epsilon_u$. Source: [Classifier-Free Diffusion Guidance, v1, Section 3.2](https://arxiv.org/html/2207.12598v1).

`[DERIVATION]` This note uses $\gamma$ to make the unconditional baseline explicit:

$$\widehat s_\gamma=\widehat s_u+\gamma(\widehat s_c-\widehat s_u).$$

| Value | Interpretation |
| --- | --- |
| $\gamma=0$ | Unconditional prediction |
| $\gamma=1$ | Ordinary conditional prediction |
| $\gamma>1$ | Extrapolation beyond the conditional prediction |

The conventions satisfy $\gamma=1+w$. A guidance value of zero can therefore mean unconditional sampling in one implementation and ordinary conditional sampling in another. The formula determines the meaning.

For a constructed one-dimensional example, let the unconditional output be 2 and the conditional output be 5. Then $\gamma=1$ gives 5, while $\gamma=2$ gives 8. The conditional difference of 3 has been amplified to 6. This is extrapolation; when $\gamma>1$, the coefficient $1-\gamma$ on the unconditional output is negative.

## Why noise predictions use the same combination

`[DERIVATION]` For Gaussian perturbations, both branches obey $s=-\epsilon/\sigma_t$ with the same $\sigma_t$. Consequently,

$$\widehat\epsilon_\gamma=\widehat\epsilon_u+\gamma(\widehat\epsilon_c-\widehat\epsilon_u)$$

maps to the corresponding score combination. The negative sign does not change the guidance coefficient because it applies to both branches.

For another prediction parameterization, the two branches must share the same affine conversion to the desired score or velocity. Mixing one branch's noise prediction with another branch's clean-sample prediction does not preserve this identity. The sampler must also interpret the output in the correct parameterization.

## Why training randomly drops the condition

`[SYNTHESIS]` Starting with a training pair $(x_0,c)$, sample $t,\epsilon$ and form $x_t=\alpha_tx_0+\sigma_t\epsilon$. Independently replace the condition with $\varnothing$ with probability $p_{\mathrm{drop}}$. The denoising objective is

$$\mathbb E\|\epsilon_\theta(x_t,t,\widetilde c)-\epsilon\|^2.$$

The same network then receives both conditioned and empty-condition inputs. At sampling time, the two predictions use the same $x_t,t$. They can be evaluated separately or in a combined batch. Classifier-free guidance removes the extra classifier; it still uses a condition and generally requires evaluating both branches.

The empty condition has the representation used during training. An arbitrary all-zero vector need not be equivalent. A negative prompt is also a specific condition, rather than a truly unconditional distribution. Replacing the baseline branch with a negative prompt changes the objects in this interpretation.

## What distribution does guidance imply?

`[DERIVATION]` If both scores are exact, then at a fixed time $t$, assuming a finite normalization constant,

$$s_\gamma=\nabla_x\log\left[p_t(x\mid c)^\gamma p_t(x)^{1-\gamma}\right].$$

This identifies a density associated with the instantaneous field. However, these separately defined densities need not form the marginals of the original forward noising process. The identity alone does not prove that the complete reverse sampler ends at exactly $p_0(x\mid c)^\gamma p_0(x)^{1-\gamma}$. Temporal consistency and sampling dynamics also matter.

A solvable diagnostic example considers two densities at one fixed time: $p_u=\mathcal N(0,4)$ and $p_c=\mathcal N(1,1)$. Their scores are $-x/4$ and $1-x$. Combining them gives

$$s_\gamma(x)=\gamma-\frac{1+3\gamma}{4}x.$$

For $\gamma\ge0$, this is the score of a Gaussian with mean $4\gamma/(1+3\gamma)$ and variance $4/(1+3\gamma)$. At $\gamma=1$, the mean and variance are both 1. At $\gamma=3$, the mean is 1.2 and the variance is 0.4. This example illustrates the algebra of an instantaneous combination. It is not a dataset result or an exact prediction of a full generation process.

## Stronger guidance can amplify errors

`[DERIVATION]` Let $e_c$ and $e_u$ be the conditional and unconditional estimation errors. The combined error is

$$e_\gamma=\gamma e_c+(1-\gamma)e_u.$$

For $\gamma\ge1$, the triangle inequality gives $\|e_\gamma\|\le\gamma\|e_c\|+(\gamma-1)\|e_u\|$. This upper bound does not prove monotonic error growth for every sample. It shows how large coefficients can amplify disagreement between the branches.

`[SYNTHESIS]` Evaluation should consider condition adherence, diversity, visible artifacts, and computational cost together. A single selected image that closely matches a prompt cannot establish better distributional quality. Fixed seeds and solvers help isolate the guidance parameter, while multiple seeds reveal stability.

## Checks and terminology

- Identical conditional and unconditional predictions make the output independent of guidance strength.
- Under $(1+w)\epsilon_c-w\epsilon_u$, ordinary conditional sampling corresponds to $w=0$.
- A time-dependent density does not imply that the condition itself is being noised as part of the state.

**Conditioning** supplies information that changes the target distribution. **Unconditional** means without that specified condition. **Guidance** changes a sampling prediction. **Extrapolation** extends beyond the interval between predictions. **Fidelity** describes conformity to target characteristics. **Mode coverage** describes how many distinct data patterns are represented.

The [Rectified Flows note](lecture-20.html) turns to paths, couplings, and integration steps. Guidance and path design address different choices and can be used together.

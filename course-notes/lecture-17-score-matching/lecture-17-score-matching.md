---
title: 'Lecture 17: Diffusion I: Score Matching and Diffusion Models'
slug: lecture-17
date: '2026-10-21'
updated: '2026-10-31'
status: checked
type: learning-note
course: CS 395T
source: https://noahgol.github.io/teaching/cs395t-f26/
summary: How Gaussian denoising gives a trainable score objective, why reverse dynamics
  need a density correction, and why the probability flow ODE contains a factor of
  one half.
public: true
language: en
tags:
- Diffusion
- Score matching
- Reverse SDE
- Probability flow ODE
related_research_notes: []
authorship: ai-assisted
---

> **AI-assisted study note.** This note develops independent explanations, derivations, and worked examples. It does not record a completed personal reading pass. Definitions and formulas were checked against the relevant [course handout](https://noahgol.github.io/teaching/cs395t-f26/lecture16-17-v2.pdf); human review remains open.

`[DIRECT]` identifies a source-supported statement. `[DERIVATION]` identifies mathematical reasoning. `[SYNTHESIS]` identifies an explanatory connection. Constructed numerical examples are not reported experiments.

[Diffusion reading guide](diffusion-unit.html) · [Separate bilingual source readings](diffusion-unit.html#bilingual-source-readings)

## The problem: learning a direction without knowing the density

`[SYNTHESIS]` A training set provides samples, but usually not a density function. Sampling requires a direction field related to that unknown distribution. Training therefore needs a target that can be computed from samples. Gaussian perturbations and conditional expectations connect these requirements.

Forward time $t$ runs from data toward noise. $X_0$ denotes data, $X_t$ its perturbed state, $s_*(t,x)=\nabla_x\log p_t(x)$ the true score, and $s_\theta$ an estimate. Reverse generation uses the separate time coordinate $r=T-t$.

## A score is a spatial derivative

`[DERIVATION]` For a one-dimensional Gaussian density $p(x)=\mathcal N(m,\sigma^2)$,

$$\log p(x)=C-\frac{(x-m)^2}{2\sigma^2},\qquad \partial_x\log p(x)=-\frac{x-m}{\sigma^2}.$$

The score is negative to the right of the mean and positive to its left. Its units are the inverse of the units of position. For $m=2$, $\sigma^2=4$, and $x=6$, the score is $-1$. This describes the local direction of increasing log density. It is not a numerical rating of sample quality. For a mixture distribution, the score need not point toward a particular training example.

## Reversing the drift alone is insufficient

`[DERIVATION]` Consider the forward SDE $dX_t=f(t,X_t)dt+g(t)dB_t$ with state-independent noise. Under suitable regularity, positive-density, and time-reversal assumptions, the reverse process satisfies

$$dY_r=[-f(T-r,Y_r)+g(T-r)^2s_*(T-r,Y_r)]dr+g(T-r)d\bar B_r.$$

The reverse Brownian motion is defined relative to the reverse filtration. It is not obtained by simply negating the particular noise increments used during training.

The Fokker–Planck equation explains the score coefficient. Since $p\nabla\log p=\nabla p$, the added drift contributes $-g^2\Delta p$ to density evolution. The stochastic term contributes $+g^2\Delta p/2$. Their sum is the required reverse diffusion contribution $-g^2\Delta p/2$.

Matching marginal densities alone does not prove the full time-reversal theorem. Transition laws and uniqueness conditions also matter. This calculation checks the coefficient, rather than replacing those assumptions.

## A computable target for an unknown marginal score

Let $X_t=\alpha_tX_0+\sigma_t\epsilon$, where $\epsilon\sim\mathcal N(0,I)$ is independent of $X_0$. Here $\alpha_t$ is the signal coefficient. It should not be confused with a differently defined single-step parameter in some DDPM notation.

`[DERIVATION]` Conditional on $X_0$, the perturbed state is Gaussian, with score

$$\nabla_x\log p_t(x\mid X_0)=-\frac{x-\alpha_tX_0}{\sigma_t^2},\qquad U=-\frac\epsilon{\sigma_t}.$$

Differentiating the mixture density and applying Bayes' rule gives

$$s_*(t,x)=\mathbb E[U\mid X_t=x,t].$$

This is a posterior average given the current noisy state. It is not an unconditional average of all targets in the dataset.

For any prediction $s_\theta$, squared error decomposes as

$$\mathbb E\|s_\theta-U\|^2=\mathbb E\|s_\theta-s_*\|^2+\mathbb E\|U-s_*\|^2.$$

The final term is independent of the parameters. Training against the computable label $U$ therefore has the same population minimizer as training against the unknown marginal score. Random labels are compatible with a conditional-mean regression target.

## Noise prediction and the denoising mean

`[DERIVATION]` Substituting $s_\theta=-\epsilon_\theta/\sigma_t$ and weighting the score loss by $\sigma_t^2$ gives

$$\mathbb E\|\epsilon_\theta(t,\alpha_tX_0+\sigma_t\epsilon)-\epsilon\|^2.$$

A training example requires a data sample, a time, and a noise sample. The noisy input can be constructed directly, without simulating the entire forward path. Ordinary backpropagation is still required, and generation still uses a numerical sampler.

The same conditional-mean identity yields

$$\mathbb E[\alpha_tX_0\mid X_t=x]=x+\sigma_t^2s_*(t,x).$$

When $\alpha_t\ne0$, division by $\alpha_t$ gives the posterior mean of $X_0$. In a VP model, the right-hand side alone is not generally the mean of the original clean sample.

`[DERIVATION · EXAMPLE]` Take $X_0\sim\mathcal N(0,1)$ and $X=X_0+\epsilon$ with independent unit-variance noise. Then $X\sim\mathcal N(0,2)$, the score is $-x/2$, and the denoising mean is $x-x/2=x/2$. For an observation of 2, the optimal squared-error prediction is 1. It does not recover the exact hidden value of $X_0$ from that particular draw.

## OU, VE, and VP processes

| Process | Drift and noise coefficient | Conditional mean and variance |
| --- | --- | --- |
| OU | $f=-\kappa x$, $g=\sigma$ | $e^{-\kappa t}x_0$; $\frac{\sigma^2}{2\kappa}(1-e^{-2\kappa t})$ |
| VE | $f=0$, $g^2=(\sigma_t^2)'$ | $x_0$; $\sigma_t^2$, assuming zero initial added noise |
| VP | $f=-\beta(t)x/2$, $g=\sqrt{\beta(t)}$ | $\alpha_tx_0$; $1-\alpha_t^2$ |

For VP, $\alpha_t=\exp[-\frac12\int_0^t\beta(s)ds]$. The name “variance preserving” does not imply unit variance for every initial distribution. An initial covariance equal to the identity is preserved by this moment calculation.

## Why the probability flow ODE has a factor of one half

`[DERIVATION]` Dividing the forward SDE's probability current by its density gives

$$v_{\mathrm{PF}}(t,x)=f(t,x)-\frac12g(t)^2s_*(t,x).$$

Under suitable uniqueness conditions, the ODE $\dot X_t=v_{\mathrm{PF}}(t,X_t)$ reproduces the same family of marginal densities. Its joint distribution across times generally differs from the SDE's.

In reverse time, its velocity is $-f+g^2s_*/2$. Deleting the random term from the reverse SDE while retaining the full $g^2s_*$ correction would give a different density evolution.

Generation error includes at least terminal-noise approximation, score estimation, and numerical discretization. Reducing the integration step size directly addresses only the third component.

## A consistency check and reading connections

`[DERIVATION]` The score of a standard Gaussian is $-x$. For stationary VP dynamics, $f=-\beta x/2$, so the probability flow velocity is $-\beta x/2+\beta x/2=0$. A zero velocity preserves the Gaussian density even though particles in the stationary SDE keep moving. This is a compact example of equal marginals with different paths.

**Score** means the spatial gradient of log density. **Denoising** means estimating a clean signal from a perturbed observation. A **posterior mean** is an expectation conditional on the observation. A **marginal** integrates out other variables.

`[DIRECT · SOURCE SCOPE]` Definitions and time conventions were checked against the combined Lecture 16 and 17 handout. The numerical examples and explanatory organization are independent derivations. Related primary research includes [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) and [DDPM](https://arxiv.org/abs/2006.11239). The note does not reproduce all appendix proofs.

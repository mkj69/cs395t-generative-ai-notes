---
title: 'Lecture 16: Diffusion Preparation: Stochastic Calculus'
slug: lecture-16
date: '2026-10-19'
updated: '2026-10-31'
status: checked
type: learning-note
course: CS 395T
source: https://noahgol.github.io/teaching/cs395t-f26/
summary: A derivation of stochastic time scaling, the Itô correction, and density
  evolution, with Brownian motion and Ornstein–Uhlenbeck examples.
public: true
language: en
tags:
- Diffusion
- Brownian motion
- Ito calculus
- Fokker-Planck
related_research_notes: []
authorship: ai-assisted
---

> **AI-assisted study note.** This note develops independent explanations, derivations, and worked examples. It does not record a completed personal reading pass. Definitions and formulas were checked against the relevant [course handout](https://noahgol.github.io/teaching/cs395t-f26/lecture16-17-v2.pdf); human review remains open.

`[DIRECT]` identifies a source-supported statement. `[DERIVATION]` identifies mathematical reasoning. `[SYNTHESIS]` identifies an explanatory connection. Constructed numerical examples are not reported experiments.

[Diffusion reading guide](diffusion-unit.html) · [Separate bilingual source readings](diffusion-unit.html#bilingual-source-readings)

## Starting question: a random path and its distribution

`[SYNTHESIS]` A stochastic process has two related descriptions. One follows the motion of an individual sample. The other follows the distribution of a population of samples. The central question is how a local random update determines the evolution of that distribution.

Let $X_t\in\mathbb R^d$ be the state, $p_t(x)$ its density, $f(t,x)$ the drift, and $g(t)$ a scalar noise coefficient that does not depend on position. Let $B_t$ be standard Brownian motion. The calculations below assume isotropic, state-independent diffusion. Matrix coefficients and position-dependent noise require more general formulas.

## Why the random increment scales with the square root of time

`[DERIVATION]` Divide a time interval of length $T$ into $n=T/h$ steps. Suppose each step adds $\sqrt h Z_k$, where the $Z_k$ are independent standard Gaussian variables. The accumulated variance is

$$\operatorname{Var}\left(\sum_{k=1}^{T/h}\sqrt h Z_k\right)=\frac{T}{h}h=T.$$

If the increment were $hZ_k$, the accumulated variance would be $Th$. It would vanish as the step size approached zero. Brownian increments instead satisfy $B_{t+h}-B_t\sim\mathcal N(0,hI)$, with independent increments over disjoint intervals.

`[DERIVATION · EXAMPLE]` With $T=1$ and $h=0.01$, there are 100 increments. A standard deviation of $0.1$ per increment gives final variance 1. A standard deviation of $0.01$ gives final variance only $0.01$. Refining the time grid should not remove the randomness accumulated over a fixed duration.

## An SDE specifies local motion through stochastic integration

$$dX_t=f(t,X_t)dt+g(t)dB_t.$$

`[DERIVATION]` One Euler–Maruyama step approximates this process by

$$X_{t+h}\approx X_t+h f(t,X_t)+g(t)\sqrt h Z.$$

The conditional mean increment is $hf$, and the conditional covariance increment is $hg^2I$. Brownian paths are almost surely nowhere differentiable. The quantity $dB_t/dt$ cannot be treated as an ordinary smooth function. An SDE has an integral interpretation involving a stochastic integral; a numerical update approximates that process.

For deterministic $g$, the covariance of the stochastic integral is $\int_0^t g(s)^2ds\,I$. It is not $(\int_0^t g(s)ds)^2I$. The latter expression incorrectly treats independent increments at different times as perfectly correlated.

## Where the Itô correction comes from

`[DERIVATION]` The one-dimensional function $F(x)=x^2$ provides a useful check. Ordinary calculus would miss the accumulated contribution of squared noise increments. Since $(\Delta B)^2$ contributes $dt$ in the quadratic-variation limit,

$$d(X_t^2)=2X_t f(t,X_t)dt+2X_tg(t)dB_t+g(t)^2dt.$$

For $X_t=B_t$, taking expectations gives $\frac{d}{dt}\mathbb E[B_t^2]=1$, hence $\mathbb E[B_t^2]=t$. Omitting the last term would incorrectly predict that the variance does not grow.

For a sufficiently smooth function $F(t,x)$, Itô's formula is

$$dF=\left(\partial_tF+f\cdot\nabla F+\frac{g^2}{2}\Delta F\right)dt+g\nabla F\cdot dB_t.$$

Both $\nabla$ and $\Delta$ act on spatial coordinates. The Laplacian sums the second derivatives over those coordinates. Integrability assumptions matter when expectations and integrals are exchanged. A stochastic integral has mean zero only under suitable conditions; that claim cannot be applied to an arbitrary explosive process.

## From test functions to a density equation

`[DERIVATION]` Let $\phi$ be a smooth, compactly supported test function. It assigns a measurement weight to each location. Itô's formula implies

$$\frac d{dt}\mathbb E[\phi(X_t)]=\int\left(f\cdot\nabla\phi+\frac{g^2}{2}\Delta\phi\right)p_t\,dx.$$

Integration by parts transfers the spatial derivatives onto the density and yields

$$\partial_t p_t=-\nabla\cdot(fp_t)+\frac{g^2}{2}\Delta p_t.$$

This is the Fokker–Planck equation. It describes the density of a population, rather than an individual trajectory. Boundary terms must vanish for this derivation. On a bounded domain, the boundary conditions are part of the model.

The probability current is $J=fp_t-\frac{g^2}{2}\nabla p_t$, so the same equation becomes $\partial_t p_t=-\nabla\cdot J$. If no net probability flows across the boundary, integration gives $\frac d{dt}\int p_tdx=0$. This checks conservation of total probability.

## A solvable Ornstein–Uhlenbeck example

`[DERIVATION]` Consider $dX_t=-\kappa X_tdt+\sigma dB_t$ with $\kappa>0$. The drift pulls the state toward zero while noise spreads it out. If the initial mean and variance are $m_0$ and $v_0$, the moment equations give

$$m_t=e^{-\kappa t}m_0,\qquad v_t=e^{-2\kappa t}v_0+\frac{\sigma^2}{2\kappa}(1-e^{-2\kappa t}).$$

For $\kappa=1$, $\sigma=\sqrt2$, and $v_0=4$, the variance is $v_t=1+3e^{-2t}$ and converges to 1. These moment formulas do not imply that every marginal is Gaussian for arbitrary initial data. Gaussian initial conditions give Gaussian marginals; more general initial distributions produce smoothed mixtures.

## Misconceptions and checks

| Claim to examine | A useful check |
| --- | --- |
| A noise increment should scale as $h$ | The total variance over a fixed duration would vanish. |
| Random individual paths prevent conservation of probability | Integrating the Fokker–Planck equation checks total mass. |
| Brownian motion obeys the ordinary chain rule | The formula for $d(B_t^2)$ needs the extra $dt$ term. |
| Equal one-time marginals imply equal stochastic processes | Two-time joint distributions or covariances can differ. |

`[SYNTHESIS]` Probability current and the density gradient are the main tools carried into reverse diffusion. They explain why reversing the process requires a score correction.

## Terminology and sources

**Drift** is the deterministic component of local motion. The **diffusion coefficient** controls the scale of random increments. A **marginal density** describes the state at one time. The **generator** describes the instantaneous change of function expectations; here it is an operator, not an image-generating neural network.

`[DIRECT · SOURCE SCOPE]` The stochastic-calculus sections and related appendices of the combined Lecture 16 and 17 handout support the definitions and notation. That handout also contains later diffusion material. This note does not cover every section of it. [Tang and Zhao's technical tutorial](https://arxiv.org/abs/2402.07487) is a supplementary reference for mathematical conditions.

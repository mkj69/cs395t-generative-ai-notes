---
title: 'Lecture 18: Diffusion II: Flow Matching and Consistency Models'
slug: lecture-18
date: '2026-10-26'
updated: '2026-10-31'
status: checked
type: learning-note
course: CS 395T
source: https://noahgol.github.io/teaching/cs395t-f26/
summary: Conditional expectations connect flow matching to tractable regression. Gaussian
  path examples distinguish a velocity field, its trajectories, and a consistency
  model that predicts endpoints.
public: true
language: en
tags:
- Diffusion
- Flow matching
- Conditional expectation
- Consistency models
related_research_notes: []
authorship: ai-assisted
---

> **AI-assisted study note.** This note develops independent explanations, derivations, and worked examples. It does not record a completed personal reading pass. Definitions and formulas were checked against the relevant [course handout](https://noahgol.github.io/teaching/cs395t-f26/lecture18-v2.pdf); human review remains open.

`[DIRECT]` identifies a source-supported statement. `[DERIVATION]` identifies mathematical reasoning. `[SYNTHESIS]` identifies an explanatory connection. Constructed numerical examples are not reported experiments.

[Diffusion reading guide](diffusion-unit.html) · [Separate bilingual source readings](diffusion-unit.html#bilingual-source-readings)

## The question: local velocity or final destination?

`[SYNTHESIS]` Flow matching learns how the current state moves at the next instant. A consistency model learns where a state on a trajectory ultimately arrives. Both involve ODEs, but the prediction targets differ.

The flow-matching sections use the course convention: $t=0$ is noise and $t=1$ is data. Let $Z\sim\mathcal N(0,I)$ and $X_1\sim P_{data}$. The main SD3 notation uses the opposite direction, so comparisons require a time-coordinate change.

## Two descriptions of a continuous generator

`[DERIVATION]` Define $\dot Y_t=v_\theta(t,Y_t)$. Given the initial noise, an ODE trajectory is deterministic. Continuity and suitable spatial Lipschitz conditions guarantee uniqueness. The flow map transports the initial distribution to $p_t$.

If a velocity $u$ transports the density, then $\partial_tp_t=-\nabla\cdot(p_tu_t)$. Applying the chain rule along a trajectory gives

$$\frac d{dt}\log p_t(Y_t)=\partial_t\log p_t(Y_t)+\nabla\log p_t(Y_t)\cdot u_t(Y_t)=-\nabla\cdot u_t(Y_t).$$

Maximum-likelihood training through a continuous normalizing flow requires integrating the trajectory and its divergence. Flow matching instead regresses a velocity that realizes a chosen density evolution. The term “simulation-free” describes the construction of training pairs. Sampling still involves an ODE, and training still involves gradients.

## Conditional velocities require posterior weighting

Choose a conditional density $r_t(x\mid x_1)$. Its marginal is $p_t(x)=\int r_t(x\mid x_1)P_{data}(dx_1)$. Each conditional path has velocity $u_t(x\mid x_1)$. Below, $W_t$ denotes a state obtained by first sampling $X_1$, then sampling from $r_t(\cdot\mid X_1)$.

`[DERIVATION]` Integrating the conditional continuity equation gives

$$\partial_tp_t=-\nabla\cdot\int r_t(x\mid x_1)u_t(x\mid x_1)P_{data}(dx_1).$$

A corresponding marginal velocity is

$$u_t(x)=\frac{\int r_t(x\mid x_1)u_t(x\mid x_1)P_{data}(dx_1)}{p_t(x)}=\mathbb E[u_t(W_t\mid X_1)\mid W_t=x,t].$$

This is a posterior average. A conditional path contributes according to its probability mass at the current location. The unweighted expression $\mathbb E_{X_1}u_t(x\mid X_1)$ is generally incorrect.

`[DERIVATION · EXAMPLE]` Suppose two conditional branches have posterior probabilities 0.8 and 0.2 at the current location, with velocities 2 and $-3$. The marginal velocity is $0.8(2)+0.2(-3)=1$. Equal weighting gives $-0.5$, reversing even the direction. This constructed example illustrates weighting, not a course experiment.

## Squared error avoids computing the posterior average

Let the random target be $U$ and let $m=\mathbb E[U\mid t,W_t]$. For a network that observes only $t,W_t$,

$$\mathbb E\|v_\theta-U\|^2=\mathbb E\|v_\theta-m\|^2+\mathbb E\|U-m\|^2.$$

`[DERIVATION]` Expanding the square produces a cross term proportional to $\mathbb E[(v_\theta-m)\cdot\mathbb E[U-m\mid t,W_t]]=0$. Conditional flow matching and marginal flow matching therefore differ by a parameter-independent constant.

The equivalence applies within the same network function class and under the same time-sampling distribution. It does not make every minibatch gradient identical. Changing the time weighting can alter the compromise made by a shared model of finite capacity.

## Deriving the velocity of a Gaussian path

Given a data point $x_1$, define $\phi_t(z\mid x_1)=\mu_t(x_1)+\sigma_t(x_1)z$, where $z$ is the initial noise.

`[DERIVATION]` Differentiate while holding $z$ fixed, recover $z$ from the current position, and substitute:

$$\dot\phi_t=\dot\mu_t+\dot\sigma_tz,\qquad z=\frac{x-\mu_t}{\sigma_t},\qquad u_t(x\mid x_1)=\dot\mu_t+\frac{\dot\sigma_t}{\sigma_t}(x-\mu_t).$$

The inverse requires $\sigma_t>0$. If $\sigma_1=0$, the endpoint may be singular. Constructing the label $\dot\mu_t+\dot\sigma_tZ$ from already sampled noise avoids division by zero, but endpoint theory and numerical sampling still need separate treatment.

A density path generally does not uniquely determine its velocity. If $\nabla\cdot(p_tw_t)=0$, then $u_t+w_t$ induces the same density evolution. Specifying the entire flow map imposes a stronger constraint than specifying only its marginals.

## Straight training paths and a Gaussian counterexample

The linear choice $\mu_t=tX_1$ and $\sigma_t=1-t$ gives

$$W_t=(1-t)Z+tX_1,\qquad U=X_1-Z.$$

A training example samples independent $t,Z,X_1$ and evaluates $\|v_\theta(t,W_t)-(X_1-Z)\|^2$. Straight interpolation does not guarantee that every learned ODE trajectory is straight.

`[DERIVATION · COUNTEREXAMPLE]` Take independent $Z,X_1\sim\mathcal N(0,1)$. The intermediate variance is $D(t)=(1-t)^2+t^2$. Gaussian conditioning yields

$$u_t(x)=\frac{2t-1}{D(t)}x.$$

The ODE solution is $Y_t=\sqrt{D(t)}Y_0$. Trajectories contract toward the center and then expand, although the endpoint distributions coincide and each training interpolation is linear. For $Y_0=2$, the midpoint is $\sqrt2$ and the endpoint is 2 again. The coupling and path determine the intermediate process even when the endpoint distributions agree.

## Consistency: states on one trajectory share an endpoint

`[DERIVATION]` Let $F(t,s,x)$ map position $x$ at time $s$ to its position at time $t$. For endpoint time 1, define $C(s,x)=F(1,s,x)$. Along one trajectory, $C(t,Y_t)=Y_1$, with boundary condition $C(1,x)=x$. Papers using forward noising time often place this boundary at zero or a small positive time instead.

Training can make endpoint predictions agree at neighboring states on the same trajectory. In distillation, a teacher provides an approximate short step. Common implementations also use stopped gradients and target networks. The course handout presents a simplified consistency objective, so those engineering details should not all be attributed to its formula.

Without a boundary condition or other constraints, a constant function can also satisfy agreement. The boundary condition rules out that trivial solution. One-step quality still depends on teacher error, distillation error, and training coverage.

## Practical checks and connections

1. The direction of time determines the sign of the velocity target.
2. The sampled endpoints remain fixed when the interpolation is differentiated with respect to time.
3. Conditional velocity, marginal velocity, and network prediction are distinct objects.
4. Network function evaluations measure sampling work more directly than solver steps alone.
5. Instantaneous prediction error and final generation error require separate evaluation.

A **flow map** transports an initial state to another time. **Marginalization** integrates out an unobserved variable. **Tractable** means practically computable in the intended setting. An **endpoint map** predicts the final state from a state along a trajectory.

`[DIRECT · SOURCE SCOPE]` Lecture 18 supports the definitions and time convention. Related primary papers are [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) and [Consistency Models](https://arxiv.org/abs/2303.01469). Comparison with the separate [SD3 bilingual source edition](https://mkj69.github.io/writing/papers/sd3/annotated/) uses its time coordinate $1-t$, with the corresponding velocity sign change.

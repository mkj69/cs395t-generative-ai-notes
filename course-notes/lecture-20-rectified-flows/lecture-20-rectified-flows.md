---
title: 'Lecture 20: Diffusion IV: Rectified Flows'
slug: lecture-20
date: '2026-11-02'
updated: '2026-10-31'
status: draft
type: learning-note
course: CS 395T
source: https://noahgol.github.io/teaching/cs395t-f26/
summary: Conditional expectations and convexity explain rectification. Worked counterexamples
  separate straight interpolation, ODE trajectories, reflow, and endpoint distillation.
public: true
language: en
tags:
- Diffusion
- rectified flow
- reflow
- optimal transport
related_research_notes: []
authorship: ai-assisted
---

> **AI-assisted study note awaiting classroom materials.** As of October 31, 2026, the course schedule lists the topic without a public slide deck or handout. This note develops the topic from public primary sources and independent derivations. It does not reconstruct the student presentation. References are supplementary sources, not a confirmed assigned reading list. The page date is the scheduled class date.

`[DIRECT]` identifies a source-supported statement. `[DERIVATION]` identifies mathematical reasoning. `[SYNTHESIS]` identifies an explanatory connection. Constructed numerical examples are not reported experiments.

[Diffusion reading guide](diffusion-unit.html) · [Separate bilingual source readings](diffusion-unit.html#bilingual-source-readings)

## The first question: which endpoints are paired?

`[SYNTHESIS]` A noise distribution $\pi_0$ and a data distribution $\pi_1$ specify how often endpoints occur. They do not uniquely specify which noise sample maps to which data sample. A **coupling** is a joint distribution of the two endpoints. Independent sampling is one possible coupling.

This note uses $t=0$ for noise and $t=1$ for data. $X_t$ denotes the constructed training interpolation, while $Y_t$ denotes the generated ODE trajectory. The two symbols help keep these different objects distinct.

## From straight interpolation to an executable velocity field

`[DERIVATION]` Holding a pair $(X_0,X_1)$ fixed, define

$$X_t=(1-t)X_0+tX_1,\qquad \dot X_t=X_1-X_0.$$

The line's velocity is easy to compute. At generation time, however, only the starting state is known. The endpoint $X_1$ is unavailable. A model therefore learns a function of the current state and time:

$$\mathcal L(\theta)=\mathbb E_{t,(X_0,X_1)}\|v_\theta(X_t,t)-(X_1-X_0)\|^2.$$

The pointwise squared-error minimizer is

$$v^*(x,t)=\mathbb E[X_1-X_0\mid X_t=x].$$

The outer training expectation averages over time and endpoint pairs. The conditional expectation fixes the current $(x,t)$ and averages over pairs that could have produced that state. It is not an unconditional average of training velocities. Generation then solves

$$\frac{dY_t}{dt}=v^*(Y_t,t),\qquad Y_0\sim\pi_0.$$

Training can sample $X_t$ directly without integrating an ODE. Generation generally still requires numerical integration. Simulation-free training describes how the regression examples are constructed; it does not automatically imply one-step generation.

## Conflicting training velocities at one state

`[DERIVATION]` Suppose two kinds of training pairs pass through the same small region at a given time. Their velocities are $(2,0)$ and $(0,2)$, with equal posterior weights. The best regression prediction is $(1,1)$. It follows the average direction, rather than either individual training line.

If the velocity field satisfies conditions ensuring a unique ODE solution, one position at one time cannot continue with two different velocities. Interpolation paths can cross, while the ODE must select a unique direction from the current state. The restriction concerns the same position at the same time. Spatial paths may visit the same position at different times.

This is a local explanatory example, not a complete generative model with singular point collisions. Discrete collisions can violate smooth-density or uniqueness assumptions. They cannot be used to bypass the hypotheses of the marginal-preservation result.

## Different trajectories can preserve every time marginal

`[DERIVATION]` For a smooth test function $\varphi$, assuming differentiation and expectation can be exchanged,

$$\frac{d}{dt}\mathbb E\varphi(X_t)=\mathbb E[\nabla\varphi(X_t)\cdot(X_1-X_0)]$$

$$=\mathbb E[\nabla\varphi(X_t)\cdot v^*(X_t,t)].$$

This is the weak form of the continuity equation with velocity $v^*$. If the equation has a unique solution for the initial distribution, the ODE and interpolation share their marginal distribution at each time. They need not preserve the identity of each training pair or the joint law of the entire path.

`[DIRECT]` The Rectified Flow paper studies marginal preservation, convex transport costs, and repeated rectification under appropriate existence, uniqueness, and integrability assumptions. Source: [Liu, Gong, and Liu, v1, Sections 2 and 3](https://arxiv.org/html/2209.03003v1). These properties concern an ideal exact field. Finite networks introduce additional approximation error.

## Why a convex transport cost cannot increase

`[DERIVATION]` Write $D=X_1-X_0$. Let $c$ be a convex displacement cost, and assume the expectations below exist. The time interval has length 1. Jensen's inequality first applies to the integrated ODE displacement:

$$\mathbb E c(Y_1-Y_0)=\mathbb E c\left(\int_0^1v^*(Y_t,t)dt\right)\le\int_0^1\mathbb E c(v^*(Y_t,t))dt.$$

Equal time marginals allow $Y_t$ to be replaced by $X_t$. Conditional Jensen then gives

$$\int_0^1\mathbb E c(\mathbb E[D\mid X_t])dt\le\int_0^1\mathbb E c(D)dt=\mathbb E c(X_1-X_0).$$

There are two averages: a time average of velocities and a posterior average of endpoint displacements at a state. Convexity bounds the cost after each average.

A lower cost than the original coupling does not establish global optimality over all possible couplings. That stronger claim would require comparison with every admissible coupling. The inequality alone does not make reflow a general optimal-transport solver for arbitrary costs.

## Reflow changes the pairing; distillation learns the endpoint

`[SYNTHESIS]` Initial training often uses independently sampled noise and data. After learning a field, one can sample $Y_0$, integrate to obtain $Y_1$, and save the resulting dependent pair. A new field is then trained on linear interpolations between those new endpoints. This is the basic reflow procedure.

Endpoint distillation can instead train $G_\phi(Y_0)\approx Y_1$. Its output is a final state rather than an instantaneous velocity. The procedures can be combined, but their targets and output meanings differ. If the teacher's terminal distribution is biased, the newly paired dataset inherits that bias.

| Object | Training input and target | Generation |
| --- | --- | --- |
| Initial rectified flow | Interpolated states and endpoint differences | Integrates a velocity field |
| Reflow | New paired endpoints generated by a teacher | Integrates the retrained field |
| Endpoint distillation | Initial noise and the teacher's endpoint | Predicts the endpoint directly |

## When is one Euler step exact?

`[DERIVATION]` A smooth trajectory has the Taylor expansion

$$Y_{t+h}=Y_t+h\,v(Y_t,t)+\frac{h^2}{2}\left(\partial_tv+(\nabla_xv)v\right)(Y_t,t)+O(h^3).$$

The expression in parentheses is acceleration along the trajectory. A constant velocity along that trajectory makes Euler integration exact across the interval. A geometrically straight path can still have a nonuniform time parameterization. Straightness alone therefore does not provide this guarantee.

For example, take $v(x,t)=2t$ in one dimension with $Y_0=0$. The exact solution is $Y_t=t^2$. It stays on the same line, but a single Euler step from $t=0$ predicts 0 while the true endpoint is 1. This counterexample isolates the importance of speed along the path.

The preceding Gaussian example also shows that straight training interpolations can induce a field whose particles contract and then expand. Few-step accuracy requires evaluation; the interpolation formula alone does not establish it.

## Connecting the notation to SD3 Section 2

`[DERIVATION]` SD3 denotes clean data by $x$, noise by $\epsilon$, and writes $z_s=(1-s)x+s\epsilon$. With this note's time coordinate $t=1-s$, the same position is $(1-t)\epsilon+tx$. Reversing time changes the derivative's sign:

$$\frac{dz_s}{ds}=\epsilon-x,\qquad \frac{dz_{1-t}}{dt}=x-\epsilon.$$

Both conventions are valid. A target from one convention cannot be inserted unchanged into the other sampling direction. Time sampling and loss weighting introduce another choice. Even when ideal pointwise minimizers agree, finite training budgets can give different results.

The existing [SD3 bilingual source edition](https://mkj69.github.io/writing/papers/sd3/annotated/) presents English source text, Chinese translations, and contextual vocabulary annotations on its own page. It is a related paper edition, not a translation of the unpublished Lecture 20 handout.

## Checks and terminology

1. Endpoint marginals do not uniquely determine a coupling.
2. Regressing a straight-line velocity does not force each generated trajectory to follow that training line.
3. A decrease in convex cost does not establish global optimal transport.
4. Field-estimation error and numerical integration error have different sources.

**Rectification** constructs a new flow through regression. A **coupling** specifies the joint distribution of endpoints. A **marginal** describes one variable or one time after others are integrated out. A **convex cost** satisfies the convexity inequality. **Reflow** retrains using endpoint pairs generated by a learned flow. **Discretization** approximates continuous evolution with finitely many steps.

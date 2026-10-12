---
title: "Lecture 11 — Policy Gradients, TRPO, and PPO"
slug: "lecture-11"
date: "2026-09-30"
updated: "2026-10-12"
status: "draft"
type: "learning-note"
course: "CS 395T"
source: "https://noahgol.github.io/teaching/cs395t-f26/lecture11.pdf"
summary: "A detailed AI-assisted reconstruction of finite-horizon policy gradients, REINFORCE, the performance-difference lemma, trust-region guarantees, TRPO, PPO clipping, value learning, entropy regularization, and GAE."
public: true
tags:
  - reinforcement learning
  - policy gradients
  - REINFORCE
  - performance difference lemma
  - TRPO
  - PPO
  - GAE
  - RLHF
related_research_notes: []
authorship: "ai-assisted"
---

> **AI-assisted first draft.** This note reconstructs the supplied Lecture 11 PDF and expands its equations and proofs step by step. It has not yet been manually verified or rewritten by the author. Statements tagged `[DIRECT]` track the lecture; `[DERIVATION]`, `[EXPLANATION]`, `[EXAMPLE]`, and `[SYNTHESIS]` make the reasoning explicit.

## Lecture context

`[DIRECT]` Source: <https://noahgol.github.io/teaching/cs395t-f26/lecture11.pdf>

`[DIRECT]` The supplied PDF is ten pages long. It develops policy gradients in a finite-horizon Markov decision process, derives REINFORCE, proves a performance-difference identity and a finite-horizon policy-improvement bound, and then uses those results to motivate TRPO and PPO.

`[SOURCE BOUNDARY]` This note follows the finite-state, finite-action, finite-horizon model used by the lecture. The practical algorithms are also used with large or continuous spaces, but this draft does not silently treat every finite-space proof as a theorem for arbitrary function approximation.

`[SYNTHESIS]` The lecture is organized around one tension:

> We can estimate the policy gradient from sampled trajectories, but the moment we update the policy, those trajectories describe the wrong state distribution.

REINFORCE gives the exact gradient at the sampling policy. The performance-difference lemma tells us exactly how a new policy changes return. TRPO controls the error made when the new policy is evaluated using states from the old policy. PPO replaces TRPO's constrained optimization with a simpler clipped surrogate.

## Reading map

| Part | Central question | Main conclusion |
| --- | --- | --- |
| Finite-horizon MDP | What is the object being optimized? | A policy controls both immediate actions and the future states that become reachable. |
| REINFORCE | How can an expectation over trajectories be differentiated? | The score-function identity turns the gradient into a sampleable reward-weighted log-policy gradient. |
| Baselines | What can be subtracted without bias? | Any state-dependent, action-independent baseline has zero expected score contribution. |
| Performance difference | How does a candidate policy change return? | The exact gap is the candidate trajectory's cumulative advantage measured under the old policy. |
| Surrogate objective | How can old-policy data evaluate a candidate? | Freeze the old state distribution and correct only the action distribution with one-step ratios. |
| Trust region | When is that frozen-state approximation safe? | Its error is second order in the largest policy change, yielding a conservative improvement bound. |
| TRPO | How is the guarantee approximated in practice? | Optimize the surrogate under an average KL constraint or penalty. |
| PPO | How can the update be made simpler? | Clip the probability-ratio incentive and use minibatch gradient ascent. |
| Value, entropy, GAE | What else is trained with the policy? | A critic reduces variance, entropy delays collapse, and GAE trades estimator bias against variance. |

## 1. Finite-horizon Markov decision processes

### 1.1 Objects in the model

`[DIRECT]` A finite-horizon Markov decision process contains:

- a finite state space $\mathcal S$;
- a finite action space $\mathcal A$;
- a horizon $T\ge 1$;
- an initial-state distribution $\rho_0$;
- transition probabilities $P_t(s'\mid s,a)$;
- bounded reward functions $r_t(s,a)$.

`[DIRECT]` A time-dependent Markov policy is

$$
\pi=(\pi_0,\ldots,\pi_{T-1}),
\qquad
\pi_t(\cdot\mid s)\in\Delta(\mathcal A).
$$

`[EXPLANATION]` The subscript $t$ matters because the best decision may depend on how many steps remain. “Markov” means that the policy only needs the current state and time, not the full raw history. If the original problem is history-dependent, the history can be folded into the state.

`[DIRECT]` A trajectory

$$
\tau=(s_0,a_0,s_1,a_1,\ldots,s_{T-1},a_{T-1},s_T)
$$

is sampled according to

$$
s_0\sim\rho_0,
\qquad
a_t\sim\pi_t(\cdot\mid s_t),
\qquad
s_{t+1}\sim P_t(\cdot\mid s_t,a_t).
$$

The return and expected-return objective are

$$
R(\tau)=\sum_{t=0}^{T-1}r_t(s_t,a_t),
\qquad
J(\pi)=\mathbb E_{\tau\sim\pi}[R(\tau)].
$$

`[EXPLANATION]` An action affects two things at once:

1. the reward received now;
2. the distribution of states from which all future decisions will be made.

The second effect is why simply choosing the action with the best immediate reward is not generally optimal.

### 1.2 Mapping language generation into the MDP

`[DIRECT]` For autoregressive language generation:

- the state is the prompt together with all tokens produced so far;
- an action is the next token;
- the transition appends that token to the prefix;
- a reward may be supplied only when the response ends.

`[EXPLANATION]` The environment transition is deterministic after a token is chosen, but the policy is stochastic. A terminal reward can encode preference-model score, a verifier result, or another response-level signal.

`[SYNTHESIS]` The state space is enormous, but the finite-horizon notation remains conceptually useful. It distinguishes a token probability at a particular prefix from the probability of ever reaching that prefix.

### 1.3 Value, action-value, and advantage

`[DIRECT]` Suppose action $a$ is taken from state $s$ at time $t$, after which policy $\pi$ is followed. Define

$$
Q_t^\pi(s,a)
:=
\mathbb E
\left[
\sum_{u=t}^{T-1}r_u(s_u,a_u)
\;\middle|\;
s_t=s,\;a_t=a,\;a_u\sim\pi_u\text{ for }u>t
\right].
$$

The state value and terminal convention are

$$
V_t^\pi(s)
:=
\sum_a\pi_t(a\mid s)Q_t^\pi(s,a),
\qquad
V_T^\pi(s):=0.
$$

The advantage is

$$
A_t^\pi(s,a):=Q_t^\pi(s,a)-V_t^\pi(s).
$$

`[EXPLANATION]` $Q_t^\pi(s,a)$ asks: “what return should I expect if I force action $a$ now and follow $\pi$ afterward?” $V_t^\pi(s)$ averages that quantity over the action that $\pi$ would normally draw. The advantage therefore measures an action relative to the policy's own local behavior, not relative to an absolute zero.

### 1.4 Two identities used repeatedly

`[DERIVATION · Bellman identity]` Separate the reward at time $t$ from later rewards:

$$
\begin{aligned}
Q_t^\pi(s,a)
&=
r_t(s,a)
+\mathbb E
\left[
\sum_{u=t+1}^{T-1}r_u(s_u,a_u)
\;\middle|\;s_t=s,a_t=a
\right]\\
&=
r_t(s,a)
+\mathbb E_{s'\sim P_t(\cdot\mid s,a)}[V_{t+1}^\pi(s')].
\end{aligned}
$$

`[DERIVATION · Zero-mean advantage]` Average the advantage under the same policy:

$$
\begin{aligned}
\sum_a\pi_t(a\mid s)A_t^\pi(s,a)
&=
\sum_a\pi_t(a\mid s)
\bigl(Q_t^\pi(s,a)-V_t^\pi(s)\bigr)\\
&=
V_t^\pi(s)-V_t^\pi(s)\sum_a\pi_t(a\mid s)\\
&=0.
\end{aligned}
$$

`[EXPLANATION]` This zero-mean property is the algebra behind both baseline subtraction and the first-order accuracy of the TRPO surrogate.

Finally,

$$
J(\pi)=\mathbb E_{s_0\sim\rho_0}[V_0^\pi(s_0)].
$$

## 2. REINFORCE from the score-function identity

### 2.1 Probability of a trajectory

`[DIRECT]` Let $\pi=\pi_\theta$ be differentiably parameterized by $\theta\in\mathbb R^d$. Assume that $\rho_0$, the rewards, and the transitions do not depend on $\theta$. Then

$$
p_\theta(\tau)
=
\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_{\theta,t}(a_t\mid s_t)
P_t(s_{t+1}\mid s_t,a_t).
$$

Taking logs converts the product into a sum. The terms unrelated to the policy disappear after differentiation:

$$
\nabla_\theta\log p_\theta(\tau)
=
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_{\theta,t}(a_t\mid s_t).
$$

`[EXPLANATION]` We never need to differentiate the environment. The stochastic trajectory may pass through complicated dynamics, but the score only contains derivatives of the action probabilities chosen by the policy.

### 2.2 Deriving the policy gradient line by line

`[DERIVATION · Step 1]` Write the objective as a finite sum:

$$
J(\theta)=\sum_\tau p_\theta(\tau)R(\tau).
$$

`[DERIVATION · Step 2]` Differentiate:

$$
\nabla_\theta J(\theta)
=
\sum_\tau R(\tau)\nabla_\theta p_\theta(\tau).
$$

`[DERIVATION · Step 3]` Use the log-derivative identity

$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)\nabla_\theta\log p_\theta(\tau).
$$

`[DERIVATION · Step 4]` Recognize an expectation and insert the trajectory score:

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
\left[
R(\tau)
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_{\theta,t}(a_t\mid s_t)
\right].
}
$$

`[CONCLUSION]` We can estimate the gradient without enumerating trajectories. Sample a trajectory, compute its return, and weight every visited action's log-probability gradient by that return.

### 2.3 The Monte Carlo estimator

For $N$ independent trajectories $\tau^{(1)},\ldots,\tau^{(N)}\sim\pi_\theta$, the REINFORCE estimator is

$$
\widehat g(\theta)
=
\frac1N
\sum_{n=1}^{N}
\sum_{t=0}^{T-1}
\nabla_\theta
\log\pi_{\theta,t}
\bigl(a_t^{(n)}\mid s_t^{(n)}\bigr)
R\bigl(\tau^{(n)}\bigr).
$$

`[DIRECT]` It is unbiased, and one gradient-ascent step is $\theta\leftarrow\theta+h\widehat g(\theta)$ for step size $h>0$.

`[EXPLANATION]` Unbiased does not mean low variance. A complete return mixes together randomness from many actions and transitions, so the update produced by one batch may fluctuate strongly.

### 2.4 Why an action-independent baseline is unbiased

`[DERIVATION · Step 1]` Conditional on state $s_t$, the expected policy score is

$$
\begin{aligned}
\mathbb E_{a_t\sim\pi_{\theta,t}(\cdot\mid s_t)}
[\nabla_\theta\log\pi_{\theta,t}(a_t\mid s_t)]
&=
\sum_a\pi_{\theta,t}(a\mid s_t)
\nabla_\theta\log\pi_{\theta,t}(a\mid s_t)\\
&=
\sum_a\nabla_\theta\pi_{\theta,t}(a\mid s_t)\\
&=
\nabla_\theta 1=0.
\end{aligned}
$$

`[DERIVATION · Step 2]` Let $b_t(s_t)$ be fixed once the state is known. Then

$$
\mathbb E
\left[
b_t(s_t)
\nabla_\theta\log\pi_{\theta,t}(a_t\mid s_t)
\mid s_t
\right]
=b_t(s_t)\cdot0=0.
$$

`[CONCLUSION]` Replacing $R(\tau)$ by $R(\tau)-b_t(s_t)$ changes the sample variance but not the expected gradient.

`[CAUTION]` The baseline must not depend on the sampled action in a way that is differentiated through the policy loss. In implementations, sampled returns, advantages, and old-policy quantities are treated as constants during the policy update.

### 2.5 Advantage form of the policy gradient

`[DIRECT]` The gradient has the equivalent form

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_{\theta,t}(a_t\mid s_t)
A_t^{\pi_\theta}(s_t,a_t)
\right].
}
$$

`[EXPLANATION]` This is not a new objective. It is the same gradient written with a lower-variance learning signal. Positive advantage raises the sampled action's log-probability; negative advantage lowers it.

### 2.6 Why the data must be on-policy

`[DIRECT]` The expectation above is over trajectories from $\pi_\theta$. Once $\theta$ changes, the stored trajectories are sampled from an older policy and are therefore off-policy.

`[EXPLANATION]` A full trajectory-level importance ratio would be

$$
\frac{p_\theta(\tau)}{p_{\theta_{\mathrm{old}}}(\tau)}
=
\prod_{t=0}^{T-1}
\frac{
\pi_{\theta,t}(a_t\mid s_t)
}{
\pi_{\theta_{\mathrm{old}},t}(a_t\mid s_t)
}.
$$

The product can have very high variance. TRPO and PPO instead build local surrogate objectives that remain useful while the candidate policy stays near the data-collecting policy.

## 3. The performance-difference lemma

### 3.1 Exact statement

`[DIRECT]` For any two policies $\pi$ and $\widetilde\pi$ in the same finite-horizon MDP,

$$
\boxed{
J(\widetilde\pi)-J(\pi)
=
\mathbb E_{\tau\sim\widetilde\pi}
\left[
\sum_{t=0}^{T-1}
A_t^\pi(s_t,a_t)
\right].
}
$$

`[EXPLANATION]` The advantages belong to the old/reference policy $\pi$, but the states and actions are visited by the candidate $\widetilde\pi$. The formula answers a counterfactual question: if the new policy follows its own trajectories, how favorable are its actions when judged using the old policy's continuation values?

### 3.2 Proof by telescoping values

`[DERIVATION · Step 1]` Combine the Bellman identity with the definition of advantage:

$$
A_t^\pi(s_t,a_t)
=
r_t(s_t,a_t)
+\mathbb E[V_{t+1}^\pi(s_{t+1})\mid s_t,a_t]
-V_t^\pi(s_t).
$$

`[DERIVATION · Step 2]` Take expectation over a trajectory generated by $\widetilde\pi$. The transition law conditioned on $(s_t,a_t)$ is the same under both policies, so

$$
\mathbb E_{\widetilde\pi}[A_t^\pi(s_t,a_t)]
=
\mathbb E_{\widetilde\pi}
[r_t(s_t,a_t)+V_{t+1}^\pi(s_{t+1})-V_t^\pi(s_t)].
$$

`[DERIVATION · Step 3]` Sum over time:

$$
\begin{aligned}
\mathbb E_{\widetilde\pi}
\left[\sum_{t=0}^{T-1}A_t^\pi(s_t,a_t)\right]
&=
\mathbb E_{\widetilde\pi}
\left[
\sum_{t=0}^{T-1}r_t(s_t,a_t)
+\sum_{t=0}^{T-1}
\bigl(V_{t+1}^\pi(s_{t+1})-V_t^\pi(s_t)\bigr)
\right]\\
&=
J(\widetilde\pi)
+\mathbb E_{\widetilde\pi}
[V_T^\pi(s_T)-V_0^\pi(s_0)].
\end{aligned}
$$

`[DERIVATION · Step 4]` The terminal value is zero. Both policies share the same initial distribution, so

$$
\mathbb E_{s_0\sim\rho_0}[V_0^\pi(s_0)]=J(\pi).
$$

Therefore

$$
\mathbb E_{\widetilde\pi}
\left[\sum_{t=0}^{T-1}A_t^\pi(s_t,a_t)\right]
=J(\widetilde\pi)-J(\pi).
$$

`[SYNTHESIS]` The proof works because the old value function acts like a potential: all intermediate values cancel, leaving only the candidate return and the old policy's initial value.

### 3.3 Occupancy measures expose the hard part

Define the time-state occupancy measure

$$
\rho_\pi(t,s):=\Pr_{\tau\sim\pi}(s_t=s).
$$

For each $t$, $\sum_s\rho_\pi(t,s)=1$, so the total mass across all times is $T$.

Expanding the performance-difference lemma gives

$$
J(\widetilde\pi)
=
J(\pi)
+\sum_{t=0}^{T-1}
\sum_s\rho_{\widetilde\pi}(t,s)
\sum_a\widetilde\pi_t(a\mid s)A_t^\pi(s,a).
$$

`[EXPLANATION]` The inner action expectation is easy to evaluate if advantages are known. The difficult term is $\rho_{\widetilde\pi}$: changing actions changes which states will later be visited.

## 4. Freezing the state distribution

### 4.1 The surrogate objective

`[DIRECT]` Replace the candidate occupancy $\rho_{\widetilde\pi}$ with the old occupancy $\rho_\pi$:

$$
L_\pi(\widetilde\pi)
:=
J(\pi)
+\sum_{t=0}^{T-1}
\sum_s\rho_\pi(t,s)
\sum_a\widetilde\pi_t(a\mid s)A_t^\pi(s,a).
$$

`[EXPLANATION]` This surrogate asks what would happen if the candidate chose new actions at states distributed exactly as they were under the old policy. It corrects the action distribution but temporarily ignores the induced shift in state visitation.

### 4.2 Estimating the surrogate with old-policy data

When the old policy covers each candidate action used at sampled states,

$$
L_\pi(\widetilde\pi)-J(\pi)
=
\mathbb E_{\tau\sim\pi}
\left[
\sum_{t=0}^{T-1}
\frac{\widetilde\pi_t(a_t\mid s_t)}{\pi_t(a_t\mid s_t)}
A_t^\pi(s_t,a_t)
\right].
$$

`[DERIVATION]` Conditional on $s_t=s$,

$$
\begin{aligned}
&\mathbb E_{a_t\sim\pi_t(\cdot\mid s)}
\left[
\frac{\widetilde\pi_t(a_t\mid s)}{\pi_t(a_t\mid s)}
A_t^\pi(s,a_t)
\right]\\
&\quad=
\sum_a\pi_t(a\mid s)
\frac{\widetilde\pi_t(a\mid s)}{\pi_t(a\mid s)}
A_t^\pi(s,a)\\
&\quad=
\sum_a\widetilde\pi_t(a\mid s)A_t^\pi(s,a).
\end{aligned}
$$

Only one action ratio appears. A product of all ratios would also correct the state distribution and recover trajectory importance sampling; the surrogate deliberately does not do that.

### 4.3 First-order agreement with the true objective

Let the old policy be $\pi_{\theta_0}$ and let the candidate be $\pi_\theta$. Then

$$
L_{\pi_{\theta_0}}(\pi_{\theta_0})=J(\theta_0),
$$

and

$$
\nabla_\theta
L_{\pi_{\theta_0}}(\pi_\theta)
\big|_{\theta=\theta_0}
=
\nabla_\theta J(\theta)
\big|_{\theta=\theta_0}.
$$

`[DERIVATION · Step 1]` Define

$$
f_t(\theta,s)
:=
\sum_a\pi_{\theta,t}(a\mid s)A_t^{\pi_{\theta_0}}(s,a).
$$

At $\theta_0$, zero-mean advantage implies $f_t(\theta_0,s)=0$ for every $(t,s)$.

`[DERIVATION · Step 2]` The exact objective can be written

$$
J(\theta)
=J(\theta_0)
+\sum_{t,s}
\rho_{\pi_\theta}(t,s)f_t(\theta,s).
$$

Differentiate at $\theta_0$:

$$
\begin{aligned}
\nabla_\theta J(\theta)\big|_{\theta_0}
=
\sum_{t,s}
\Bigl[
&\nabla_\theta\rho_{\pi_\theta}(t,s)
\underbrace{f_t(\theta_0,s)}_{0}\\
&+\rho_{\pi_{\theta_0}}(t,s)
\nabla_\theta f_t(\theta,s)\big|_{\theta_0}
\Bigr].
\end{aligned}
$$

The occupancy-derivative term vanishes. The remaining term is exactly the derivative of the surrogate.

`[SYNTHESIS]` Freezing the state distribution is wrong for a finite move but harmless to first order at the old policy. The remaining question is how large the move may be before higher-order error dominates.

## 5. A trust-region policy-improvement bound

### 5.1 Measuring policy change

For distributions $p$ and $q$ on $\mathcal A$, define total variation and KL divergence:

$$
D_{\mathrm{TV}}(p,q)
:=
\frac12\sum_a|p(a)-q(a)|,
$$

$$
D_{\mathrm{KL}}(p\|q)
:=
\sum_a p(a)\log\frac{p(a)}{q(a)}.
$$

For policies, use the worst case over time and state:

$$
D_{\mathrm{TV}}^{\max}(\pi,\widetilde\pi)
:=
\max_{0\le t<T,\,s\in\mathcal S}
D_{\mathrm{TV}}
(\pi_t(\cdot\mid s),\widetilde\pi_t(\cdot\mid s)),
$$

$$
D_{\mathrm{KL}}^{\max}(\pi,\widetilde\pi)
:=
\max_{0\le t<T,\,s\in\mathcal S}
D_{\mathrm{KL}}
(\pi_t(\cdot\mid s)\|\widetilde\pi_t(\cdot\mid s)).
$$

`[DIRECT]` These maxima include states that the old policy never visits. This strength is important for the theorem and is later relaxed by practical TRPO.

### 5.2 The finite-horizon bound

Let

$$
\alpha=D_{\mathrm{TV}}^{\max}(\pi,\widetilde\pi),
\qquad
\varepsilon_A=max_{t,s,a}|A_t^\pi(s,a)|.
$$

Then

$$
\boxed{
|J(\widetilde\pi)-L_\pi(\widetilde\pi)|
\le
2\varepsilon_A T(T-1)\alpha^2.
}
$$

Equivalently,

$$
J(\widetilde\pi)
\ge
L_\pi(\widetilde\pi)
-2\varepsilon_A T(T-1)\alpha^2.
$$

`[SYNTHESIS]` There are two factors of policy difference:

1. one because the candidate's expected advantage at a fixed state is small when its action distribution is close to the old one;
2. one because the candidate and old trajectories are unlikely to have reached different states when their earlier action distributions are close.

That is why the surrogate error is quadratic in $\alpha$.

### 5.3 First factor: action-distribution change

Define

$$
f_t(s)
:=
\sum_a\widetilde\pi_t(a\mid s)A_t^\pi(s,a).
$$

Because the old policy has zero expected advantage,

$$
f_t(s)
=
\sum_a
(\widetilde\pi_t(a\mid s)-\pi_t(a\mid s))
A_t^\pi(s,a).
$$

Therefore

$$
\begin{aligned}
|f_t(s)|
&\le
\sum_a
|\widetilde\pi_t(a\mid s)-\pi_t(a\mid s)|
|A_t^\pi(s,a)|\\
&\le
2\varepsilon_A
D_{\mathrm{TV}}
(\pi_t(\cdot\mid s),\widetilde\pi_t(\cdot\mid s))\\
&\le2\varepsilon_A\alpha.
\end{aligned}
$$

### 5.4 Second factor: state-distribution change via coupling

`[DIRECT]` Couple one trajectory from $\pi$ with one from $\widetilde\pi$. Begin from the same initial state. As long as the histories agree, couple the two action draws so that they disagree with probability at most $\alpha$. If the actions agree, draw the same next state. After the first disagreement, the two trajectories may evolve independently.

`[DERIVATION]` Before state $s_t$ is reached, there are $t$ opportunities for an action disagreement. A union bound gives

$$
\Pr(s_t\ne\widetilde s_t)\le t\alpha.
$$

Subtracting the exact and frozen-state objectives gives

$$
J(\widetilde\pi)-L_\pi(\widetilde\pi)
=
\sum_{t=0}^{T-1}
\mathbb E[f_t(\widetilde s_t)-f_t(s_t)].
$$

When the states agree, the difference is zero. Otherwise each term has magnitude at most $2\varepsilon_A\alpha$, so the difference has magnitude at most $4\varepsilon_A\alpha$. Hence

$$
\begin{aligned}
|J(\widetilde\pi)-L_\pi(\widetilde\pi)|
&\le
\sum_{t=0}^{T-1}
4\varepsilon_A\alpha\Pr(s_t\ne\widetilde s_t)\\
&\le
4\varepsilon_A\alpha^2
\sum_{t=0}^{T-1}t\\
&=
4\varepsilon_A\alpha^2
\frac{T(T-1)}2\\
&=
2\varepsilon_A T(T-1)\alpha^2.
\end{aligned}
$$

### 5.5 From total variation to KL

Pinsker's inequality states

$$
D_{\mathrm{TV}}(p,q)^2
\le\frac12D_{\mathrm{KL}}(p\|q).
$$

Applying it to the policy bound gives

$$
J(\widetilde\pi)
\ge
L_\pi(\widetilde\pi)
-C_\pi D_{\mathrm{KL}}^{\max}(\pi,\widetilde\pi),
$$

where

$$
C_\pi:=\varepsilon_A T(T-1).
$$

`[EXPLANATION]` This lower bound suggests maximizing a surrogate reward minus a penalty for moving too far from the old policy.

### 5.6 Why the idealized update is monotone

Define

$$
M_i(\widetilde\pi)
:=
L_{\pi_i}(\widetilde\pi)
-C_iD_{\mathrm{KL}}^{\max}(\pi_i,\widetilde\pi).
$$

Suppose $\pi_{i+1}$ is chosen so that $M_i(\pi_{i+1})\ge M_i(\pi_i)$. The lower bound and equality at the old policy imply

$$
J(\pi_{i+1})-J(\pi_i)
\ge
M_i(\pi_{i+1})-M_i(\pi_i)
\ge0.
$$

`[LIMIT]` This is an idealized guarantee. It assumes exact advantages, exact occupancies, optimization over all Markov policies, and a worst-case KL over every time and state.

## 6. Why monotonic improvement does not solve exploration

`[DIRECT]` Consider a two-step MDP. At the initial state $s_0$:

- **safe** gives reward $1/2$ and terminates;
- **try** gives reward $0$ and moves to $s_*$.

At $s_*$:

- **good** gives reward $1$;
- **bad** gives reward $0$.

Suppose the current policy chooses safe at $s_0$ and bad at $s_*$. Its return is $1/2$. The globally better behavior is try followed by good, with return $1$.

However, under the current policy, choosing try and then following the current continuation leads to bad, so

$$
A_0^\pi(s_0,\mathrm{try})=-\frac12.
$$

Also $s_*$ is never visited, so $\rho_\pi(1,s_*)=0$. Consequently the surrogate cannot see the benefit of changing the action at $s_*$, and it penalizes moving toward try at $s_0$.

`[SYNTHESIS]` Local policy improvement can get stuck because coordinated changes are required at both a visited state and an unvisited state. Exact advantages do not eliminate this problem: the advantage of the first exploratory action is evaluated under the old, still-bad continuation.

`[DIRECT]` If try is assigned only a small probability $p$, then $N$ sampled episodes all miss $s_*$ with probability

$$
(1-p)^N.
$$

Positive action probabilities therefore do not by themselves imply efficient exploration.

## 7. Practical TRPO

### 7.1 The population surrogate

Let $\pi_{\mathrm{old}}=\pi_{\theta_{\mathrm{old}}}$ be the policy that collected the batch. Let $\mathbb E_{\mathrm{old}}$ draw $(t,s,a)$ by first choosing $(t,s)$ from the normalized old occupancy $\rho_{\mathrm{old}}(t,s)/T$ and then drawing $a\sim\pi_{\mathrm{old},t}(\cdot\mid s)$.

Define

$$
F(\theta)
:=
\mathbb E_{\mathrm{old}}
\left[
\frac{\pi_{\theta,t}(a\mid s)}
{\pi_{\mathrm{old},t}(a\mid s)}
A_t^{\pi_{\mathrm{old}}}(s,a)
\right],
$$

and the average KL

$$
\overline D_{\mathrm{KL}}(\theta_{\mathrm{old}},\theta)
:=
\mathbb E_{\mathrm{old}}
\left[
D_{\mathrm{KL}}
(\pi_{\mathrm{old},t}(\cdot\mid s)
\|\pi_{\theta,t}(\cdot\mid s))
\right].
$$

The factor $1/T$ in the sampling convention does not change the optimizer of the surrogate.

### 7.2 Hard constraint and soft penalty

TRPO replaces the theorem's worst-case constraint with an average over states visited by the old policy. Two natural forms are

$$
\max_\theta F(\theta)
\quad\text{subject to}\quad
\overline D_{\mathrm{KL}}(\theta_{\mathrm{old}},\theta)\le\delta,
$$

and

$$
\max_\theta
\left\{
F(\theta)-\beta
\overline D_{\mathrm{KL}}(\theta_{\mathrm{old}},\theta)
\right\}.
$$

`[EXPLANATION]` The hard version is the usual TRPO formulation. The soft version permits adaptive tuning of $\beta$: increase the penalty if the observed KL is too large and decrease it if the update is too conservative.

`[LIMIT]` Average KL is practical but weaker than the maximum KL used in the theorem. A small average can hide a large change at a rare state.

### 7.3 Empirical optimization

With $N$ trajectories, empirical expectations average over $NT$ time steps. During optimization:

- trajectories remain fixed;
- old action probabilities remain fixed;
- advantage estimates remain fixed;
- only the candidate policy probabilities change.

`[EXPLANATION]` The batch becomes off-policy as soon as $\theta$ moves, but the surrogate can reuse it for several optimization steps as long as the new policy remains near $\pi_{\mathrm{old}}$.

## 8. Proximal policy optimization

### 8.1 The probability ratio

For a sampled transition $(t,s_t,a_t)$, define

$$
w_t(\theta)
:=
\frac{
\pi_{\theta,t}(a_t\mid s_t)
}{
\pi_{\mathrm{old},t}(a_t\mid s_t)
},
\qquad
w_t(\theta_{\mathrm{old}})=1.
$$

Let $\widehat A_t$ estimate the old policy's advantage. The ordinary empirical surrogate contribution is $w_t(\theta)\widehat A_t$.

`[EXPLANATION]` If $\widehat A_t>0$, gradient ascent tries to make $w_t$ larger: increase the sampled action's probability. If $\widehat A_t<0$, it tries to make $w_t$ smaller.

### 8.2 The clipped objective

Choose $0<\epsilon<1$ and define

$$
\operatorname{clip}(w,1-\epsilon,1+\epsilon)
:=
\min\{1+\epsilon,\max\{1-\epsilon,w\}\}.
$$

PPO's clipped per-sample objective is

$$
\boxed{
\ell_t^{\mathrm{CLIP}}(\theta)
:=
\min
\left\{
w_t(\theta)\widehat A_t,
\operatorname{clip}(w_t(\theta),1-\epsilon,1+\epsilon)
\widehat A_t
\right\}.
}
$$

The batch objective is

$$
\widehat L^{\mathrm{CLIP}}(\theta)
=
\widehat{\mathbb E}[\ell_t^{\mathrm{CLIP}}(\theta)].
$$

### 8.3 Why the minimum is essential

The objective can be written piecewise:

$$
\ell_t^{\mathrm{CLIP}}(\theta)
=
\begin{cases}
\widehat A_t\min\{w_t(\theta),1+\epsilon\},
&\widehat A_t\ge0,\\[4pt]
\widehat A_t\max\{w_t(\theta),1-\epsilon\},
&\widehat A_t<0.
\end{cases}
$$

`[EXPLANATION · Positive advantage]` We want to increase the action probability, but once $w_t$ reaches $1+\epsilon$, further increase gives no additional objective value. A move in the wrong direction, below $1-\epsilon$, is **not** clipped away; it continues to reduce the objective.

`[EXPLANATION · Negative advantage]` We want to decrease the action probability, but once $w_t$ reaches $1-\epsilon$, further decrease gives no additional objective value. A wrong-direction increase above $1+\epsilon$ continues to hurt.

`[SYNTHESIS]` PPO does not simply clamp every ratio. It removes the incentive for an excessively large move in the beneficial direction while preserving the penalty for moving far in the harmful direction.

### 8.4 Numerical check for a clipping width of 0.2

| Ratio $w$ | $\widehat A=+1$ | $\widehat A=-1$ | Interpretation |
| ---: | ---: | ---: | --- |
| $0.5$ | $0.5$ | $-0.8$ | A positive action was suppressed; a negative action has already been sufficiently suppressed. |
| $0.9$ | $0.9$ | $-0.9$ | Both lie inside the active interval. |
| $1.0$ | $1.0$ | $-1.0$ | Old policy. |
| $1.1$ | $1.1$ | $-1.1$ | Both lie inside the active interval. |
| $1.5$ | $1.2$ | $-1.5$ | Positive improvement saturates; increasing a negative action is still penalized. |

`[CHECK]` For $\widehat A=-1$ and $w=0.5$, the two candidates are $-0.5$ and $-0.8$; the minimum is $-0.8$. This is why the objective is flat below $1-\epsilon$ rather than becoming artificially better.

## 9. Value learning, entropy, and advantage estimation

### 9.1 Critic loss

Suppose the parameters also define a value head $V_{\theta,t}(s)$. Given a fixed target $V_t^{\mathrm{targ}}$, use

$$
\ell_t^{\mathrm{VF}}(\theta)
:=
\bigl(V_{\theta,t}(s_t)-V_t^{\mathrm{targ}}\bigr)^2.
$$

`[EXPLANATION]` The critic is a learned baseline. It does not determine which action was sampled; it predicts future return from the state so that the policy can learn from a centered, lower-variance signal.

### 9.2 Entropy bonus

The policy entropy at a state is

$$
\mathcal H(\pi_{\theta,t}(\cdot\mid s))
:=
-\sum_a
\pi_{\theta,t}(a\mid s)
\log\pi_{\theta,t}(a\mid s).
$$

Rewarding entropy discourages premature concentration on a few actions. It does not guarantee efficient exploration, as the two-step counterexample showed.

### 9.3 Combined PPO objective

For coefficients $c_1,c_2\ge0$, the lecture writes the maximization objective as

$$
\boxed{
\widehat L^{\mathrm{PPO}}(\theta)
=
\widehat{\mathbb E}
\left[
\ell_t^{\mathrm{CLIP}}(\theta)
-c_1\ell_t^{\mathrm{VF}}(\theta)
+c_2\mathcal H(\pi_{\theta,t}(\cdot\mid s_t))
\right].
}
$$

`[SIGN CHECK]` If software minimizes a loss instead of maximizing an objective, the signs reverse:

$$
\mathcal L_{mathrm{min}}
=
-\widehat L^{\mathrm{CLIP}}
+c_1\widehat L^{\mathrm{VF}}
-c_2\widehat{\mathcal H}.
$$

### 9.4 Bootstrapped advantage estimates

Suppose a sampled segment begins at $t$ and stops at $b$, with $t<b\le T$. Let $\widehat V$ be the value predictor used when data were collected and set $\widehat V_T=0$. Define

$$
V_t^{\mathrm{targ}}
:=
\sum_{u=t}^{b-1}r_u(s_u,a_u)
+\widehat V_b(s_b),
$$

$$
\widehat A_t
:=
V_t^{\mathrm{targ}}-\widehat V_t(s_t).
$$

`[EXPLANATION]` The bootstrap value stands in for the unobserved return beyond the segment. If the value estimates are exact, the conditional expectation of $\widehat A_t$ is the true advantage.

### 9.5 Generalized advantage estimation

Define the one-step temporal-difference residual

$$
\delta_u
:=
r_u(s_u,a_u)
+\widehat V_{u+1}(s_{u+1})
-\widehat V_u(s_u).
$$

For $\lambda\in[0,1]$, GAE uses

$$
\boxed{
\widehat A_t^{\mathrm{GAE}}
=
\sum_{u=t}^{b-1}
\lambda^{u-t}\delta_u.
}
$$

`[DERIVATION · $\lambda=1$]` The value terms telescope:

$$
\begin{aligned}
\sum_{u=t}^{b-1}\delta_u
&=
\sum_{u=t}^{b-1}r_u
+\sum_{u=t}^{b-1}
(\widehat V_{u+1}-\widehat V_u)\\
&=
\sum_{u=t}^{b-1}r_u
+\widehat V_b-\widehat V_t.
\end{aligned}
$$

This recovers the bootstrapped segment-return advantage.

`[EXPLANATION]` Smaller $\lambda$ emphasizes short lookaheads. This usually lowers sampling variance but relies more heavily on the learned value function and can therefore introduce more bias when the critic is inaccurate.

### 9.6 Backward computation of GAE

GAE need not be computed with a nested sum. Traverse the segment backward:

```python
gae = 0.0
advantages = zeros_like(rewards)

for t in reversed(range(segment_length)):
    delta = rewards[t] + values[t + 1] - values[t]
    gae = delta + lam * gae
    advantages[t] = gae

value_targets = advantages + values[:-1]
```

`[IMPLEMENTATION NOTE]` For discounted problems, implementations usually insert a discount $\gamma$ into both the TD residual and recursion. The lecture's displayed finite-horizon formulas use undiscounted returns, so $\gamma$ is omitted here rather than silently added.

## 10. The PPO training loop

`[DIRECT]` One outer iteration follows this sequence:

1. Freeze $\theta_{\mathrm{old}}$.
2. Collect trajectories using $\pi_{\theta_{\mathrm{old}}}$.
3. Store old action log-probabilities.
4. Compute fixed advantages and value targets.
5. Initialize the trainable parameters at $\theta_{\mathrm{old}}$.
6. Take several epochs of minibatch gradient steps on the combined objective.
7. Hold the old log-probabilities, advantages, and targets fixed throughout those epochs.
8. Make the resulting policy the next data-collection policy and repeat.

### 10.1 Minimal implementation skeleton

```python
# rollout tensors are fixed for all PPO epochs
obs, actions, old_logp, advantages, value_targets = rollout_batch

for _ in range(ppo_epochs):
    for mb in minibatches(rollout_batch):
        logp, values, entropy = model.evaluate(mb.obs, mb.actions)

        ratio = exp(logp - mb.old_logp)
        unclipped = ratio * mb.advantages
        clipped = clip(ratio, 1 - eps, 1 + eps) * mb.advantages
        policy_objective = mean(minimum(unclipped, clipped))

        value_loss = mean((values - mb.value_targets) ** 2)
        entropy_bonus = mean(entropy)

        # optimizer minimizes, so negate the maximization objective
        loss = -policy_objective + c1 * value_loss - c2 * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

`[CHECK]` The ratio should equal one before the first update on a fresh batch. If it does not, the stored old log-probability and the recomputed log-probability are not describing the same policy, observations, actions, or masking convention.

`[CHECK]` Advantage normalization, masking padded tokens, value clipping, gradient clipping, KL-based early stopping, and distributed batching are common engineering additions, but they are not part of the lecture's displayed derivation and should be documented separately when used.

## 11. Connections to language-model post-training

### 11.1 PPO for language-model alignment

`[DIRECT]` InstructGPT uses PPO with a learned value function and adds a per-token KL penalty toward a fixed reference policy $\pi_{\mathrm{ref}}$, usually the supervised fine-tuned model.

`[EXPLANATION]` Two different anchors are present:

- $\pi_{\mathrm{old}}$ changes every batch and controls the size of one PPO update;
- $\pi_{\mathrm{ref}}$ stays fixed and discourages the trained policy from drifting too far from the supervised model over the whole run.

The reference KL is not the same as an entropy bonus. At a fixed state,

$$
-D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})
=
\mathcal H(\pi)
+\mathbb E_{a\sim\pi}[\log\pi_{\mathrm{ref}}(a)].
$$

It rewards entropy but also specifically favors actions the reference considers likely.

### 11.2 Removing the learned critic

`[DIRECT]` GRPO keeps a clipped policy term but estimates advantages by sampling several responses to the same prompt and comparing their rewards. It therefore removes the learned value model. Its objective includes a reference-policy KL and no separate entropy bonus.

`[SYNTHESIS]` This creates the bridge to Lecture 13. PPO's policy-ratio machinery can remain while the source of the advantage estimate changes from a token-level critic to a same-prompt group statistic.

`[CAUTION]` Group-relative response-level estimators are not generally unbiased estimators of the token-level advantage $A_t^{\pi_{\mathrm{old}}}(s_t,a_t)$ defined in this lecture.

### 11.3 Later changes to regularization

`[DIRECT]` DAPO removes the fixed-reference KL penalty and uses different lower and upper clipping thresholds. A larger upper threshold allows more increase for actions with positive estimated advantage, which is intended to counter entropy collapse.

`[SYNTHESIS]` Across PPO, GRPO, and DAPO, the clipped policy term is comparatively stable while three design choices vary:

1. how advantages are estimated;
2. whether a learned critic is used;
3. how the policy is regularized.

## 12. What each approximation changes

| Step | Exact object | Approximation or replacement | Main risk |
| --- | --- | --- | --- |
| REINFORCE | $\nabla J(\theta)$ under current-policy trajectories | Monte Carlo batch average | High variance |
| Baseline/critic | Exact advantage | Learned value prediction | Bias or noise from critic error |
| Surrogate | Candidate occupancy $\rho_{\widetilde\pi}$ | Old occupancy $\rho_\pi$ | Error grows when the policy moves far |
| TRPO theory → practice | Maximum KL over all states | Average KL over sampled old states | Rare-state changes can be hidden |
| TRPO → PPO | Explicit KL constraint | Clipped per-sample incentive | Clipping does not strictly enforce a KL bound |
| Full episodes → segments | Complete return | Bootstrap from $\widehat V$ | Dependence on value accuracy |
| Long lookahead → GAE | Monte Carlo-like advantage | $\lambda$-weighted TD residuals | Bias-variance tradeoff |

`[SYNTHESIS]` “PPO is stable” should not be interpreted as a theorem that clipping guarantees improvement. The monotonic theorem belongs to a stronger idealized setup. PPO is a practical surrogate whose behavior should still be monitored through KL, clipping fraction, value error, entropy, and achieved return.

## 13. Common conceptual and implementation mistakes

### 13.1 Confusing old and reference policies

The old policy is the batch behavior policy. The reference policy is a longer-term regularization anchor. They may coincide at one moment but have different roles.

### 13.2 Recomputing “old” quantities after an update

If `old_logp` is recomputed with current parameters, the ratio collapses toward one and no longer measures policy change. Old log-probabilities must be stored or evaluated by a frozen model.

### 13.3 Differentiating through the advantage target

The policy-gradient argument treats $\widehat A_t$ as fixed. If actor and critic share a network, the advantage should be detached for the policy term unless a deliberately different objective is intended.

### 13.4 Thinking clipping bounds the ratio everywhere

Clipping bounds the objective's incentive in one favorable direction; it does not project the policy into $[1-\epsilon,1+\epsilon]$ for every sample. Ratios can and do leave that interval.

### 13.5 Assuming a small average KL implies every state is safe

The theoretical guarantee uses a maximum over all states. Practical average KL may miss a large shift at a state that is rare under the old policy.

### 13.6 Treating monotone local improvement as global optimization

The exploration example shows that coordinated changes at unvisited states may be invisible or look locally harmful.

## 14. Compact derivation chain

The lecture's main chain can be compressed into six statements:

1. **Score identity**

   $$
   \nabla J(\theta)
   =\mathbb E_{\pi_\theta}
   \left[R(\tau)\nabla\log p_\theta(\tau)\right].
   $$

2. **Trajectory score decomposes**

   $$
   \nabla\log p_\theta(\tau)
   =\sum_t\nabla\log\pi_{\theta,t}(a_t\mid s_t).
   $$

3. **Exact performance difference**

   $$
   J(\widetilde\pi)-J(\pi)
   =\mathbb E_{\widetilde\pi}\left[\sum_tA_t^\pi\right].
   $$

4. **Freeze the state distribution**

   $$
   L_\pi(\widetilde\pi)-J(\pi)
   =\mathbb E_{\pi}
   \left[
   \sum_t
   \frac{\widetilde\pi_t(a_t\mid s_t)}{\pi_t(a_t\mid s_t)}
   A_t^\pi(s_t,a_t)
   \right].
   $$

5. **Control the surrogate error**

   $$
   J(\widetilde\pi)
   \ge
   L_\pi(\widetilde\pi)
   -C_\pi D_{\mathrm{KL}}^{\max}(\pi,\widetilde\pi).
   $$

6. **Replace constrained optimization with clipping**

   $$
   \ell^{\mathrm{CLIP}}
   =\min\{w\widehat A,\operatorname{clip}(w,1-\epsilon,1+\epsilon)\widehat A\}.
   $$

`[SYNTHESIS]` Every step trades something: exactness for sampleability, variance for critic dependence, candidate-state accuracy for data reuse, worst-case control for practical averages, and a hard trust region for a clipped heuristic.

## 15. Open questions for manual study

- `[OPEN]` In what regimes does average KL provide enough control even though the theorem requires maximum KL?
- `[OPEN]` Can a practical diagnostic detect rare-state policy changes that average KL misses?
- `[OPEN]` How should $\epsilon$, the number of PPO epochs, and learning rate be coupled rather than tuned independently?
- `[OPEN]` When actor and critic share most parameters, how much does critic optimization alter the effective policy step outside the clipped surrogate analysis?
- `[OPEN]` In language modeling, when is a token-level critic preferable to group-relative response baselines?
- `[OPEN]` Which parts of the trust-region derivation survive when rewards or transitions depend on the model parameters?
- `[OPEN]` How does the exploration counterexample change under intrinsic rewards, optimistic value estimates, or explicit coverage objectives?

## 16. Source trail

`[DIRECT]` The lecture notes cite the following primary works:

1. Williams (1992), [REINFORCE](https://doi.org/10.1007/BF00992696).
2. Kakade and Langford (2002), *Approximately Optimal Approximate Reinforcement Learning*.
3. Schulman et al. (2015), [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477).
4. Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
5. Schulman et al. (2016), [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
6. Ouyang et al. (2022), [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155).
7. Shao et al. (2024), [DeepSeekMath](https://arxiv.org/abs/2402.03300).
8. Yu et al. (2025), [DAPO](https://arxiv.org/abs/2503.14476).

## Revision log

- **2026-10-12 — AI-assisted first draft.** Reconstructed the complete Lecture 11 sequence, expanded both main proofs, added clipping case analysis, GAE recursion, implementation checks, and a bridge to Lecture 13. Manual verification remains pending.

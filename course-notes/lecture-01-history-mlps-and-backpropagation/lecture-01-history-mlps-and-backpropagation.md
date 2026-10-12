---
title: "Lecture 01 — From Perceptrons to Backpropagation"
slug: "lecture-01"
date: "2026-08-24"
updated: "2026-10-12"
status: "draft"
type: "learning-note"
course: "CS 395T"
source: "https://noahgol.github.io/teaching/cs395t-f26/lecture1.pdf"
summary: "An AI-assisted reconstruction of Lecture 1, covering perceptrons, margin-based mistake bounds, MLP expressivity, approximation limits, depth separation, SGD, and backpropagation as dynamic programming."
public: true
tags:
  - perceptron
  - multi-layer perceptron
  - universal approximation
  - depth separation
  - stochastic gradient descent
  - backpropagation
related_research_notes: []
authorship: "ai-assisted"
---

> **AI-assisted first draft.** This note reconstructs the official nine-page Lecture 1 handout and expands its arguments into step-by-step derivations and examples. It is not a transcript and has not yet been manually verified or rewritten by the author. Statements tagged `[DIRECT]` follow the handout; `[DERIVATION]`, `[EXPLANATION]`, and `[SYNTHESIS]` make the reasoning explicit.

## Source boundary

`[DIRECT]` Official handout: <https://noahgol.github.io/teaching/cs395t-f26/lecture1.pdf>.

`[DIRECT]` The handout has three substantive parts:

1. threshold neurons, the perceptron, and multi-layer perceptrons;
2. expressivity, including Boolean universality, universal approximation, approximation rates, and a depth-separation statement;
3. empirical risk, stochastic gradient descent, and backpropagation on a computational graph.

`[SOURCE BOUNDARY]` The historical and theorem statements labeled `[DIRECT]` come from the handout. The examples, intermediate algebra, and interpretations below are reconstructions rather than reported classroom wording.

## Reading map

| Part | Question | Main idea |
| --- | --- | --- |
| MCP neuron | How can a logical neuron be modeled? | Threshold a weighted sum, with special inhibition. |
| Perceptron | How can a linear threshold be learned online? | Correct mistakes by adding the signed input. |
| Mistake bound | Why does margin make learning finite? | Alignment grows linearly while norm grows only as a square root. |
| MLP expressivity | What functions can layered neurons represent? | Width and depth expand the available function class. |
| Universal approximation | Is one hidden layer theoretically enough? | Dense representability does not imply efficient representability. |
| Backpropagation | How are all parameter derivatives computed efficiently? | Cache a forward pass and reuse adjoints in reverse topological order. |

## 1. From a logical neuron to a learned separator

### 1.1 McCulloch-Pitts neuron

`[DIRECT]` An MCP neuron receives excitatory binary inputs $x_1,\ldots,x_m$, real weights $w_1,\ldots,w_m$, inhibitory binary inputs $z_1,\ldots,z_r$, and a threshold $\vartheta$.

Its output is

$$
y=
\begin{cases}
1,
&\sum_{i=1}^{m}w_ix_i\ge\vartheta
\text{ and }z_j=0\text{ for all }j,\\
0,&\text{otherwise.}
\end{cases}
$$

`[EXPLANATION]` The excitatory inputs accumulate evidence. Any active inhibitory input vetoes the output. The important limitation is that the model specifies no learning rule: the weights are chosen rather than learned.

### 1.2 Perceptron prediction rule

Suppress the special inhibitory inputs and use labels in $\{-1,+1\}$. A homogeneous linear classifier predicts

$$
\hat y_t=
\begin{cases}
+1,&\langle w_t,x_t\rangle>0,\\
-1,&\langle w_t,x_t\rangle\le0.
\end{cases}
$$

`[EXPLANATION]` A nonzero threshold can be absorbed into the weight vector. Append a constant coordinate to the input:

$$
\tilde x=(x,1),
\qquad
\tilde w=(w,-\vartheta).
$$

Then

$$
\langle\tilde w,\tilde x\rangle
=\langle w,x\rangle-\vartheta.
$$

The affine separator in the original space becomes a homogeneous separator in the augmented space.

### 1.3 Online perceptron update

Initialize $w_1=0$. After predicting on $(x_t,y_t)$, update only after a mistake:

$$
w_{t+1}=
\begin{cases}
w_t+y_tx_t,&\hat y_t\ne y_t,\\
w_t,&\hat y_t=y_t.
\end{cases}
$$

`[EXPLANATION]`

- If $y_t=+1$ was predicted negative, add $x_t$ and increase its score.
- If $y_t=-1$ was predicted positive, subtract $x_t$ and decrease its score.
- Correct examples cause no update.

### 1.4 The update as a loss subgradient

Define perceptron loss

$$
\ell_t(w)=\max\{0,-y_t\langle w,x_t\rangle\}.
$$

When the signed score is negative,

$$
\nabla_w\ell_t(w)=-y_tx_t.
$$

A unit subgradient step is therefore

$$
w\leftarrow w-\nabla\ell_t(w)=w+y_tx_t.
$$

`[SYNTHESIS]` The perceptron can be viewed historically as a learning rule and mathematically as online subgradient descent on a piecewise-linear loss.

## 2. Perceptron mistake bound

### 2.1 Assumptions

`[DIRECT]` Suppose each input satisfies

$$
\|x_t\|_2=1,
$$

and there exists a unit vector $u$ with positive margin

$$
\gamma
:=
\min_{1\le t\le T}
y_t\langle u,x_t\rangle
>0.
$$

Then the perceptron makes at most

$$
\boxed{M\le\frac1{\gamma^2}}
$$

mistakes.

### 2.2 Why two inequalities are enough

Let $(\tilde x_s,\tilde y_s)$ be the example causing the $s$-th mistake, and let $v_s$ be the weight vector immediately after that update:

$$
v_s=v_{s-1}+\tilde y_s\tilde x_s,
\qquad v_0=0.
$$

The proof tracks two quantities:

1. progress along the target direction $u$;
2. growth of the total norm $\|v_s\|$.

### 2.3 Progress toward the target separator

`[DERIVATION · Step 1]` Take an inner product with $u$:

$$
\langle v_s,u\rangle
=\langle v_{s-1},u\rangle
+\tilde y_s\langle\tilde x_s,u\rangle.
$$

By the margin assumption,

$$
\tilde y_s\langle\tilde x_s,u\rangle\ge\gamma.
$$

Thus

$$
\langle v_s,u\rangle
\ge
\langle v_{s-1},u\rangle+\gamma.
$$

After $M$ mistakes,

$$
\boxed{\langle v_M,u\rangle\ge M\gamma.}
$$

`[EXPLANATION]` Every mistake update moves the weight vector at least $\gamma$ forward in the target direction.

### 2.4 Norm growth is slow

`[DERIVATION · Step 2]` Expand the squared norm:

$$
\begin{aligned}
\|v_s\|^2
&=\|v_{s-1}+\tilde y_s\tilde x_s\|^2\\
&=\|v_{s-1}\|^2
+2\tilde y_s\langle v_{s-1},\tilde x_s\rangle
+\|\tilde x_s\|^2.
\end{aligned}
$$

Because this example was misclassified,

$$
\tilde y_s\langle v_{s-1},\tilde x_s\rangle\le0.
$$

Since $\|\tilde x_s\|=1$,

$$
\|v_s\|^2\le\|v_{s-1}\|^2+1.
$$

Induction gives

$$
\boxed{\|v_M\|^2\le M,\qquad\|v_M\|\le\sqrt M.}
$$

### 2.5 Combine progress and norm

By Cauchy-Schwarz and $\|u\|=1$,

$$
\langle v_M,u\rangle
\le\|v_M\|.
$$

Therefore

$$
M\gamma
\le
\langle v_M,u\rangle
\le
\|v_M\|
\le
\sqrt M.
$$

For $M>0$, divide by $\sqrt M$:

$$
\sqrt M\,\gamma\le1.
$$

Squaring yields

$$
M\le\frac1{\gamma^2}.
$$

`[SYNTHESIS]` The proof works because useful alignment accumulates linearly in the number of mistakes, while the vector norm grows at most like the square root. A positive margin makes indefinite mistakes geometrically impossible.

### 2.6 What the theorem does not say

- It assumes linear separability with positive margin.
- The bound concerns mistakes, not probabilistic calibration or test error.
- A small margin gives a weak bound.
- The result is independent of dimension only after inputs and the separator are normalized.

## 3. Multi-layer perceptrons

### 3.1 Layer equations

Let $h^{(0)}(x)=x$. For layers $\ell=1,\ldots,L$,

$$
z^{(\ell)}(x)
=W^{(\ell)}h^{(\ell-1)}(x)+b^{(\ell)},
$$

$$
h^{(\ell)}(x)
=\sigma_\ell(z^{(\ell)}(x)),
$$

where the activation is applied coordinatewise.

A scalar output may be

$$
f_\theta(x)=a^\top h^{(L)}(x)+c.
$$

`[EXPLANATION]` Without a nonlinear activation, stacked affine maps collapse into one affine map. For example,

$$
W_2(W_1x+b_1)+b_2
=(W_2W_1)x+(W_2b_1+b_2).
$$

The nonlinearity is what prevents depth from being algebraically redundant.

## 4. Boolean universality

`[DIRECT]` Every function

$$
F:\{0,1\}^d\to\{0,1\}
$$

can be represented exactly by a one-hidden-layer network with threshold activations.

### 4.1 Detect one binary pattern

Fix $a\in\{0,1\}^d$. Define the number of matching coordinates

$$
s_a(x)
=
\sum_{i:a_i=1}x_i
+\sum_{i:a_i=0}(1-x_i).
$$

Then

$$
s_a(x)=d
\quad\Longleftrightarrow\quad
x=a.
$$

A threshold at $d-\tfrac12$ produces

$$
I_a(x)=\mathbf 1\{x=a\}.
$$

### 4.2 OR the satisfying patterns

Let

$$
S=\{a\in\{0,1\}^d:F(a)=1\}.
$$

Create one hidden unit $I_a$ for every $a\in S$ and threshold their sum:

$$
F(x)=
\mathbf 1
\left\{
\sum_{a\in S}I_a(x)\ge\frac12
\right\}.
$$

`[SYNTHESIS]` This is disjunctive normal form implemented as a network: each hidden unit is an AND-like exact-match test, and the output is an OR.

`[LIMIT]` The construction may require one hidden unit per satisfying input—up to $2^d$. Expressibility alone says nothing about parameter efficiency or learnability.

## 5. Universal approximation with ReLU

### 5.1 Statement and interpretation

`[DIRECT]` On a compact set $K\subset\mathbb R^d$, finite sums of ReLU ridge functions can uniformly approximate every continuous function. For every $f\in C(K)$ and $\varepsilon>0$, there exist $N$, $a_j$, $w_j$, and $b_j$ such that

$$
\sup_{x\in K}
\left|
f(x)-
\sum_{j=1}^{N}
a_j\rho(\langle w_j,x\rangle+b_j)
\right|
<\varepsilon,
$$

where $\rho(t)=\max\{0,t\}$.

`[EXPLANATION]` The theorem is an existence statement. It does not provide the weights, guarantee that gradient descent finds them, or guarantee a small width.

### 5.2 Proof architecture

Let $H_K$ be the linear span of all ReLU ridge functions restricted to $K$.

`[DERIVATION · Step 1]` Suppose the closure $\overline{H_K}$ were a proper subspace of $C(K)$.

`[DERIVATION · Step 2]` Hahn-Banach separation gives a nonzero continuous linear functional $\Lambda$ that annihilates $H_K$:

$$
\Lambda(h)=0
\quad\forall h\in H_K,
$$

but $\Lambda(f_0)=1$ for some $f_0\notin\overline{H_K}$.

`[DERIVATION · Step 3]` By Riesz representation, there is a finite signed measure $\mu$ on $K$ with

$$
\Lambda(g)=\int_Kg(x)\,d\mu(x).
$$

Because every ridge function lies in $H_K$,

$$
\int_K
\rho(\langle w,x\rangle+b)
\,d\mu(x)=0
$$

for all $w,b$.

`[DERIVATION · Step 4]` Push $\mu$ forward along each one-dimensional projection $x\mapsto\langle w,x\rangle$. ReLU translates generate continuous piecewise-linear functions on the resulting compact interval, so the pushforward measure integrates every continuous function to zero.

`[DERIVATION · Step 5]` All one-dimensional projections vanish. Equivalently, the Fourier transform of $\mu$ vanishes everywhere, forcing $\mu=0$.

This contradicts $\Lambda(f_0)=1$. Therefore

$$
\overline{H_K}=C(K).
$$

`[SYNTHESIS]` The proof is indirect: if ridge functions failed to be dense, a nonzero signed measure would distinguish the missing function while being invisible to every ReLU ridge function. The projection argument shows that no such measure exists.

## 6. Approximation rate and the curse of dimensionality

`[DIRECT]` The handout gives a representative smooth-function approximation rate of the form

$$
N\le C\varepsilon^{-d/n}
$$

for functions with $n$ derivatives controlled in a Sobolev norm.

Solving for the error gives the equivalent scaling

$$
\varepsilon
\lesssim
N^{-n/d}.
$$

`[EXPLANATION]` Greater smoothness $n$ improves the rate. Greater dimension $d$ makes the exponent worse. To hold error fixed while dimension increases, the required width can grow exponentially.

`[SYNTHESIS]` Universal approximation says approximation is possible; a rate asks whether it is affordable. These are different questions.

## 7. Why depth can matter

`[DIRECT]` The handout states a depth-separation result: certain functions represented by a sufficiently deep ReLU network of polynomial size require exponentially many nodes for substantially shallower networks to achieve even fixed constant error.

`[EXPLANATION]` The point is not that every deep network beats every shallow one. It is that repeated composition can represent some functions much more economically than a shallow expansion.

### 7.1 Composition intuition

If a small module $g$ creates several oscillations, composing it repeatedly,

$$
g^{\circ L}=g\circ g\circ\cdots\circ g,
$$

can multiply the number of oscillatory regions with depth. A shallow network must reproduce all regions in one layer and may need exponentially more units.

`[LIMIT]` A separation theorem concerns a constructed worst-case function and a specified approximation norm. It is not by itself an explanation of why a particular trained model generalizes.

## 8. Empirical risk and stochastic gradient descent

Given data

$$
S=\{(x_i,y_i):i=1,\ldots,n\},
$$

model $f_\theta$, and loss $\ell$, empirical risk is

$$
L(\theta)
=\frac1n\sum_{i=1}^{n}
\ell(f_\theta(x_i),y_i).
$$

For mini-batch $B_t$, SGD updates

$$
\theta_{t+1}
=\theta_t
-\eta_t
\frac1{|B_t|}
\sum_{i\in B_t}
\nabla_\theta
\ell(f_{\theta_t}(x_i),y_i).
$$

If the batch is sampled uniformly under the appropriate convention,

$$
\mathbb E[\widehat g_t\mid\theta_t]
=\nabla L(\theta_t).
$$

`[EXPLANATION]` The remaining computational problem is how to evaluate every component of $\nabla_\theta L$ without separately expanding every chain-rule path.

## 9. Backpropagation on a computational graph

### 9.1 Scalar DAG

Let a directed acyclic graph contain scalar nodes

$$
u^{(1)},\ldots,u^{(m)}
$$

in topological order. For $i>m_0$,

$$
u^{(i)}
=\phi_i\left((u^{(j)})_{j\in\operatorname{Pa}(i)}\right).
$$

Let the final scalar loss be

$$
J=u^{(m)}.
$$

Define the adjoint

$$
\bar u^{(j)}
:=\frac{\partial J}{\partial u^{(j)}}.
$$

### 9.2 Local chain-rule recurrence

`[DIRECT]` The chain rule gives

$$
\boxed{
\bar u^{(j)}
=
\sum_{i:j\in\operatorname{Pa}(i)}
\bar u^{(i)}
\frac{\partial u^{(i)}}{\partial u^{(j)}}.
}
$$

`[EXPLANATION]` A node can influence the loss through several children, so its derivative is the sum of contributions over all outgoing edges.

### 9.3 Why naive recursion can be exponential

A purely recursive derivative routine can recompute the same downstream derivative once for every path from a node to the loss. A graph can contain only $O(L)$ nodes and edges but exponentially many distinct directed paths.

`[SYNTHESIS]` The expensive object is the unfolded recursion tree, not the original computational graph.

### 9.4 Dynamic programming solution

Backpropagation performs:

1. a forward topological pass that computes and stores every node value;
2. initialization $\bar u^{(m)}=1$ and all other adjoints to zero;
3. one reverse-topological pass;
4. for every edge $j\to i$, the accumulation

$$
\bar u^{(j)}
\mathrel{+}=
\bar u^{(i)}
\frac{\partial u^{(i)}}{\partial u^{(j)}}.
$$

If every local derivative is constant-time, forward and backward passes each take

$$
O(|E|)
$$

time.

`[CONCLUSION]` Backpropagation is reverse-mode automatic differentiation implemented as dynamic programming on a DAG.

## 10. Worked one-hidden-layer gradients

Consider

$$
z=Wx+b,
\qquad
h=\rho(z),
$$

$$
s=v^\top h+c,
\qquad
J=\ell(s,y).
$$

Let

$$
\delta=\frac{\partial\ell(s,y)}{\partial s},
$$

and define the ReLU mask

$$
r_j=\mathbf1\{z_j>0\}.
$$

### 10.1 Output layer

Since $s=v^\top h+c$,

$$
\nabla_vJ=\delta h,
\qquad
\frac{\partial J}{\partial c}=\delta.
$$

### 10.2 Hidden activation and preactivation

$$
\nabla_hJ=\delta v.
$$

Because $h=\rho(z)$ coordinatewise,

$$
\nabla_zJ
=(\delta v)\odot r.
$$

### 10.3 Affine layer

From $z=Wx+b$,

$$
\nabla_bJ=\nabla_zJ,
$$

$$
\boxed{
\nabla_WJ
=(\nabla_zJ)x^\top.
}
$$

`[SHAPE CHECK]`

- $\nabla_zJ\in\mathbb R^q$;
- $x\in\mathbb R^d$;
- $(\nabla_zJ)x^\top\in\mathbb R^{q\times d}$, matching $W$.

`[EXPLANATION]` The gradient of a dense layer weight matrix is an outer product: error arriving at the layer times input entering the layer.

## 11. Conceptual connections

`[CONNECTION TO LECTURE 3]` Transformer blocks remain computational graphs trained by the same reverse-mode principle. Attention changes the local operations, not the chain rule.

`[CONNECTION TO OPTIMIZATION LECTURES]` Backpropagation produces a gradient. SGD, Adam, Muon, and other optimizers decide how that gradient changes parameters.

`[CONNECTION TO LORA]` If $W=W_0+BA$, backpropagation sends gradients through the product into $A$ and $B$ while $W_0$ remains frozen.

## 12. Common confusions

### Universality does not imply trainability

An architecture can contain a good approximation while optimization fails to find it.

### Backpropagation is not an optimizer

Backpropagation computes derivatives. SGD or Adam uses those derivatives to update parameters.

### Depth is not automatically useful

Depth increases compositional capacity but may make optimization harder without suitable parameterization, normalization, or residual structure.

### Zero training loss is not generalization

Empirical risk only measures the observed sample. Statistical assumptions are needed to connect it to new data.

## 13. Open questions

1. Which real data distributions avoid the worst-case curse of dimensionality?
2. When does a learned representation enlarge the effective margin of a simple classifier?
3. Which useful functions are represented exponentially more efficiently by depth?
4. How does backpropagation interact with parameter symmetries and flat directions?
5. What information must be stored in the forward pass, and what can be recomputed to save memory?
6. Why do highly overparameterized networks often optimize more easily than smaller ones?

## 14. What to verify manually

- Reproduce the perceptron mistake proof without looking at the handout.
- Build the exact-match threshold unit for a three-bit pattern.
- Distinguish universal approximation from approximation rate.
- Draw a DAG with two paths to one node and show where naive recursion repeats work.
- Check every shape in the one-hidden-layer gradient derivation.
- Replace the synthesis paragraphs with your own explanations after studying the handout.

## Source trail

- Official Lecture 1 handout: <https://noahgol.github.io/teaching/cs395t-f26/lecture1.pdf>
- Official Lecture 1 slides: <https://noahgol.github.io/teaching/cs395t-f26/lecture-01-slides.pdf>
- Course schedule and suggested readings: <https://noahgol.github.io/teaching/cs395t-f26/>

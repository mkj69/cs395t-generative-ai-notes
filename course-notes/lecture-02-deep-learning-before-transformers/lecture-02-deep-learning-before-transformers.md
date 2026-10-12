---
title: "Lecture 02 — Deep Learning Before Transformers"
slug: "lecture-02"
date: "2026-08-26"
updated: "2026-10-12"
status: "draft"
type: "learning-note"
course: "CS 395T"
source: "https://noahgol.github.io/teaching/cs395t-f26/lecture02.pdf"
summary: "An AI-assisted reconstruction of pre-Transformer deep learning, from convolutional vision and VAEs/GANs through recurrent language models, attention, and adversarial robustness."
public: true
tags:
  - convolutional networks
  - image generation
  - VAE
  - GAN
  - recurrent neural networks
  - LSTM
  - attention
  - adversarial examples
related_research_notes: []
authorship: "ai-assisted"
---

> **AI-assisted first draft.** This note reconstructs the official 139-slide Lecture 2 deck and expands its equations into a connected study narrative. It is not a slide transcript and has not yet been manually verified or rewritten by the author. Statements tagged `[DIRECT]` follow the deck; `[DERIVATION]`, `[EXPLANATION]`, and `[SYNTHESIS]` make the reasoning explicit.

## Source boundary

`[DIRECT]` Official slides: <https://noahgol.github.io/teaching/cs395t-f26/lecture02.pdf>.

`[DIRECT]` The deck is organized into three broad parts:

1. image modeling: convolutional networks, classification, VAEs, GANs, and generative metrics;
2. language modeling before Transformers: tokenization, RNNs, LSTMs, encoder-decoder models, and attention;
3. adversarial examples and adversarial training.

`[SOURCE BOUNDARY]` The slide deck uses repeated build slides, so its 139 pages contain fewer than 139 distinct conceptual units. This note consolidates those builds into one derivation per idea. It does not reproduce the deck's images.

## Reading map

| Part | Representation | Training signal | Central limitation |
| --- | --- | --- | --- |
| Convnets | Spatial feature maps | Cross-entropy | Locality helps, but depth must remain trainable |
| VAE | Probabilistic latent variable | ELBO | Likelihood approximation can produce smooth samples |
| GAN | Implicit generator and discriminator | Minimax game | Training instability and mode coverage |
| RNN/LSTM | Recurrent hidden state | Token cross-entropy | Long-range gradients and fixed-state bottlenecks |
| Attention | Weighted access to all source states | End-to-end likelihood | Still sequential when coupled to an RNN |
| Adversarial training | Worst-case neighborhood | Min-max loss | Robustness is threat-model-specific and expensive |

# Part I — Image modeling

## 1. Why images need structural priors

An image of height $H$, width $W$, and $C$ channels can be flattened into

$$
x\in\mathbb R^{HWC}.
$$

A dense layer with $m$ outputs requires

$$
mHWC
$$

weights.

For a $224\times224\times3$ image and $m=4096$ hidden units, this is approximately

$$
224\cdot224\cdot3\cdot4096
\approx 6.17\times10^8
$$

weights in one layer.

`[EXPLANATION]` Flattening also discards the fact that nearby pixels are related and that a useful local pattern may appear anywhere in the image.

Convolution builds in two priors:

- **locality:** nearby pixels are processed together;
- **weight sharing:** the same detector is applied across spatial positions.

## 2. Convolution and equivariance

### 2.1 Single-channel discrete convolution

For image $x\in\mathbb R^{n\times n}$ and kernel $w\in\mathbb R^{k\times k}$, a valid cross-correlation-style operation is

$$
z_{i,j}
=
\sum_{a=0}^{k-1}
\sum_{b=0}^{k-1}
w_{a,b}x_{i+a,j+b}.
$$

Deep-learning libraries usually call this operation convolution even when the kernel is not spatially flipped.

### 2.2 Stride, padding, and output size

With input width $n$, kernel width $k$, padding $p$, and stride $s$, output width is

$$
n_{mathrm{out}}
=
\left\lfloor
\frac{n+2p-k}{s}
\right\rfloor+1.
$$

`[CHECK]` This shape formula should be verified for every network stage; off-by-one padding mistakes silently alter feature-map sizes.

### 2.3 Multi-channel convolutional layer

For input

$$
h^{(\ell-1)}\in\mathbb R^{n\times n\times C}
$$

and $F$ learned filters

$$
W^{(\ell)}\in\mathbb R^{k\times k\times C\times F},
$$

the preactivation of output channel $f$ is

$$
z_{i,j,f}^{(\ell)}
=
\sum_{a,b,c}
W_{a,b,c,f}^{(\ell)}
h_{i+a,j+b,c}^{(\ell-1)}
+b_f^{(\ell)}.
$$

Then

$$
h^{(\ell)}=\sigma(z^{(\ell)}).
$$

Parameter count is

$$
k^2CF+F,
$$

independent of the spatial image size.

### 2.4 Translation equivariance

Let $T_\delta$ translate an image and let $K*x$ denote convolution. Ignoring boundaries,

$$
K*(T_\delta x)
=T_\delta(K*x).
$$

`[EXPLANATION]` Moving an object moves its feature response; the same detector need not be relearned at every pixel. Equivariance is not invariance. Pooling or aggregation can make the final prediction less sensitive to position.

## 3. From feature maps to classification

Let a backbone produce representation

$$
h=f_\theta(x)\in\mathbb R^d.
$$

A linear classifier gives logits

$$
z=Wh+b\in\mathbb R^K.
$$

Softmax probabilities are

$$
p_k
=
\frac{e^{z_k}}
{\sum_{j=1}^{K}e^{z_j}}.
$$

For label $y$, cross-entropy is

$$
L=-\log p_y.
$$

### 3.1 Cross-entropy gradient

`[DERIVATION]` Since

$$
\log p_y=z_y-\log\sum_j e^{z_j},
$$

we have

$$
\frac{\partial L}{\partial z_k}
=p_k-\mathbf1\{k=y\}.
$$

`[EXPLANATION]` The correct logit receives gradient $p_y-1$, while every incorrect logit receives $p_k$. Confidently wrong predictions produce large correction.

### 3.2 Maximum-likelihood interpretation

For dataset $\{(x_i,y_i)\}_{i=1}^{n}$,

$$
\min_\theta
-\sum_i\log p_\theta(y_i\mid x_i)
$$

is equivalent to maximum conditional likelihood.

`[SYNTHESIS]` Convolution specifies the architecture's spatial bias; cross-entropy specifies the statistical fitting objective; SGD and backpropagation perform optimization. These are separate design choices.

## 4. The path from LeNet to ResNet

### 4.1 Scaling data and compute

`[DIRECT]` The slides use MNIST, CIFAR, and ImageNet to illustrate increasing resolution, dataset scale, and class complexity. LeNet established convolutional digit recognition; AlexNet demonstrated the impact of scaling convnets on ImageNet with GPUs and modern training choices.

`[SYNTHESIS]` Architectural progress cannot be separated cleanly from data, hardware, implementation, and optimization. Historical benchmark jumps bundle these factors together.

### 4.2 VGG and repeated small kernels

Two $3\times3$ convolution layers have an effective receptive field of $5\times5$ but introduce a nonlinearity between them. With equal channel count $C$, their weight count is roughly

$$
2\cdot9C^2=18C^2,
$$

versus

$$
25C^2
$$

for one $5\times5$ layer.

`[EXPLANATION]` Repeated small kernels can be more parameter-efficient and more expressive because of the intermediate activation.

### 4.3 The degradation problem

`[DIRECT]` Very deep plain networks can have higher *training* error than shallower ones. This is not ordinary overfitting because optimization itself becomes worse.

### 4.4 Residual learning

A residual block computes

$$
y=x+F(x;\theta).
$$

Instead of directly learning desired map $H(x)$, the branch learns

$$
F(x)=H(x)-x.
$$

If the identity is appropriate, setting $F\approx0$ is easy in principle.

### 4.5 Gradient path through a residual block

For loss $L(y)$,

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\left(I+J_F(x)\right),
$$

where $J_F$ is the Jacobian of the residual branch.

`[EXPLANATION]` The identity term supplies a direct gradient path even when $J_F$ is small. Residual connections do not eliminate every optimization problem, but they change both representation and gradient flow.

## 5. From classification to generation

Classification models

$$
p(y\mid x),
$$

while unconditional generation models a high-dimensional distribution

$$
p(x).
$$

`[EXPLANATION]` A classifier only needs to separate a finite set of labels. A generator must assign coherent probability across the space of possible images and provide a procedure for sampling new ones.

## 6. Variational autoencoders

### 6.1 Latent-variable model

Let

$$
z\sim p(z),
\qquad
x\sim p_\theta(x\mid z).
$$

The marginal likelihood is

$$
p_\theta(x)
=\int p(z)p_\theta(x\mid z)\,dz,
$$

which can be difficult to evaluate and differentiate exactly.

Introduce an encoder

$$
q_\phi(z\mid x)
$$

to approximate the posterior.

### 6.2 Deriving the ELBO

`[DERIVATION · Step 1]` Multiply and divide inside the integral:

$$
\begin{aligned}
\log p_\theta(x)
&=\log
\int q_\phi(z\mid x)
\frac{p(z)p_\theta(x\mid z)}{q_\phi(z\mid x)}\,dz.
\end{aligned}
$$

`[DERIVATION · Step 2]` Jensen's inequality gives

$$
\log p_\theta(x)
\ge
\mathbb E_{q_\phi(z\mid x)}
\left[
\log\frac{p(z)p_\theta(x\mid z)}{q_\phi(z\mid x)}
\right].
$$

Therefore the evidence lower bound is

$$
\boxed{
\mathcal L_{\mathrm{ELBO}}
=
\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}
(q_\phi(z\mid x)\|p(z)).
}
$$

The first term rewards reconstruction. The second regularizes the approximate posterior toward the prior.

### 6.3 Exact gap to log likelihood

Another identity is

$$
\log p_\theta(x)
=mathcal L_{\mathrm{ELBO}}
+D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)
\|p_\theta(z\mid x)
\right).
$$

Because KL is nonnegative, the ELBO is indeed a lower bound. It is tight exactly when the approximate posterior equals the true posterior almost everywhere.

### 6.4 Reparameterization trick

For a diagonal Gaussian encoder,

$$
q_\phi(z\mid x)
=\mathcal N
(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))).
$$

Sample

$$
\varepsilon\sim\mathcal N(0,I),
$$

and set

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon.
$$

`[EXPLANATION]` Randomness is moved into $\varepsilon$, whose distribution is independent of $\phi$. The sampled $z$ becomes a differentiable function of encoder outputs.

For standard normal prior, the KL has closed form

$$
D_{\mathrm{KL}}(q_\phi\|p)
=\frac12\sum_j
\left(
\mu_j^2+\sigma_j^2-\log\sigma_j^2-1
\right).
$$

## 7. Generative adversarial networks

### 7.1 Two-player objective

Let generator $G$ transform noise $z\sim p_z$ into samples, and discriminator $D(x)\in(0,1)$ estimate whether $x$ came from the data.

The original minimax objective is

$$
\min_G\max_D
V(D,G)
=
\mathbb E_{x\sim p_{\mathrm{data}}}[\log D(x)]
+
\mathbb E_{z\sim p_z}
[\log(1-D(G(z)))].
$$

### 7.2 Optimal discriminator for a fixed generator

Let $p_g$ be the generator distribution. For each $x$, maximize

$$
p_{\mathrm{data}}(x)\log D(x)
+p_g(x)\log(1-D(x)).
$$

Differentiate with respect to $D(x)$:

$$
\frac{p_{\mathrm{data}}(x)}{D(x)}
-
\frac{p_g(x)}{1-D(x)}=0.
$$

Solving gives

$$
\boxed{
D^*(x)
=
\frac{p_{\mathrm{data}}(x)}
{p_{\mathrm{data}}(x)+p_g(x)}.
}
$$

### 7.3 Why the ideal game matches distributions

Substituting $D^*$ into the value function yields

$$
V(D^*,G)
=-log4
+2\,D_{\mathrm{JS}}
(p_{\mathrm{data}}\|p_g).
$$

Jensen-Shannon divergence is nonnegative and zero exactly when the distributions match, so the idealized global optimum satisfies

$$
p_g=p_{\mathrm{data}}.
$$

`[LIMIT]` This equilibrium argument assumes an optimal discriminator and distribution-level optimization. Finite networks and alternating stochastic updates need not behave like the ideal game.

### 7.4 Saturation and non-saturating generator loss

Early in training, $D(G(z))$ may be near zero. Minimizing

$$
\log(1-D(G(z)))
$$

can yield weak gradients. A common alternative maximizes

$$
\log D(G(z)),
$$

which has the same desired fixed point but stronger early gradients.

### 7.5 Mode collapse

`[EXPLANATION]` The generator may map many noise values to a small collection of plausible outputs. Sample fidelity can look high while diversity is poor.

`[SYNTHESIS]` A generative metric must measure both fidelity and coverage; either one alone can be gamed.

## 8. Evaluating generated images

### 8.1 Inception Score

Inception Score uses a pretrained classifier:

$$
\operatorname{IS}
=exp
\left(
\mathbb E_x
D_{\mathrm{KL}}(p(y\mid x)\|p(y))
\right).
$$

Low entropy $p(y\mid x)$ rewards individually recognizable images; high entropy marginal $p(y)$ rewards class diversity.

`[LIMIT]` IS does not compare directly with real data and inherits the classifier's biases.

### 8.2 Fréchet Inception Distance

Embed real and generated images into feature vectors. Approximate each feature distribution as Gaussian:

$$
\mathcal N(\mu_r,\Sigma_r),
\qquad
\mathcal N(\mu_g,\Sigma_g).
$$

FID is

$$
\operatorname{FID}
=
\|\mu_r-\mu_g\|_2^2
+
\operatorname{Tr}
\left(
\Sigma_r+\Sigma_g
-2(\Sigma_r\Sigma_g)^{1/2}
\right).
$$

Lower is better.

`[LIMIT]` Finite-sample estimation is biased; scores depend on the feature extractor and preprocessing; matching two moments does not imply matching full distributions.

# Part II — Language modeling before Transformers

## 9. Tokens and embeddings

A tokenizer maps text to symbols

$$
w_1,\ldots,w_T,
\qquad
w_t\in\{1,\ldots,|\mathcal V|\}.
$$

An embedding matrix

$$
E\in\mathbb R^{|\mathcal V|\times d}
$$

maps a token ID to

$$
x_t=E_{w_t,:}.
$$

`[SYNTHESIS]` Tokenization defines the atomic prediction problem. It changes sequence length, vocabulary size, softmax cost, and which strings share statistical strength.

### 9.1 Word, character, and subword units

- Word tokenization produces shorter sequences but suffers from unknown words and a large vocabulary.
- Character tokenization avoids unknown words but produces much longer sequences.
- Subword tokenization interpolates between them.

### 9.2 Byte-pair encoding

A basic BPE training procedure is:

1. begin with characters or bytes as symbols;
2. count adjacent symbol pairs in a corpus;
3. merge the most frequent pair;
4. replace its occurrences;
5. repeat for a fixed number of merges.

`[EXPLANATION]` Frequent strings become single tokens while rare words remain decomposable. The learned vocabulary is a compression code shaped by the training corpus.

## 10. Recurrent neural networks

A simple RNN updates

$$
h_t
=\sigma(W_{hh}h_{t-1}+W_{xh}x_t+b_h),
$$

and predicts

$$
\hat y_t
=\operatorname{softmax}(W_{hy}h_t+b_y).
$$

The same matrices are reused for every time step.

`[EXPLANATION]` Parameter sharing allows variable sequence length and expresses a time-homogeneous transition rule. The hidden state is a fixed-size summary of the prefix.

## 11. Backpropagation through time

Unrolling the recurrence produces a depth-$T$ computational graph with tied parameters.

For a final-time loss $L_T$, the sensitivity to an earlier state is

$$
\frac{\partial L_T}{\partial h_t}
=
\frac{\partial L_T}{\partial h_T}
\prod_{k=t+1}^{T}
\frac{\partial h_k}{\partial h_{k-1}}.
$$

For the simple RNN,

$$
\frac{\partial h_k}{\partial h_{k-1}}
=
\operatorname{diag}(\sigma'(a_k))W_{hh}.
$$

### 11.1 Vanishing and exploding gradients

If relevant Jacobian singular values behave roughly like $\lambda$, then

$$
\left\|
\prod_{k=t+1}^{T}
\frac{\partial h_k}{\partial h_{k-1}}
\right\|
\approx
\lambda^{T-t}.
$$

- $\lambda<1$: gradients vanish exponentially;
- $\lambda>1$: gradients explode exponentially.

Gradient clipping controls explosion:

$$
g\leftarrow
g\cdot
\min\left(1,\frac{c}{\|g\|}
\right).
$$

`[LIMIT]` Clipping does not fix vanishing gradients. Truncated BPTT reduces cost but prevents gradients from crossing the truncation boundary.

## 12. LSTM memory path

An LSTM computes gates

$$
i_t=\sigma(W_i[h_{t-1},x_t]+b_i),
$$

$$
f_t=\sigma(W_f[h_{t-1},x_t]+b_f),
$$

$$
o_t=\sigma(W_o[h_{t-1},x_t]+b_o),
$$

and candidate

$$
g_t=\tanh(W_g[h_{t-1},x_t]+b_g).
$$

The cell and hidden state update as

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,
$$

$$
h_t=o_t\odot\tanh(c_t).
$$

### 12.1 Gradient through the cell

Holding gate values fixed for the local derivative,

$$
\frac{\partial c_t}{\partial c_{t-1}}
=\operatorname{diag}(f_t).
$$

Across many steps,

$$
\frac{\partial c_T}{\partial c_t}
\approx
\prod_{k=t+1}^{T}
\operatorname{diag}(f_k).
$$

`[EXPLANATION]` When forget gates stay near one, the cell provides an approximately additive path along which information and gradients can persist. The network learns when to remember, overwrite, or expose memory.

## 13. Autoregressive language modeling

By the chain rule,

$$
p_\theta(w_{1:T})
=\prod_{t=1}^{T}
p_\theta(w_t\mid w_{<t}).
$$

Training minimizes token negative log likelihood:

$$
L(\theta)
=-
\sum_{t=1}^{T}
\log p_\theta(w_t\mid w_{<t}).
$$

Average per-token NLL is

$$
\overline L
=-rac1T\sum_t
\log p_\theta(w_t\mid w_{<t}).
$$

Perplexity is

$$
\boxed{
\operatorname{PPL}=\exp(\overline L).
}
$$

`[EXPLANATION]` If the model assigned equal probability to $m$ choices at every position, perplexity would be $m$. This interpretation is only approximate for nonuniform distributions.

`[CHECK]` Perplexities are comparable only under the same tokenization and evaluation protocol.

## 14. Encoder-decoder sequence models

For source sequence $x_{1:T}$, an encoder produces a fixed vector

$$
c=h_T.
$$

A decoder defines

$$
p(y_{1:T'}\mid x_{1:T})
=
\prod_{t=1}^{T'}
p(y_t\mid c,y_{<t}).
$$

`[DIRECT]` Early sequence-to-sequence systems showed that recurrent encoder-decoder models could learn translation end to end.

### 14.1 Fixed-vector bottleneck

Every source detail must pass through one vector $c$. As source length grows, the encoder must compress more information into the same dimensionality, and the decoder cannot directly revisit individual source positions.

`[SYNTHESIS]` Attention solves an information-access problem: instead of requiring one vector to contain every detail equally, let the decoder retrieve different source states at different output steps.

## 15. Additive attention

Let bidirectional encoder annotations be

$$
h_j=[\overrightarrow h_j;\overleftarrow h_j].
$$

At decoder step $i$, compute scores

$$
e_{ij}
=v^\top
\tanh(Ws_{i-1}+Uh_j).
$$

Normalize across source positions:

$$
\alpha_{ij}
=
\frac{\exp(e_{ij})}
{\sum_{k=1}^{T}\exp(e_{ik})}.
$$

The context is

$$
c_i
=\sum_{j=1}^{T}
\alpha_{ij}h_j.
$$

### 15.1 Interpretation

- $e_{ij}$ measures compatibility between decoder state and source position.
- $\alpha_{ij}$ is a probability distribution over source positions.
- $c_i$ is a differentiable weighted retrieval from encoder memory.

`[EXPLANATION]` The same mechanism can attend over spatial CNN locations for image captioning. “Position” need not mean a word; it can mean any indexed collection of feature vectors.

### 15.2 Dot-product and general scoring

Other scores include

$$
\operatorname{score}(s,h)=s^\top h,
$$

or

$$
\operatorname{score}(s,h)=s^\top Wh.
$$

`[CONNECTION TO LECTURE 3]` Transformer attention removes recurrence from the main sequence-mixing operation and computes many query-key interactions in parallel, but the normalize-and-weight pattern is already visible here.

## 16. Dataset scale shapes architecture conclusions

`[DIRECT]` The slides review early language-modeling, translation, and vision-language datasets, ranging from small Penn Treebank-scale corpora to the One Billion Word Benchmark, WMT translation pairs, and MSCOCO captions.

`[SYNTHESIS]` A model result is conditional on dataset scale and preprocessing. Word-level unknown tokens, sentence-length distributions, and benchmark reuse can dominate apparent architectural differences.

# Part III — Adversarial examples

## 17. Threat model

Given correctly classified $(x,y)$ and classifier logits $f_\theta(x)$, find perturbation $\delta$ such that

$$
\|\delta\|_\infty\le\varepsilon
$$

but

$$
\arg\max_k f_\theta(x+\delta)_k\ne y.
$$

`[EXPLANATION]` Robustness is defined relative to a norm, radius, attack knowledge, and allowed computation. “Adversarially robust” without a threat model is incomplete.

## 18. Fast Gradient Sign Method

Linearize the loss around $x$:

$$
L(\theta,x+\delta,y)
\approx
L(\theta,x,y)
+\nabla_xL(\theta,x,y)^\top\delta.
$$

To maximize this linear term under $\|\delta\|_\infty\le\varepsilon$, choose each coordinate at its bound with matching sign:

$$
\boxed{
\delta_{\mathrm{FGSM}}
=\varepsilon\operatorname{sign}
(\nabla_xL(\theta,x,y)).
}
$$

Thus

$$
x'=x+delta_{\mathrm{FGSM}}.
$$

`[DERIVATION]` Hölder's inequality gives

$$
g^\top\delta
\le\|g\|_1\|\delta\|_\infty
\le\varepsilon\|g\|_1,
$$

and the sign choice achieves equality.

## 19. Adversarial training

Replace ordinary empirical risk with the robust objective

$$
\boxed{
\min_\theta
\mathbb E_{(x,y)\sim\mathcal D}
\left[
\max_{\delta\in\mathcal S}
L(\theta,x+\delta,y)
\right].
}
$$

- The inner problem searches for a damaging allowed perturbation.
- The outer problem updates parameters to reduce loss on those perturbations.

### 19.1 Projected gradient descent attack

Initialize within the threat set and iterate

$$
\delta^{k+1}
=
\Pi_{\mathcal S}
\left(
\delta^k
+\alpha
\operatorname{sign}
(\nabla_\delta L(\theta,x+\delta^k,y))
\right).
$$

For an $\ell_\infty$ ball, projection clips each coordinate to $[-\varepsilon,\varepsilon]$ and usually also clips $x+\delta$ to the valid input range.

### 19.2 Training loop

```python
for x, y in loader:
    delta = uniform(-epsilon, epsilon, shape=x.shape)

    for _ in range(attack_steps):
        delta.requires_grad_(True)
        attack_loss = cross_entropy(model(x + delta), y)
        grad = gradient(attack_loss, delta)
        delta = clip(delta + step_size * sign(grad), -epsilon, epsilon)
        delta = clip(x + delta, 0, 1) - x

    robust_loss = cross_entropy(model(x + stop_gradient(delta)), y)
    optimizer.zero_grad()
    robust_loss.backward()
    optimizer.step()
```

`[LIMIT]` PGD is an approximate inner maximizer. Robustness to one attack is not proof of robustness to every attack, norm, or larger radius.

## 20. Unifying lessons from the pre-Transformer era

### 20.1 Inductive bias controls efficiency

Convolution shares spatial detectors; recurrence shares a transition across time; attention provides content-dependent access. Each architecture makes some functions easier to represent than a generic dense network.

### 20.2 Gradient paths control trainability

Residual identities, LSTM cells, and attention shortcuts all create alternative information and gradient routes. Expressivity alone does not explain optimization.

### 20.3 Objectives define what “good” means

- Cross-entropy rewards conditional likelihood.
- ELBO trades reconstruction against latent regularization.
- GAN training seeks an adversarial equilibrium.
- Adversarial training optimizes worst-case neighborhood loss.

### 20.4 Evaluation metrics are partial views

Accuracy, perplexity, IS, FID, and robust accuracy measure different projections of model behavior. No one scalar captures fidelity, diversity, calibration, efficiency, and robustness simultaneously.

## 21. Common confusions

### Convolution is not automatically invariant

It is equivariant before pooling or aggregation.

### The ELBO is not the log likelihood

Its gap is the posterior KL. A flexible decoder can also ignore the latent variable.

### The GAN equilibrium is not a training guarantee

The divergence result assumes an optimal discriminator for each generator.

### LSTM does not eliminate all long-context problems

It creates a controllable memory path, but gates can still forget and the state has fixed size.

### Attention weights are not automatically causal explanations

They are learned mixture weights in the computation. Their interpretability depends on the question and model.

### Adversarial robustness is local and specified

Robustness inside one norm ball does not imply robustness to other perturbations or natural distribution shifts.

## 22. Open questions

1. Which image symmetries should be built in, and which should be learned from augmentation?
2. When do likelihood-based and perceptual-quality objectives disagree most strongly?
3. How should diversity be measured without relying on a fixed pretrained classifier?
4. Which properties of residual networks made scaling depth practical beyond simple gradient flow?
5. What information is systematically lost in a fixed-dimensional recurrent state?
6. When do attention weights correspond to stable alignments rather than brittle correlations?
7. Can one threat model predict robustness under another?
8. Which historical limitations were architectural and which were consequences of data or hardware?

## 23. What to verify manually

- Compute convolution output shapes for several stride and padding choices.
- Re-derive the softmax-cross-entropy gradient.
- Derive the ELBO both with Jensen's inequality and with the posterior-KL identity.
- Re-derive the optimal GAN discriminator and the Jensen-Shannon objective.
- Trace one RNN gradient through three time steps and inspect the Jacobian product.
- Explain how the LSTM forget gate changes the cell-state derivative.
- Derive FGSM from the duality between $\ell_\infty$ and $\ell_1$ norms.
- Replace the synthesis sections with your own connections after reviewing the official slides.

## Source trail

- Official Lecture 2 slides: <https://noahgol.github.io/teaching/cs395t-f26/lecture02.pdf>
- Course schedule and suggested readings: <https://noahgol.github.io/teaching/cs395t-f26/>

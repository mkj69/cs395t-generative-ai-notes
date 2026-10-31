---
title: 'Lecture 17: Diffusion I: Score Matching and Diffusion Models'
slug: lecture-17
date: '2026-10-21'
updated: '2026-10-31'
status: checked
type: learning-note
course: CS 395T
source: https://noahgol.github.io/teaching/cs395t-f26/
summary: 从高斯去噪例子理解 score、逆时间 SDE、去噪 score matching 和 probability flow ODE，逐步核对符号与二分之一系数。
public: true
language: zh-CN
tags:
- Diffusion
- Score matching
- Reverse SDE
- Probability flow ODE
related_research_notes: []
authorship: ai-assisted
---

> **AI 辅助学习笔记。** 本文是独立讲解与推导，不是讲义全文译文，也不代表作者已经完成个人复习。讲义来源：[课程原始 PDF](https://noahgol.github.io/teaching/cs395t-f26/lecture16-17-v2.pdf)。第 16、17 课共用一份讲义。公式已与相应来源核对，未做人工终审。

`[DIRECT]` 标明来源直接支持的简述；`[DERIVATION]` 标明数学推导；`[SYNTHESIS]` 标明概念连接与解释。自行构造的数值例子不代表论文实验。

[返回 Diffusion 单元导读](diffusion-unit.html)

## 阅读问题：不知道数据密度，如何学习回到数据的方向

`[SYNTHESIS]` 训练集提供样本，却通常不提供密度函数。这节课的核心困难是：采样需要一个与未知分布有关的方向场，但训练标签必须能从样本算出来。下面通过高斯例子、条件期望与概率流重建这个联系，不按原讲义段落改写。

采用数据到噪声的前向时间 $t$。$X_0$ 是数据，$X_t$ 是加噪状态，$s_*(t,x)=\nabla_x\log p_t(x)$ 是真实 score，$s_\theta$ 是估计值。为避免反向时间符号混乱，生成时单独使用 $r=T-t$。

## Score 是空间导数，不是概率分数

`[DERIVATION]` 对一维 $p(x)=\mathcal N(m,\sigma^2)$，

$$\log p(x)=C-\frac{(x-m)^2}{2\sigma^2},\qquad \partial_x\log p(x)=-\frac{x-m}{\sigma^2}.$$

它在均值右边为负、左边为正，单位是位置单位的倒数。取 $m=2,\sigma^2=4,x=6$，score 等于 $-1$。这描述局部 log-density 上升方向，不表示“该样本质量得分为负”。混合分布的 score 不一定指向某个固定训练样本。

## 逆向运动为什么不只是把漂移变号

`[DERIVATION]` 对状态无关噪声的前向 SDE $dX_t=f(t,X_t)dt+g(t)dB_t$，在足够正则、密度为正并满足时间反演条件时，逆时间漂移为

$$dY_r=[-f(T-r,Y_r)+g(T-r)^2s_*(T-r,Y_r)]dr+g(T-r)d\bar B_r.$$

反向布朗运动相对于反向滤过定义，不能简单解释成把训练时每一步噪声原样取负。score 修正的必要性可以通过 Fokker–Planck 检查：用 $p\nabla\log p=\nabla p$，新增漂移对密度贡献 $-g^2\Delta p$；再加上反向 SDE 自带的 $+g^2\Delta p/2$，合起来正好是逆向所需的 $-g^2\Delta p/2$。

只核对每个时刻的密度不能单独证明整个时间反演过程相同，完整结论还需要转移概率与唯一性条件。该推导解释系数，不能替代全部定理假设。

## 从未知的 marginal score 到可计算的标签

令 $X_t=\alpha_tX_0+\sigma_t\epsilon$，$\epsilon\sim\mathcal N(0,I)$ 与 $X_0$ 独立。这里 $\alpha_t$ 是信号系数，不是某些 DDPM 文献中另一个同名的单步参数。

`[DERIVATION]` 固定 $X_0$ 后，条件密度是高斯，其 score 为

$$\nabla_x\log p_t(x\mid X_0)=-\frac{x-\alpha_tX_0}{\sigma_t^2},\qquad U=-\frac\epsilon{\sigma_t}.$$

对混合密度求导并用贝叶斯公式，可得

$$s_*(t,x)=\mathbb E[U\mid X_t=x,t].$$

注意这里是给定当前噪声状态后的后验平均，不是用数据分布无条件平均所有标签。

对任意预测 $s_\theta$，平方误差分解为

$$\mathbb E\|s_\theta-U\|^2=\mathbb E\|s_\theta-s_*\|^2+\mathbb E\|U-s_*\|^2.$$

最后一项与参数无关，所以可以直接使用 $U$ 做监督。标签有随机性并不妨碍均方误差的最优解是条件均值。

## 噪声预测与去噪均值

`[DERIVATION]` 将 $s_\theta=-\epsilon_\theta/\sigma_t$ 代入，若 score 损失乘以 $\sigma_t^2$，就得到

$$\mathbb E\|\epsilon_\theta(t,\alpha_tX_0+\sigma_t\epsilon)-\epsilon\|^2.$$

一次训练只需抽取数据、时间和噪声，直接构造带噪输入，不必模拟整条前向随机路径。训练仍需普通反向传播，生成仍需数值采样。

由同一个条件均值恒等式还得到

$$\mathbb E[\alpha_tX_0\mid X_t=x]=x+\sigma_t^2s_*(t,x).$$

若 $\alpha_t\ne0$，除以 $\alpha_t$ 才是 $X_0$ 的后验均值。不能在 VP 设置里直接把右边叫成原始干净样本。

`[DERIVATION · 算例]` 取 $X_0\sim\mathcal N(0,1)$，$X=X_0+\epsilon$ 且噪声方差为1。于是 $X\sim\mathcal N(0,2)$，score 为 $-x/2$，去噪均值是 $x-x/2=x/2$。当观测为2时，最优均方预测是1，并不恢复某次抽样隐藏的确切 $X_0$。

## OU、VE 和 VP：比较时看系数，不只看名称

| 过程 | 漂移与噪声系数 | 条件均值与方差 |
| --- | --- | --- |
| OU | $f=-\kappa x$, $g=\sigma$ | $e^{-\kappa t}x_0$；$\frac{\sigma^2}{2\kappa}(1-e^{-2\kappa t})$ |
| VE | $f=0$, $g^2=(\sigma_t^2)'$ | $x_0$；$\sigma_t^2$，假设初始附加噪声为0 |
| VP | $f=-\beta(t)x/2$, $g=\sqrt{\beta(t)}$ | $\alpha_tx_0$；$1-\alpha_t^2$ |

VP 中 $\alpha_t=\exp[-\frac12\int_0^t\beta(s)ds]$。“variance preserving”并不保证任意初始分布在每个时刻都有单位方差。单位协方差的初始条件才会在这一计算中保持单位协方差。

## 确定性 ODE 为什么出现二分之一

`[DERIVATION]` 将前向 SDE 的概率流除以密度，得到

$$v_{\mathrm{PF}}(t,x)=f(t,x)-\frac12g(t)^2s_*(t,x).$$

ODE $\dot X_t=v_{\mathrm{PF}}(t,X_t)$ 在合适唯一性条件下复现同一族边缘密度。它与 SDE 的跨时间联合分布一般不同。反向时间下速度是 $-f+g^2s_*/2$，不能只从反向 SDE 删去随机项而保留完整 $g^2s_*$。

生成误差至少包括三部分：终端噪声近似误差、score 估计误差和数值离散误差。更小步长只直接处理第三部分。

## 自检与阅读连接

`[DERIVATION]` 对标准高斯，score 为 $-x$。稳态 VP 中 $f=-\beta x/2$，PF 速度为 $-\beta x/2+\beta x/2=0$。零速度可以保持高斯密度，而对应稳态 SDE 中每个粒子仍在运动。这是“相同边缘不等于相同路径”的最短检查。

**score**：log-density 的空间梯度。**denoising**：从带噪观测估计干净信号。**posterior mean**：给定观测后的条件均值。**marginal**：把未观测变量积分掉。

`[DIRECT · 来源范围]` 定义和时间方向与课程16–17合并讲义核对；上述数值例子和组织方式为独立推导。相关原始研究为 [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) 和 [DDPM](https://arxiv.org/abs/2006.11239)。本篇不提供逐页原文或完整翻译，也不声称包含所有附录证明。

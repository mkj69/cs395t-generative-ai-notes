---
title: 'Lecture 16: Diffusion Preparation: Stochastic Calculus'
slug: lecture-16
date: '2026-10-19'
updated: '2026-10-31'
status: checked
type: learning-note
course: CS 395T
source: https://noahgol.github.io/teaching/cs395t-f26/
summary: 从随机步长、Itô 修正和概率守恒推导 SDE 的密度演化，用布朗运动与 OU 算例核对时间尺度。
public: true
language: zh-CN
tags:
- Diffusion
- Brownian motion
- Ito calculus
- Fokker-Planck
related_research_notes: []
authorship: ai-assisted
---

> **AI 辅助学习笔记。** 本文是独立讲解与推导，不是讲义全文译文，也不代表作者已经完成个人复习。讲义来源：[课程原始 PDF](https://noahgol.github.io/teaching/cs395t-f26/lecture16-17-v2.pdf)。第 16、17 课共用一份讲义。公式已与相应来源核对，未做人工终审。

`[DIRECT]` 标明来源直接支持的简述；`[DERIVATION]` 标明数学推导；`[SYNTHESIS]` 标明概念连接与解释。自行构造的数值例子不代表论文实验。

[返回 Diffusion 单元导读](diffusion-unit.html)

## 阅读起点：一条随机路径和一群粒子的分布

`[SYNTHESIS]` 本篇从一个问题出发：当每一步都有随机扰动时，怎样区分“一个样本如何运动”和“所有样本的分布如何变化”？学习目标是能亲手写出一步更新，并检查它产生的均值、方差和概率流。以下解释和算例是独立的学习辅助，不是讲义逐句译文。

先固定符号。$X_t\in\mathbb R^d$ 是随机位置，$p_t(x)$ 是在时间 $t$ 的密度，$f(t,x)$ 是漂移，$g(t)$ 是与位置无关的标量噪声系数。$B_t$ 是标准布朗运动。本文只讨论这种各向同性、状态无关的扩散系数；矩阵系数或依赖 $x$ 的噪声需要改写公式。

## 为什么随机项乘的是平方根步长

`[DERIVATION]` 把长度为 $T$ 的时间分成 $n=T/h$ 段。若一步噪声为 $\sqrt h Z_k$，其中 $Z_k$ 独立且服从标准高斯，那么总增量的方差为

$$\operatorname{Var}\left(\sum_{k=1}^{T/h}\sqrt h Z_k\right)=\frac{T}{h}h=T.$$

若误写为 $hZ_k$，总方差就变成 $Th$，在步长趋于零时消失。布朗运动的增量应满足 $B_{t+h}-B_t\sim\mathcal N(0,hI)$，相邻不重叠时间区间的增量独立。

`[DERIVATION · 算例]` 取 $T=1,h=0.01$。100 次、每次标准差为 $0.1$ 的独立增量，最后方差为1；每次标准差若误用0.01，最后方差只有0.01。更多小步不应让固定时长内的总随机性凭空消失。

## SDE 是局部更新规则，不是普通可导路径

$$dX_t=f(t,X_t)dt+g(t)dB_t.$$

`[DERIVATION]` Euler–Maruyama 一步为

$$X_{t+h}\approx X_t+h f(t,X_t)+g(t)\sqrt h Z.$$

条件均值增量是 $hf$，条件协方差增量是 $hg^2I$。布朗路径几乎必然处处不可导，不能把 $dB_t/dt$ 当作普通光滑函数直接操作。SDE 的精确含义来自积分式及随机积分；数值更新是近似，不是它的定义所允许的任意替代品。

对于确定性的 $g$，随机积分的协方差为 $\int_0^t g(s)^2ds\,I$，不是 $(\int_0^t g(s)ds)^2I$。后者错误地把不同时刻的独立增量当成完全相关。

## Itô 公式中多出来的一项从哪里来

`[DERIVATION]` 用一维 $F(x)=x^2$ 做检查。普通链式法则会漏掉平方噪声的贡献。因为一步的 $(\Delta B)^2$ 在累积意义下贡献 $dt$，所以

$$d(X_t^2)=2X_t f(t,X_t)dt+2X_tg(t)dB_t+g(t)^2dt.$$

纯布朗运动 $X_t=B_t$ 时，取期望得到 $\frac{d}{dt}\mathbb E[B_t^2]=1$，于是 $\mathbb E[B_t^2]=t$。如果没有最后一项，就会错误得到方差不增长。

一般的光滑 $F(t,x)$ 满足

$$dF=\left(\partial_tF+f\cdot\nabla F+\frac{g^2}{2}\Delta F\right)dt+g\nabla F\cdot dB_t.$$

此处的 $\nabla$ 与 $\Delta$ 对空间求导。$\Delta F$ 是各坐标二阶导数之和。应用公式和交换期望、积分需要相应可积性；随机积分均值为零也需要条件，不能对任意爆炸过程直接套用。

## 从测试函数推到整个密度

`[DERIVATION]` 取光滑且紧支撑的测试函数 $\phi$。它可以看成在空间不同位置设置的测量权重。Itô 公式给出

$$\frac d{dt}\mathbb E[\phi(X_t)]=\int\left(f\cdot\nabla\phi+\frac{g^2}{2}\Delta\phi\right)p_t\,dx.$$

将空间导数通过分部积分移到密度上，得到

$$\partial_t p_t=-\nabla\cdot(fp_t)+\frac{g^2}{2}\Delta p_t.$$

这就是 Fokker–Planck 方程。它不是在预测一个粒子的轨迹，而是在预测粒子群的概率密度。需要检查边界项消失；在有边界的区域内，边界条件属于模型的一部分。

定义概率流 $J=fp_t-\frac{g^2}{2}\nabla p_t$，同一个方程可写成 $\partial_t p_t=-\nabla\cdot J$。积分后，若边界没有净流出，就有 $\frac d{dt}\int p_tdx=0$。这是概率质量守恒的检查。

## 一个可手算的 OU 检查

`[DERIVATION]` 考虑 $dX_t=-\kappa X_tdt+\sigma dB_t$，其中 $\kappa>0$。漂移把状态拉向0，噪声不断扩散。若初始均值为 $m_0$、方差为 $v_0$，由一阶与二阶矩方程可得

$$m_t=e^{-\kappa t}m_0,\qquad v_t=e^{-2\kappa t}v_0+\frac{\sigma^2}{2\kappa}(1-e^{-2\kappa t}).$$

取 $\kappa=1,\sigma=\sqrt2,v_0=4$，有 $v_t=1+3e^{-2t}$，最终趋于1。只有初始分布本来是相应高斯等特殊情形，才能直接把整条边缘分布都说成高斯；任意初始数据通常仍是平滑后的混合分布。

## 常见误解与自检

| 误解 | 检查方法 |
| --- | --- |
| 噪声增量的尺度是 $h$ | 计算固定时长内总方差是否消失 |
| 单个路径的随机性等于密度不守恒 | 对 Fokker–Planck 方程积分 |
| Brownian 增量是一条普通函数曲线 | 检查 $d(B_t^2)$ 是否遗漏 $dt$ |
| 一时刻分布相同意味着随机过程相同 | 比较两个时刻的联合分布或协方差 |

`[SYNTHESIS]` 接下来学习逆扩散时，最有用的两样工具是概率流与密度梯度：它们会解释为什么逆向运动中必须加入 score。

## 词汇与来源

**drift**：漂移，确定性的局部变化趋势。**diffusion coefficient**：扩散系数，决定随机增量大小。**marginal density**：某个时刻的边缘密度。**generator**：生成算子，描述函数期望的瞬时变化，不是生成图片的神经网络。

`[DIRECT · 来源范围]` 课程16–17合并讲义的随机微积分部分及相关附录用于核对定义与记号。完整讲义还覆盖后续扩散内容，本篇不声称等同于整份讲义。关于数学严谨条件可继续阅读 [Tang and Zhao 的技术教程](https://arxiv.org/abs/2402.07487)。

---
title: 'Lecture 19: Diffusion III: Conditional Generation and Classifier-Free Guidance'
slug: lecture-19
date: '2026-10-28'
updated: '2026-10-31'
status: draft
type: learning-note
course: CS 395T
source: https://noahgol.github.io/teaching/cs395t-f26/
summary: 从贝叶斯公式推导条件 score，解释 CFG 的两种强度记法、训练中的条件丢弃，以及引导强度放大误差的原因。
public: true
language: zh-CN
tags:
- Diffusion
- conditional generation
- classifier-free guidance
related_research_notes: []
authorship: ai-assisted
---

> **AI 辅助主题学习笔记，待课堂资料补充。** 截至 2026-10-31，课程表只有本课主题，没有公开课件。本篇依据公开原始论文与独立推导编写，不能视为课堂记录；所列论文是补充参考，不声称是教师指定阅读。页面日期为课程安排日期。

`[DIRECT]` 标明来源直接支持的简述；`[DERIVATION]` 标明数学推导；`[SYNTHESIS]` 标明概念连接与解释。自行构造的数值例子不代表论文实验。

[返回 Diffusion 单元导读](diffusion-unit.html)

## 从“生成一张图”到“按条件生成”

`[SYNTHESIS]` 无条件模型希望生成符合总体数据规律的样本；条件模型还接收条件 $c$，例如类别或文本。输入同一个噪声种子并改变条件，应当改变样本落入的语义区域。条件不是训练标签旁边的备注，它改变了目标分布 $p_t(x\mid c)$。

这一讲的关键不是记住一个相减公式，而是理解：为什么两个去噪预测之差含有条件信息，为什么放大这个差会改变生成行为，以及这种放大为什么有代价。

## 贝叶斯公式给出条件方向

`[DERIVATION]` 在密度为正、可微的区域，由

$$p_t(x\mid c)=\frac{p_t(c\mid x)p_t(x)}{p(c)}$$

对 $x$ 取对数再求梯度。$p(c)$ 不依赖 $x$，所以

$$s_c(x,t)=s_u(x,t)+\nabla_x\log p_t(c\mid x),$$

其中 $s_c=\nabla_x\log p_t(x\mid c)$ 是条件 score，$s_u=\nabla_x\log p_t(x)$ 是无条件 score。因此

$$s_c-s_u=\nabla_x\log p_t(c\mid x).$$

右边回答的不是“哪里更像所有真实图片”，而是“当前位置朝哪个方向变化，会让条件 $c$ 更可信”。分类器引导需要估计右边；CFG 则用左边的两个生成模型预测来近似。

这里的分类器必须理解噪声层级 $t$。只会识别干净图片的分类器，其梯度未必适合高度加噪的 $x_t$。贝叶斯等式是精确数学关系，但两个神经网络预测之差仍有估计误差。

## CFG 的两个公式其实差一个1

`[DIRECT]` Ho 与 Salimans 的方法联合训练有条件和无条件的去噪预测，并在采样时线性组合。原论文的强度记法是 $(1+w)\epsilon_c-w\epsilon_u$。来源：[Classifier-Free Diffusion Guidance，v1，§3.2](https://arxiv.org/html/2207.12598v1)。

`[DERIVATION]` 本笔记采用更方便讨论基线的 $\gamma$：

$$\widehat s_\gamma=\widehat s_u+\gamma(\widehat s_c-\widehat s_u).$$

| 数值 | 含义 |
| --- | --- |
| $\gamma=0$ | 无条件预测 |
| $\gamma=1$ | 普通条件预测 |
| $\gamma>1$ | 超出普通条件预测的外推 |

两种记法关系是 $\gamma=1+w$。看到代码中“guidance scale=0”时，必须先看公式；有时它表示无条件，有时表示不加额外引导。不能只根据数值判断。

设无条件预测为2、条件预测为5，这只是一个人为构造的一维输出。$\gamma=1$ 得5，$\gamma=2$ 得8。CFG 把条件差从3放大为6，不是在两个输出之间取概率平均。$\gamma>1$ 时，无条件项的系数 $1-\gamma$ 为负。

## 为什么噪声预测可以使用同一个组合

`[DERIVATION]` 对高斯扰动，条件和无条件分支都满足 $s=-\epsilon/\sigma_t$，且共享同一个 $\sigma_t$。因此

$$\widehat\epsilon_\gamma=\widehat\epsilon_u+\gamma(\widehat\epsilon_c-\widehat\epsilon_u)$$

映射回 score 后，恰好得到上式。负号不会改变引导强度，因为它同时作用于两项。

如果换成其他预测参数化，需要先核对两个分支是否通过同一个仿射变换转换到速度或 score。不能混合一个分支的噪声预测和另一个分支的干净样本预测。采样器也必须知道网络实际输出什么。

## 训练为何要随机丢弃条件

`[SYNTHESIS]` 设训练样本为 $(x_0,c)$，随机取 $t,\epsilon$，构造 $x_t=\alpha_tx_0+\sigma_t\epsilon$。再独立抛一枚概率为 $p_{\mathrm{drop}}$ 的硬币，将部分条件替换成空条件 $\varnothing$。网络学习

$$\mathbb E\|\epsilon_\theta(x_t,t,\widetilde c)-\epsilon\|^2.$$

同一网络因而见过有条件和空条件两类输入。采样时，在同一个 $x_t,t$ 上调用两次或合并成一个批次，然后组合输出。“classifier-free”省掉的是额外的分类器，不代表省掉条件，也不代表每一步一定只做一次样本量意义上的网络计算。

空条件是模型训练时规定的表示，不能随意假设全零向量就等价。负面提示词也不严格等同于无条件分布；把一个指定负面条件放进基线分支，会改变这套解释的对象。

## 引导对应什么分布：一个容易过度推论的地方

`[DERIVATION]` 若两个 score 都精确，固定一个时间 $t$，且归一化常数有限，则

$$s_\gamma=\nabla_x\log\left[p_t(x\mid c)^\gamma p_t(x)^{1-\gamma}\right].$$

这说明瞬时向量场偏向什么区域。但逐时定义这些密度，不保证它们共同构成原加噪过程的有效边缘分布族。因此，仅靠这个等式，不能断言整个反向采样器的终点密度就严格等于 $p_0(x\mid c)^\gamma p_0(x)^{1-\gamma}$。这里还涉及时间一致性和采样动力学。

一个可计算的诊断例子：只考虑某一固定时刻的两个高斯密度，$p_u=\mathcal N(0,4)$、$p_c=\mathcal N(1,1)$。它们的 score 分别是 $-x/4$ 与 $1-x$。于是

$$s_\gamma(x)=\gamma-\frac{1+3\gamma}{4}x.$$

当 $\gamma\ge0$ 时，这个瞬时场对应均值 $4\gamma/(1+3\gamma)$、方差 $4/(1+3\gamma)$ 的高斯。$\gamma=1$ 时均值1、方差1；$\gamma=3$ 时均值1.2、方差0.4。这个例子只演示组合的数学作用，不是某数据集实验，也不是对完整生成过程的精确预测。

## 为什么越强不一定越好

`[DERIVATION]` 记条件、无条件估计误差为 $e_c,e_u$。引导后的误差为

$$e_\gamma=\gamma e_c+(1-\gamma)e_u.$$

所以 $\gamma\ge1$ 时，$\|e_\gamma\|\le\gamma\|e_c\|+(\gamma-1)\|e_u\|$。这只是上界，不说明每个样本误差必然单调增大，但说明大引导会放大两个分支的不一致。

`[SYNTHESIS]` 评价时应同时看条件符合度、样本多样性、明显伪影和计算预算。仅展示最符合提示词的一张图，不能说明整体分布变得更好。固定种子与求解器可以帮助比较引导参数，但仍需多个种子估计稳定性。

## 自测与词义

- 如果条件分支和无条件分支完全相同，引导强度再大也不改变结果。
- 如果代码采用 $(1+w)\epsilon_c-w\epsilon_u$，普通条件采样对应 $w=0$。
- 概率密度可以随 $t$ 变化，条件 $c$ 本身并不因此成为被加噪的随机状态。

**conditioning**：将指定信息作为生成条件。**unconditional**：未附加该条件。**guidance**：在采样时改变预测方向的引导。**extrapolation**：超出两点之间范围的外推。**fidelity**：样本与目标特征的符合程度。**mode coverage**：覆盖不同数据模式的程度。

下一篇 [Rectified Flows](lecture-20.html) 继续讨论速度场，但问题转向路径、耦合和数值步数。CFG 与路径设计是不同维度的选择，可以同时存在。

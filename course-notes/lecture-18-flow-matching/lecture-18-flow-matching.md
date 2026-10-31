---
title: 'Lecture 18: Diffusion II: Flow Matching and Consistency Models'
slug: lecture-18
date: '2026-10-26'
updated: '2026-10-31'
status: checked
type: learning-note
course: CS 395T
source: https://noahgol.github.io/teaching/cs395t-f26/
summary: 用条件期望证明 FM 与 CFM 的关系，推导高斯路径速度，并区分局部速度、生成轨迹和 consistency 的终点映射。
public: true
language: zh-CN
tags:
- Diffusion
- Flow matching
- Conditional expectation
- Consistency models
related_research_notes: []
authorship: ai-assisted
---

> **AI 辅助学习笔记。** 本文是独立讲解与推导，不是讲义全文译文，也不代表作者已经完成个人复习。讲义来源：[课程原始 PDF](https://noahgol.github.io/teaching/cs395t-f26/lecture18-v2.pdf)。第 16、17 课共用一份讲义。公式已与相应来源核对，未做人工终审。

`[DIRECT]` 标明来源直接支持的简述；`[DERIVATION]` 标明数学推导；`[SYNTHESIS]` 标明概念连接与解释。自行构造的数值例子不代表论文实验。

[返回 Diffusion 单元导读](diffusion-unit.html)

## 阅读问题：学习局部速度，还是直接学习终点

`[SYNTHESIS]` 本篇把两个学习任务放在同一张逻辑图中：Flow Matching 学“当前位置下一刻如何移动”；Consistency Model 学“沿这条轨迹最终到哪里”。二者都涉及 ODE，却预测不同对象。这里的讲解使用独立算例和推导，不是课程讲义逐句翻译。

本篇流匹配部分使用课程方向：$t=0$ 是噪声，$t=1$ 是数据。记 $Z\sim\mathcal N(0,I)$，$X_1\sim P_{data}$。SD3 论文的主要记号恰好相反，做比较时先转换时间。

## 一个连续生成器的两个视角

`[DERIVATION]` 定义 $\dot Y_t=v_\theta(t,Y_t)$。给定初始噪声，ODE 轨迹确定；在连续、空间 Lipschitz 等适当条件下，解唯一。流映射把初始分布推送成 $p_t$。

若密度随速度 $u$ 运输，则 $\partial_tp_t=-\nabla\cdot(p_tu_t)$。沿轨迹用链式法则：

$$\frac d{dt}\log p_t(Y_t)=\partial_t\log p_t(Y_t)+\nabla\log p_t(Y_t)\cdot u_t(Y_t)=-\nabla\cdot u_t(Y_t).$$

最大似然路线需要积分轨迹和散度。流匹配选择另一种监督：直接回归能实现目标密度演化的速度。这里“无需模拟”专指构造训练对，不表示采样不用 ODE，也不表示没有梯度计算。

## 为何不能直接平均所有条件速度

选定条件密度 $r_t(x\mid x_1)$，边缘密度是 $p_t(x)=\int r_t(x\mid x_1)P_{data}(dx_1)$。每个条件路径有自己的速度 $u_t(x\mid x_1)$。以下的 $W_t$ 表示先抽取 $X_1$、再按 $r_t(\cdot\mid X_1)$ 抽取的状态。

`[DERIVATION]` 对条件连续性方程积分：

$$\partial_tp_t=-\nabla\cdot\int r_t(x\mid x_1)u_t(x\mid x_1)P_{data}(dx_1).$$

因此可以选

$$u_t(x)=\frac{\int r_t(x\mid x_1)u_t(x\mid x_1)P_{data}(dx_1)}{p_t(x)}=\mathbb E[u_t(W_t\mid X_1)\mid W_t=x,t].$$

这是后验平均。只有条件路径在当前位置有概率质量，它的速度才应该参与这里的运输。简单写成 $\mathbb E_{X_1}u_t(x\mid X_1)$ 通常错误。

`[DERIVATION · 算例]` 假设当前位置来自两个条件分支的后验概率为0.8和0.2，对应速度为2和−3，则整体速度是 $0.8(2)+0.2(-3)=1$。等权平均得到−0.5，连方向都不同。这个离散分支例子只解释加权逻辑，不是课程实验。

## 平方误差为什么解决难算的后验平均

记随机标签为 $U$，$m=\mathbb E[U\mid t,W_t]$。对任何只看到 $t,W_t$ 的网络，

$$\mathbb E\|v_\theta-U\|^2=\mathbb E\|v_\theta-m\|^2+\mathbb E\|U-m\|^2.$$

`[DERIVATION]` 展开平方后交叉项是 $\mathbb E[(v_\theta-m)\cdot\mathbb E[U-m\mid t,W_t]]=0$。所以训练条件标签的 CFM 与训练边缘速度的 FM 相差一个与参数无关的常数。

这个等价性在使用相同时间采样分布时适用于同一网络函数类。它不等于每个小批量梯度完全一样。额外改变时间权重，则可能改变有限容量共享模型的折中。

## 高斯路径速度的三步推导

给定数据点 $x_1$，设 $\phi_t(z\mid x_1)=\mu_t(x_1)+\sigma_t(x_1)z$，$z$ 为起点噪声。

`[DERIVATION]` 先固定 $z$ 求时间导数，再反解当前状态，最后代回：

$$\dot\phi_t=\dot\mu_t+\dot\sigma_tz,\qquad z=\frac{x-\mu_t}{\sigma_t},\qquad u_t(x\mid x_1)=\dot\mu_t+\frac{\dot\sigma_t}{\sigma_t}(x-\mu_t).$$

需要 $\sigma_t>0$ 才能这样求逆。若 $\sigma_1=0$，端点可奇异，但使用已采样的 $Z$ 构造标签 $\dot\mu_t+\dot\sigma_tZ$ 不必除以零。数值采样和端点理论仍需分别处理。

密度路径通常不唯一决定速度；若 $\nabla\cdot(p_tw_t)=0$，则 $u_t+w_t$ 有相同密度演化。固定完整流映射后，才有更强的速度确定性。

## 直线标签与一维高斯反例

线性选择 $\mu_t=tX_1,\sigma_t=1-t$ 给出

$$W_t=(1-t)Z+tX_1,\qquad U=X_1-Z.$$

训练一次只需抽三个独立变量 $t,Z,X_1$，计算 $\|v_\theta(t,W_t)-(X_1-Z)\|^2$。它不能保证学到的每条 ODE 轨迹严格为直线。

`[DERIVATION · 反例]` 取独立 $Z,X_1\sim\mathcal N(0,1)$。中间方差 $D(t)=(1-t)^2+t^2$，由联合高斯条件均值得到

$$u_t(x)=\frac{2t-1}{D(t)}x.$$

解为 $Y_t=\sqrt{D(t)}Y_0$。轨迹先向中心收缩再展开，虽然两端分布相同而训练插值是直线。取 $Y_0=2$，中点为 $\sqrt2$，终点又回到2。路径选择和耦合决定了中间过程，端点相同不代表路径无意义。

## Consistency：同一轨迹上的点共享终点

`[DERIVATION]` 若 $F(t,s,x)$ 表示从时间 $s$ 的位置 $x$ 出发，走到时间 $t$，则以终点1为例，$C(s,x)=F(1,s,x)$。对同一条轨迹，$C(t,Y_t)=Y_1$。边界条件是 $C(1,x)=x$。当文献改用前向加噪时间时，边界通常写在0或一个小正数，不能直接混用。

训练可以让同一轨迹上相邻点的终点预测一致。蒸馏时，教师提供一个小步近似；常见实现还使用停止梯度和目标网络。课程讲义给出的是简化一致性目标，不能把所有工程细节都归为讲义公式本身。

只有一致性而没有边界或其他约束时，常数函数也能使两边相同。边界条件不是可有可无的装饰。一步生成的质量还取决于教师误差、蒸馏误差和训练覆盖，不能由名称推出完美结果。

## 实践检查与连接

1. 明确时间从噪声到数据，还是从数据到噪声。
2. 每次训练固定抽到的两个端点，对时间求导时不重新抽样。
3. 分清条件速度、边缘速度与网络估计。
4. 比较计算成本时记录网络调用次数，不能只记录求解器步数。
5. 把当前时刻预测误差与最终生成误差分开评价。

**flow map**：将起点映到当前时刻的变换。**marginalization**：对未观测变量积分。**tractable**：实际可计算。**endpoint map**：把当前状态映到轨迹终点。

`[DIRECT · 来源范围]` 课程18讲义用于核对流匹配的时间约定与定义；相关原始论文是 [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) 与 [Consistency Models](https://arxiv.org/abs/2303.01469)。与 [SD3 全文中英对照阅读页](https://mkj69.github.io/writing/papers/sd3/annotated/) 对照时，令其时间为 $1-t$，速度符号随之反转。

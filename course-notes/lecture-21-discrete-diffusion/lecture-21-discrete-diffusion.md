---
title: 'Lecture 21: Diffusion V: Discrete Diffusion Models'
slug: lecture-21
date: '2026-11-04'
updated: '2026-10-31'
status: draft
type: learning-note
course: CS 395T
source: https://noahgol.github.io/teaching/cs395t-f26/
summary: 从有限状态转移矩阵出发，手算反向后验，再连接吸收态掩码、训练损失和离散文本采样的误差来源。
public: true
language: zh-CN
tags:
- Diffusion
- discrete diffusion
- D3PM
- masked language modeling
related_research_notes: []
authorship: ai-assisted
---

> **AI 辅助主题学习笔记，待课堂资料补充。** 截至 2026-10-31，课程表只有本课主题，没有公开课件。本篇依据公开原始论文与独立推导编写，不能视为课堂记录；所列论文是补充参考，不声称是教师指定阅读。页面日期为课程安排日期。

`[DIRECT]` 标明来源直接支持的简述；`[DERIVATION]` 标明数学推导；`[SYNTHESIS]` 标明概念连接与解释。自行构造的数值例子不代表论文实验。

[返回 Diffusion 单元导读](diffusion-unit.html)

## Token 为什么不能直接套连续 score

`[SYNTHESIS]` 连续图像状态可以加一个很小的实数扰动。离散 token 则只有若干类别，token 编号之间的减法并不天然代表语义距离。离散扩散保留“逐步破坏、学习反向恢复”的设计，但要把高斯扰动换成类别间的随机转移。

离散状态也可以采用连续时间，这两者不是反义词。本篇先用离散时间步 $t=0,1,\ldots,T$，最后再说明连续时间速率矩阵。$t=0$ 始终表示干净数据。

## 用转移矩阵写前向过程

`[DERIVATION]` 令 $Q_t[i,j]=q(X_t=j\mid X_{t-1}=i)$。行是出发状态，列是到达状态，因此每行元素非负且和为1。用行向量表示概率，有

$$\overline Q_t=Q_1Q_2\cdots Q_t,\qquad q(X_t=j\mid X_0=i)=\overline Q_t[i,j].$$

矩阵乘法把所有可能中间路径的概率加起来。只要 $\overline Q_t$ 可高效计算，就可以训练时直接从任意 $t$ 的条件分布抽样，不必逐步执行前向链。一般稠密大词表矩阵会很贵，结构化转移能改变计算成本。

`[DIRECT]` D3PM 将前向离散扰动写成转移矩阵，并研究均匀、局部与吸收态等结构；训练结合变分目标与辅助去噪交叉熵。来源：[Austin 等，v1，§3](https://arxiv.org/html/2107.03006v1)。下面用自行构造的小矩阵展开其概率含义，不复现论文实验。

## 一个可以手算的两状态例子

`[DERIVATION]` 状态为 A、B，令

$$Q=\begin{pmatrix}0.8&0.2\\0.3&0.7\end{pmatrix}.$$

从 A 出发，两步仍为 A 的概率为 $0.8\times0.8+0.2\times0.3=0.70$，两步变成 B 的概率为 $0.30$。第一步走向哪一个状态不再需要显式保存，因为矩阵乘法已将它求和。

现在知道 $X_0=A$ 且 $X_2=B$，问第一步是什么。两条可能路径的未归一化概率为

$$P(A\to A\to B)=0.8\times0.2=0.16,$$

$$P(A\to B\to B)=0.2\times0.7=0.14.$$

所以 $q(X_1=A\mid X_2=B,X_0=A)=8/15$，另一个为 $7/15$。注意不是把 $Q$ 转置就得到反向概率。逆向推断必须结合前一时刻的概率质量。

## 反向后验公式来自贝叶斯，而不是矩阵求逆

`[DERIVATION]` 若分母为正，

$$q(X_{t-1}=k\mid X_t=j,X_0=i)=\frac{\overline Q_{t-1}[i,k]Q_t[k,j]}{\overline Q_t[i,j]}.$$

分子是“先到 $k$，再从 $k$ 到 $j$”；分母是所有 $k$ 的总和。训练时 $X_0$ 已知，因此这个后验可以计算。生成时 $X_0$ 未知，必须由模型提供反向概率或预测干净类别的分布。

精确关系是

$$q(X_{t-1}=k\mid X_t=j)=\sum_i q(X_{t-1}=k\mid X_t=j,X_0=i)q(X_0=i\mid X_t=j).$$

这个混合恒等式说明干净类别预测为何有用。具体模型可能使用不同的参数化或归一化方式，不能把上式自动当作所有 D3PM 实现逐字采用的代码。

对整段文本，前向噪声可以逐位置独立施加，但一个位置的反向预测通常依赖整段加噪文本。前向独立并不推出语言中的 token 独立。

## 吸收态：一旦被遮住就留在 MASK

`[DERIVATION]` 加一个特殊状态 M。对普通 token $a$，每步以概率 $\beta_t$ 变为 M，否则保留 $a$；从 M 出发永远停在 M。定义累计保留概率

$$\overline\alpha_t=\prod_{r=1}^t(1-\beta_r).$$

若原始 $X_0=a$，则 $q(X_t=a\mid a)=\overline\alpha_t$，$q(X_t=M\mid a)=1-\overline\alpha_t$。从前向链看，token 只会被遮盖，不会直接改成另一个词。

对 $s<t$，已知 $X_t=M$ 与 $X_0=a$ 时，

$$q(X_s=a\mid X_t=M,X_0=a)=\frac{\overline\alpha_s-\overline\alpha_t}{1-\overline\alpha_t},$$

$$q(X_s=M\mid X_t=M,X_0=a)=\frac{1-\overline\alpha_s}{1-\overline\alpha_t}.$$

例如 $\overline\alpha_s=0.8$、$\overline\alpha_t=0.5$，反向一步揭开概率是0.6，继续遮住概率0.4。若 $X_t$ 已经是普通 token，在这条特定前向过程的精确反向链中，它的更早状态也必须是同一个 token。

这解释了纯吸收态反向采样中的“揭开后保留”。允许重新遮盖、替换或编辑已生成 token 的其他采样器，改变了这个基本过程，不能混作同一条精确反向链。

## 训练到底优化什么

`[DERIVATION]` 对离散时间潜变量链，负对数似然的变分上界包含三类项：

$$\mathcal L=\mathbb E_{X_0}\mathrm{KL}(q(X_T\mid X_0)\|p(X_T))$$

$$+\sum_{t=2}^T\mathbb E_{X_0,X_t}\mathrm{KL}(q(X_{t-1}\mid X_t,X_0)\|p_\theta(X_{t-1}\mid X_t))$$

$$-\mathbb E_{X_0,X_1}\log p_\theta(X_0\mid X_1).$$

第一项衡量终端加噪分布与采样先验的差；中间各项让模型接近可计算的后验；最后一项负责从最靠近数据的噪声层恢复数据。不同参数化和时间采样会改变实际估计器与权重。仅说“做交叉熵”遗漏了训练信号的来源和权重。

`[DIRECT]` MDLM 在吸收态掩码模型中通过结构化输出约束简化目标，并给出相应采样方法。来源：[Sahoo 等，v1，§3–4](https://arxiv.org/html/2406.07524v1)。

`[DERIVATION]` 在上述单 token 吸收态例子里，已被揭开的状态没有类别不确定性；对于 M，反向步的类别损失可写成揭开概率乘以预测原始类别的负对数概率：

$$\frac{\overline\alpha_s-\overline\alpha_t}{1-\overline\alpha_t}\left[-\log r_\theta(a\mid X_t)\right].$$

这是给定当前被遮盖时的贡献。对当前状态继续取期望，还要考虑当前被遮盖的概率。它说明“多种遮盖比例的语言建模损失”如何出现，也说明均匀随机遮盖加任意交叉熵不必然等于同一个似然界。

## 从全 MASK 到一段文本并不等于一次独立填空

`[SYNTHESIS]` 如果终端加噪分布设为完全遮盖，反向生成的初始就是整段 MASK，网络先根据当前上下文预测多个位置的类别分布，再依据采样规则揭开部分位置，随后重新计算。后续迭代能利用此前揭开的内容。一个批次并行预测多个位置可以减少串行轮次，但并不自动保证这些位置的联合关系正确。

设一个人为数据集只含两种等概率序列：AA 和 BB。全 MASK 时每个位置的正确边缘分布都是 A、B 各一半。如果独立同时抽两个位置，AA、AB、BA、BB 各出现四分之一，其中一半不在真实支持集内。先揭开一个位置再条件预测另一个，可以保持一致性。这个例子说明边缘预测正确仍不足以保证联合采样正确，也解释大反向步可能损失依赖关系。

若每轮处理长度 $L$ 的序列，共做 $K$ 轮，不能仅以 $K<L$ 就宣称比自回归更快。每轮计算量、缓存、批大小和硬件都影响实际时间。质量比较还应匹配 token 数量与计算预算。

## 连续时间不等于连续状态

`[DERIVATION]` 连续时间有限状态过程使用速率矩阵 $R(t)$，非对角元素非负，每行和为0。小时间步满足

$$P(X_{t+h}=j\mid X_t=i)=\delta_{ij}+hR_{ij}(t)+o(h).$$

状态仍然是有限类别。若 $p_t(j)>0$，反向时间中的从 $j$ 到 $i$ 的瞬时速率含有 $p_t(i)R_{ij}(t)/p_t(j)$。这是概率流入与当前质量的比值，和连续扩散依赖密度信息的思想相通，但不是对 token 编号求普通导数。

## 自测与词义

- 用给定矩阵从 A 出发两步后到 B，概率是0.30。
- 将转移矩阵转置，通常不能得到反向链。
- MASK 的前向吸收性决定了特定反向链的保留规则。
- 正确单位置边缘分布，不能替代正确整段联合分布。

**categorical**：有限类别分布。**transition matrix**：状态转移概率矩阵。**absorbing state**：进入后无法离开的状态。**posterior**：在观测到结果后更新的条件分布。**masking**：将可见 token 替换成遮盖状态。**rate matrix**：单位时间跳转强度矩阵。**ancestral sampling**：按模型分解顺序逐级抽样。

本篇完成 Diffusion 单元的概念闭环：连续与离散模型都通过可控的前向破坏构造训练信号，再学习恢复过程；它们使用的状态空间、概率对象和采样误差不同。

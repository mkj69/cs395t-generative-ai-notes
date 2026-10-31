---
title: 'Diffusion Unit: Reading Guide and Source Map'
slug: diffusion-unit
date: '2026-10-31'
updated: '2026-10-31'
status: checked
type: learning-note
course: CS 395T
source: https://noahgol.github.io/teaching/cs395t-f26/
summary: 六次课的中文推导笔记、英文术语与原始资料入口，覆盖随机微积分准备至离散扩散。另列逐句译文的现有覆盖与资料缺口。
public: true
language: zh-CN
authorship: ai-assisted
tags:
  - Diffusion
  - Reading guide
  - Source map
---

> **AI 辅助学习资料。** 这里收录六篇独立讲解笔记，包含推导、可手算的例子、常见误解与英文词义。资料状态核对于 2026-10-31。它们不代表个人已经完成的阅读记录，也不是讲义全文翻译。

## 六次课，从随机过程到离散生成

`[DIRECT · COURSE SCHEDULE]` 范围依据[课程公开日程](https://noahgol.github.io/teaching/cs395t-f26/)。日期是课程安排日期；11月的条目目前是主题学习准备。第16与17课链接同一份16页讲义，第18课另有6页讲义；后面三课尚无公开课件。

| 课次与日期 | 中文讲解笔记 | 这篇解决的问题 | 原始资料 |
| --- | --- | --- | --- |
| 16 · 10月19日 | [Stochastic Calculus](lecture-16.html) | 随机步长为何是平方根？Itô 修正如何产生密度方程？ | [16–17合并讲义](https://noahgol.github.io/teaching/cs395t-f26/lecture16-17-v2.pdf) |
| 17 · 10月21日 | [Score Matching](lecture-17.html) | 不知道数据密度，如何训练 score？为什么 ODE 系数减半？ | [同一份原始 PDF](https://noahgol.github.io/teaching/cs395t-f26/lecture16-17-v2.pdf) |
| 18 · 10月26日 | [Flow Matching and Consistency](lecture-18.html) | 条件速度如何变成边缘速度？预测速度与预测终点有何不同？ | [Lecture 18 PDF](https://noahgol.github.io/teaching/cs395t-f26/lecture18-v2.pdf) |
| 19 · 10月28日 | [Conditional Generation and CFG](lecture-19.html) | 两个预测之差为何提供条件方向？强度参数为何差一个1？ | 无公开课件，笔记内列补充论文 |
| 20 · 11月2日 | [Rectified Flows](lecture-20.html) | 插值直线是否意味着生成轨迹直？Reflow 改变了什么？ | 无公开课件，可连接已有 SD3 精读 |
| 21 · 11月4日 | [Discrete Diffusion](lecture-21.html) | 如何用矩阵加噪、手算反向后验并从 MASK 恢复文本？ | 无公开课件，笔记内列补充论文 |

## 建议阅读顺序与时间记号

`[SYNTHESIS]` 16课提供随机过程与概率守恒工具，17课用它解释 score 和逆过程，18课换成速度回归，19课加入生成条件，20课研究耦合与少步生成，21课将恢复过程搬到离散状态空间。

| 阅读对象 | 时间0 | 时间终点 | 常见预测对象 |
| --- | --- | --- | --- |
| 本站16–17课 | 数据 | 噪声 | score 或噪声 |
| 本站18课流匹配部分、20课 | 噪声 | 数据 | 速度 |
| SD3 原文主记号 | 数据 | 噪声 | 对原文时间求导的速度 |
| 本站21课前向链 | 干净 token | 被扰动 token | 反向类别概率 |

先统一时间方向，再比较公式。令新时间为 $t=1-s$ 时，轨迹导数也要变号。名字相似的参数不保证定义相同。

## 原文中英逐句阅读的现有覆盖

- **已存在：** [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis 精读页](https://mkj69.github.io/writing/papers/sd3/annotated/)。包含英文原文、中文译文和语境词义标注；它是相关论文，不是第20课课件。
- **课程16–17及18讲义：** 本站目前提供原始 PDF 链接与独立讲解，尚未提供全文中英逐句译本。要制作该版本，需要直接提供用于翻译的 PDF，或确认允许全文改编的授权。
- **课程19–21课件：** 当前公开页面没有提供原始 PDF。补充论文不能冒充缺失的课程课件。

词义说明放在每篇笔记末尾，并在推导中结合语境解释。比如 score 在这里是对数密度的空间梯度，coupling 是端点的联合分布，marginal 是对部分变量积分后的分布。它们不能只按日常英文直译。

## 怎样区分内容的依据

`[DIRECT]` 用于明确来源支持的简述。`[DERIVATION]` 用于展示推导、条件与可核算的例子。`[SYNTHESIS]` 用于组织概念关系。前3篇已进行对应讲义的来源核对，后3篇保留 draft 状态以说明尚缺课堂材料；全部仍需个人阅读与人工复核。

这里没有复刻课堂幻灯片，没有把补充论文列成未经确认的指定阅读，也没有将示意算例写成实验结果。

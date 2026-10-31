---
title: 'Diffusion Unit: Reading Guide and Source Map'
slug: diffusion-unit
date: '2026-10-31'
updated: '2026-10-31'
status: checked
type: learning-note
course: CS 395T
source: https://noahgol.github.io/teaching/cs395t-f26/
summary: Six English study notes connect stochastic calculus to discrete diffusion.
  A separate source-reading section tracks original PDFs and bilingual annotated editions.
public: true
language: en
authorship: ai-assisted
tags:
- Diffusion
- Reading guide
- Source map
---


> **AI-assisted study guide.** The six English notes below contain independent explanations, derivations, worked examples, and terminology. Source availability was checked on October 31, 2026. These pages do not record completed personal reading passes. Bilingual editions of source PDFs are separate reading materials.

## English study notes

`[DIRECT · COURSE SCHEDULE]` This unit follows the [public course schedule](https://noahgol.github.io/teaching/cs395t-f26/). Dates refer to scheduled classes. The November entries are topic preparation. Lectures 16 and 17 share one 16-page handout. Lecture 18 has a separate six-page handout. The remaining three sessions have no public course handout at the time of checking.

| Class and date | English study note | Central question | Course source |
| --- | --- | --- | --- |
| 16 · October 19 | [Stochastic Calculus](lecture-16.html) | Why does noise scale with the square root of time? How does the Itô correction determine density evolution? | [Combined Lecture 16 and 17 PDF](https://noahgol.github.io/teaching/cs395t-f26/lecture16-17-v2.pdf) |
| 17 · October 21 | [Score Matching](lecture-17.html) | How can a score be learned without the density? Why does the ODE coefficient contain one half? | [Same course PDF](https://noahgol.github.io/teaching/cs395t-f26/lecture16-17-v2.pdf) |
| 18 · October 26 | [Flow Matching and Consistency](lecture-18.html) | How does conditional velocity become marginal velocity? How does a velocity differ from an endpoint prediction? | [Lecture 18 PDF](https://noahgol.github.io/teaching/cs395t-f26/lecture18-v2.pdf) |
| 19 · October 28 | [Conditional Generation and CFG](lecture-19.html) | Why does the difference between predictions encode a conditional direction? | No public handout; supplementary papers are identified in the note. |
| 20 · November 2 | [Rectified Flows](lecture-20.html) | Does straight interpolation imply straight generation? What does reflow change? | No public handout; the related SD3 source edition is listed below. |
| 21 · November 4 | [Discrete Diffusion](lecture-21.html) | How do transition matrices, reverse posteriors, and masked recovery fit together? | No public handout; supplementary papers are identified in the note. |

## Conceptual sequence and time conventions

`[SYNTHESIS]` Lecture 16 develops stochastic motion and probability conservation. Lecture 17 uses those tools to explain scores and reversal. Lecture 18 introduces velocity regression. Lecture 19 adds conditions to generation. Lecture 20 studies couplings and coarse integration. Lecture 21 moves the recovery process to a discrete state space.

| Reading | Time zero | Terminal time | Typical prediction |
| --- | --- | --- | --- |
| Notes 16 and 17 | Data | Noise | Score or noise |
| Flow matching in note 18; note 20 | Noise | Data | Velocity |
| Main SD3 convention | Data | Noise | Velocity with respect to the paper's time coordinate |
| Forward chain in note 21 | Clean tokens | Corrupted tokens | Reverse category probabilities |

Changing coordinates to $t=1-s$ changes the sign of a trajectory derivative. Similar parameter names do not guarantee identical definitions.

## Bilingual source readings

This is a separate reading format: an original English source passage, its Chinese translation, and contextual vocabulary annotations. The English study notes above retain their own explanatory structure.

| Source | Original | Separate bilingual edition |
| --- | --- | --- |
| Lecture 16 and 17 handout | [Original PDF](https://noahgol.github.io/teaching/cs395t-f26/lecture16-17-v2.pdf) | Pending. No complete bilingual edition is published. |
| Lecture 18 handout | [Original PDF](https://noahgol.github.io/teaching/cs395t-f26/lecture18-v2.pdf) | Pending. No complete bilingual edition is published. |
| Lecture 19, 20, and 21 course materials | No public handouts currently linked by the course | Pending source availability. |
| Scaling Rectified Flow Transformers for High-Resolution Image Synthesis | Related paper, not a course handout | [Open the English and Chinese annotated edition](https://mkj69.github.io/writing/papers/sd3/annotated/) |

Complete bilingual course-handout editions require directly supplied PDFs or a verified license permitting full adaptation. The existing SD3 edition is linked as related reading. It does not stand in for an unavailable Lecture 20 handout.

## Evidence and review status

`[DIRECT]` identifies a statement supported by a cited source. `[DERIVATION]` exposes calculations, assumptions, and worked examples. `[SYNTHESIS]` connects concepts. The first three notes have been checked against relevant handout definitions and formulas. The final three retain draft status because classroom materials are missing. All six remain open to personal study and human review.

Supplementary papers are not presented as confirmed assigned readings. Constructed examples are not reported experimental results. The English notes and bilingual source editions serve different reading purposes and remain separately identified.

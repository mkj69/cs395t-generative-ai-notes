- ---
  title: "Lecture 10 — GPUs, FlashAttention, and Inference"
  slug: "lecture-10"
  date: "2026-09-30"
  updated: "2026-09-30"
  status: "draft"
  type: "learning-note"
  course: "CS 395T"
  source: "https://noahgol.github.io/teaching/cs395t-f26/lecture10.pdf"
  summary: "An AI-assisted first reconstruction of GPU memory hierarchy, roofline analysis, FlashAttention's online softmax and IO complexity, KV caching, and the different bottlenecks of prefill and generation."
  public: true
  tags:
    - GPU systems
    - roofline model
    - FlashAttention
    - KV cache
    - inference
    - arithmetic intensity
  related_research_notes: []
  authorship: "ai-assisted"
  ---
- > **AI-assisted first draft.** This note reconstructs the supplied Lecture 10 PDF and expands its equations into step-by-step derivations and small checks. It has not yet been manually verified or rewritten by the author. Statements tagged `[DIRECT]` follow the lecture; `[DERIVATION]` and `[SYNTHESIS]` make the reasoning explicit.
- ## Lecture context
- `[DIRECT]` Source: <https://noahgol.github.io/teaching/cs395t-f26/lecture10.pdf>
- `[DIRECT]` The supplied PDF is eight pages long. Its substantive sections cover GPU background, FlashAttention, and inference tricks.
- `[SOURCE BOUNDARY]` The opening sentence says that the lecture will also analyze self-improvement through sharpening, and the reference list includes a sharpening paper. However, this PDF contains no sharpening section or corresponding derivation. This draft therefore does not invent one.
- `[SYNTHESIS]` The lecture is organized around one systems question: when a Transformer operation is slow, is the machine spending its time doing arithmetic, or moving data? FlashAttention and KV caching are useful because they change data movement without changing the mathematical model being evaluated.
- ## Reading map

  | Part | Central question | Main idea |
  | --- | --- | --- |
  | GPU hierarchy | Where do computation and data live? | Fast arithmetic is useful only when data reaches it quickly enough. |
  | Roofline model | Is an operation compute-bound or memory-bound? | Compare arithmetic intensity with the hardware balance. |
  | Naive attention | Why can exact attention waste memory traffic? | The score and probability matrices are written to and reread from HBM. |
  | FlashAttention | How can exact attention avoid those transfers? | Tile the computation, maintain online softmax statistics, and discard score tiles. |
  | KV caching | What can causal decoding reuse? | Earlier keys and values do not change when a new token is appended. |
  | Inference arithmetic | Why do prefill and generation behave differently? | Prompt tokens share weights; generated sequences do not share their KV caches. |
- ## 1. GPU execution and memory
- ### 1.1 Streaming multiprocessors and arithmetic units
- `[DIRECT]` An NVIDIA GPU contains many streaming multiprocessors (SMs). The lecture uses the H100 SXM as its running example and reports 132 SMs.
- `[DIRECT]` Each H100 SM is divided into four subpartitions. A subpartition contains a warp scheduler, registers, CUDA cores, and tensor-core capacity.
- `[DIRECT]` A warp is a group of 32 threads. The scheduler issues a common instruction to the threads, which apply that instruction to different data.
- `[EXPLANATION]` Parallelism is therefore hierarchical. Many SMs work at once; several warps may reside on one SM; and the threads of a warp perform the same kind of operation on different elements.
- `[DIRECT]` CUDA cores handle ordinary scalar or vector-like work, including pointwise operations and reductions. Tensor cores perform small matrix multiply-accumulate operations and supply most of the throughput used by large Transformer matrix multiplications.
- `[SYNTHESIS]` Calling a GPU “fast” is incomplete. Tensor cores can execute matrix arithmetic extremely quickly, but a kernel benefits only if its work can be expressed in suitable matrix tiles and if those tiles are delivered without starving the arithmetic units.
- ### 1.2 The memory hierarchy
- `[DIRECT]` The hierarchy in the lecture runs from registers, to an SM-local shared-memory/L1 pool, to chip-wide L2 cache, and finally to high-bandwidth memory (HBM).
- `[DIRECT]` Registers are thread-local working storage. The H100 figure used by the lecture assigns 256 KiB of registers to an SM in total. A kernel that requires many registers per thread can reduce occupancy because fewer threads fit simultaneously.
- `[DIRECT]` Each H100 SM also has a 256 KiB pool divided between programmer-managed shared memory (SMEM) and hardware-managed L1 cache. FlashAttention uses this kind of on-chip storage for reusable tiles.
- `[DIRECT]` L2 is shared across SMs and is about 50 MB in the stated H100 configuration. HBM is much larger: the H100 SXM example has 80 GB and approximately $3.35$ TB/s of peak bandwidth.
- `[EXPLANATION]` Registers and on-chip caches use SRAM, which is fast but area-expensive. HBM uses DRAM, which stores far more data but is farther from the arithmetic units. Capacity increases as we move outward; latency and movement cost also increase.
- `[SYNTHESIS]` The key distinction is not simply “cache versus memory.” It is whether an intermediate value can stay on-chip long enough to be reused. If it is written to HBM and later read back, the same mathematical computation may take much longer even when the FLOP count is unchanged.
- ### 1.3 Arithmetic intensity and the roofline model
- `[DIRECT]` Let $F$ be an operation's FLOP count and let $D$ be the number of bytes transferred between HBM and the chip. Its arithmetic intensity is

  $$
  I=\frac{F}{D}\qquad\text{FLOPs per byte}.
  $$
- `[DIRECT]` If the usable compute rate is $C$ FLOPs/s and HBM bandwidth is $\beta$ bytes/s, define the hardware balance

  $$
  I^*=\frac{C}{\beta}.
  $$
- `[DERIVATION · Step 1]` Arithmetic alone would require

  $$
  t_{\mathrm{comp}}=\frac{F}{C}.
  $$
- `[DERIVATION · Step 2]` Moving the required bytes alone would require

  $$
  t_{\mathrm{mem}}=\frac{D}{\beta}.
  $$
- `[DIRECT]` The idealized roofline model assumes that computation and transfer overlap, giving

  $$
  t\approx\max\left\{t_{\mathrm{comp}},t_{\mathrm{mem}}\right\}.
  $$
- `[DERIVATION · Step 3]` Compare the two times:

  $$
  \frac{t_{\mathrm{comp}}}{t_{\mathrm{mem}}}
  =\frac{F/C}{D/\beta}
  =\frac{F/D}{C/\beta}
  =\frac{I}{I^*}.
  $$
- `[CONCLUSION]` If $I>I^*$, compute time is larger and the idealized operation is compute-bound. If $I<I^*$, transfer time is larger and the operation is memory-bound.
- #### H100 balance check
- `[DIRECT]` The lecture uses roughly $9.9\times10^{14}$ dense FP16/BF16 Tensor Core FLOPs/s and $3.35\times10^{12}$ HBM bytes/s.
- `[DERIVATION]` Dividing gives

  $$
  I^*\approx
  \frac{9.9\times10^{14}}{3.35\times10^{12}}
  \approx 295.5\ \text{FLOPs/byte}.
  $$
- `[EXPLANATION]` An operation must perform roughly 300 useful FLOPs per byte fetched from HBM to keep peak matrix arithmetic busier than peak memory bandwidth in this simplified comparison.
- #### Small numerical example
- `[EXAMPLE]` Suppose a kernel performs $10^{12}$ FLOPs and transfers $10^{10}$ bytes. Then $I=100$ FLOPs/byte, below the stated H100 balance.
- `[DERIVATION]` At the cited peaks,

  $$
  t_{\mathrm{comp}}\approx\frac{10^{12}}{9.9\times10^{14}}
  \approx1.01\ \text{ms},
  $$

  while

  $$
  t_{\mathrm{mem}}\approx\frac{10^{10}}{3.35\times10^{12}}
  \approx2.99\ \text{ms}.
  $$
- `[CONCLUSION]` The roofline estimate predicts a memory bottleneck and an ideal time near $2.99$ ms, not the sum of the two times.
- `[LIMIT]` Peak specifications do not guarantee application throughput. Kernel launches, imperfect occupancy, synchronization, unsupported data shapes, cache behavior, and incomplete overlap can all increase runtime. The roofline maximum is an optimistic lower-bound model, not a timing oracle.
- ## 2. Where naive attention moves data
- ### 2.1 Column-oriented attention
- `[DIRECT]` For one head, let the sequence length be $N$ and let query, key, and value width all equal $d$. The lecture stores tokens as columns:

  $$
  Q=[q_1,\ldots,q_N],\quad
  K=[k_1,\ldots,k_N],\quad
  V=[v_1,\ldots,v_N]\in\mathbb{R}^{d\times N}.
  $$
- `[DIRECT]` Define the score matrix, column-wise attention matrix, and output by

  $$
  S=\frac{K^\top Q}{\sqrt d},
  \qquad
  A_{ji}=\frac{e^{S_{ji}}}{\sum_{u=1}^N e^{S_{ui}}},
  \qquad
  Y=VA.
  $$
- `[EXPLANATION]` Column $i$ of $S$ contains the scores produced by query $q_i$ against every key. Column $i$ of $A$ is therefore a probability vector over key positions, and column $i$ of $Y$ is the weighted sum of the values for query $i$.
- `[DIRECT]` For causal attention, entries with $j>i$ are masked by setting their scores to $-\infty$. This changes which entries contribute but not the systems argument about materializing a large score matrix.
- ### 2.2 FLOPs are not the whole cost
- `[DERIVATION]` Multiplying $K^\top\in\mathbb{R}^{N\times d}$ by $Q\in\mathbb{R}^{d\times N}$ costs approximately $2N^2d$ FLOPs when a multiply and an addition count separately.
- `[DERIVATION]` Multiplying $V\in\mathbb{R}^{d\times N}$ by $A\in\mathbb{R}^{N\times N}$ costs another approximately $2N^2d$ FLOPs. The two large matrix products therefore use about

  $$
  4N^2d
  $$

  FLOPs, or $\Theta(N^2d)$ asymptotically.

- `[DIRECT]` A naive multi-kernel implementation writes $S$ to HBM, rereads it for softmax, writes $A$, and rereads $A$ for the value product. Both $S$ and $A$ have $N^2$ entries.
- `[SYNTHESIS]` The undesirable term is not a new mathematical object required by attention's definition. It is an implementation boundary: one kernel produces a full intermediate matrix in HBM so another kernel can consume it.
- `[LIMIT]` “Fuse the kernels” is not by itself a complete solution. The entire $N\times N$ matrix usually cannot fit in on-chip SRAM. The fused computation must be reorganized so only small pieces are live at once.
- ## 3. FlashAttention
- ### 3.1 Tiling and recomputation
- `[DIRECT]` FlashAttention computes exact attention while avoiding a persistent $N\times N$ attention matrix in HBM.
- `[DIRECT]` Tiling loads blocks of queries, keys, and values into SRAM. It computes one score tile, uses the tile to update a small running state, and discards the tile before loading the next one.
- `[DIRECT]` During training, recomputation keeps the output and softmax normalization statistics but recreates score and probability tiles during the backward pass. Reading a large stored intermediate from HBM can cost more time than recomputing it on-chip.
- `[SYNTHESIS]` FlashAttention does not approximate or sparsify dense attention. Its output is the same mathematical $V\operatorname{softmax}(K^\top Q/\sqrt d)$; the algorithm changes the order and location of the work.
- ### 3.2 The online softmax invariant
- `[DIRECT]` Fix query $i$ and suppose a set $J$ of keys has already been processed. Maintain

  $$
  m_i=\max_{j\in J}S_{ji},
  $$

  $$
  \ell_i=\sum_{j\in J}e^{S_{ji}-m_i},
  $$

  and

  $$
  y_i=
  \frac{\sum_{j\in J}e^{S_{ji}-m_i}v_j}{\ell_i}.
  $$
- `[EXPLANATION]` $m_i$ makes the exponentials numerically stable, $\ell_i$ is the shifted softmax denominator, and $y_i$ is already the normalized output over the processed keys.
- `[DERIVATION · Step 1]` The unnormalized old numerator can be recovered from the stored state:

  $$
  u_i^{\mathrm{old}}=\ell_i y_i
  =\sum_{j\in J}e^{S_{ji}-m_i}v_j.
  $$
- `[DIRECT]` For a new block $\widetilde J$, compute its local maximum $\widetilde m_i$, shifted sum $\widetilde\ell_i$, and shifted unnormalized value sum

  $$
  \widetilde u_i
  =\sum_{j\in\widetilde J}e^{S_{ji}-\widetilde m_i}v_j.
  $$
- `[DERIVATION · Step 2]` The combined maximum must be

  $$
  m_i'=\max\{m_i,\widetilde m_i\}.
  $$
- `[DERIVATION · Step 3]` Rewrite each old exponential relative to the new maximum:

  $$
  e^{S_{ji}-m_i'}
  =e^{m_i-m_i'}e^{S_{ji}-m_i}.
  $$
- `[DERIVATION · Step 4]` The new block obeys the analogous identity

  $$
  e^{S_{ji}-m_i'}
  =e^{\widetilde m_i-m_i'}e^{S_{ji}-\widetilde m_i}.
  $$
- `[DERIVATION · Step 5]` Summing the two rescaled blocks yields the updated denominator

  $$
  \ell_i'
  =e^{m_i-m_i'}\ell_i
  +e^{\widetilde m_i-m_i'}\widetilde\ell_i.
  $$
- `[DERIVATION · Step 6]` The corresponding normalized output is

  $$
  y_i'
  =\frac{
  e^{m_i-m_i'}\ell_i y_i
  +e^{\widetilde m_i-m_i'}\widetilde u_i
  }{\ell_i'}.
  $$
- `[CONCLUSION]` The updated triple $(m_i',\ell_i',y_i')$ has exactly the same form as the original invariant, now for $J\cup\widetilde J$. No earlier score has to remain in memory.
- `[DIRECT]` Initialization uses $m_i=-\infty$, $\ell_i=0$, and $y_i=0$. The old contribution in the first update vanishes.
- #### Two-block sanity check
- `[EXAMPLE]` Suppose a scalar-valued query sees scores $[0,1]$ with values $[2,4]$ in the first block and score $2$ with value $10$ in the second block.
- `[DERIVATION]` The first block has $m=1$, $\ell=e^{-1}+1$, and

  $$
  y=\frac{2e^{-1}+4}{e^{-1}+1}.
  $$
- `[DERIVATION]` The second block has $\widetilde m=2$, $\widetilde\ell=1$, and $\widetilde u=10$. The merged maximum is $m'=2$, so

  $$
  \ell'=e^{-1}(e^{-1}+1)+1=e^{-2}+e^{-1}+1.
  $$
- `[DERIVATION]` The merged numerator becomes

  $$
  e^{-1}(2e^{-1}+4)+10
  =2e^{-2}+4e^{-1}+10.
  $$
- `[CHECK]` This is precisely what ordinary stable softmax produces when all three scores are shifted by the global maximum $2$. The block boundary disappears from the final answer.
- ### 3.3 Tiles and loop order
- `[DIRECT]` Measure SRAM capacity as $M$ scalar words. Split query columns into blocks of at most $B_q$ tokens and key/value columns into blocks of at most $B_k$ tokens, using

  $$
  B_k=\left\lceil\frac{M}{4d}\right\rceil,
  \qquad
  B_q=\min\{B_k,d\}.
  $$
- `[DIRECT]` The numbers of blocks are

  $$
  T_q=\left\lceil\frac{N}{B_q}\right\rceil,
  \qquad
  T_k=\left\lceil\frac{N}{B_k}\right\rceil.
  $$
- `[DERIVATION]` A live key/value pair uses $O(dB_k)$ words. A query/output pair uses $O(dB_q)$ words. The score tile uses $O(B_kB_q)$ words.
- `[DERIVATION]` Because $B_q\le d$, the score tile satisfies $B_kB_q\le dB_k$. Because $B_q\le B_k$, the query/output storage is no larger in order than the key/value storage. With $B_k=\Theta(M/d)$, all live tiles fit in $O(M)$ words.
- `[DIRECT]` The outer loop fixes one key/value block in SRAM. The inner loop sweeps over every query block, loading its current output and normalization state, computing a score tile, merging it, and writing the updated running state back to HBM.
- `[EXPLANATION]` Reusing a resident key/value tile across many query blocks is the source of the HBM savings. The conceptual score matrix is still partitioned into tiles, but no complete score or probability matrix is stored in HBM.
- ### 3.4 Correctness and storage
- `[DIRECT]` The lecture states that the tiled forward pass returns exact dense attention, uses $O(N^2d)$ FLOPs, and requires only $O(N)$ additional HBM storage beyond its inputs and output, plus the SRAM workspace.
- `[DERIVATION · Correctness]` The online update preserves the invariant after each processed key block. Induction over all $T_k$ blocks therefore leaves each query with the normalized weighted sum over every key, which is exactly its column of $V\operatorname{softmax}(K^\top Q/\sqrt d)$.
- `[DERIVATION · FLOPs]` Every query-key pair participates once in a length-$d$ dot product and once in a length-$d$ weighted value contribution. There are $N^2$ pairs, so the total remains $O(N^2d)$.
- `[DERIVATION · Storage]` The persistent normalization state contains one maximum and one denominator per query, or $O(N)$ scalars. Score tiles are temporary SRAM values, so they do not contribute $O(N^2)$ HBM storage.
- `[SYNTHESIS]` FlashAttention's main asymptotic improvement is in memory traffic and intermediate storage, not arithmetic count.
- ### 3.5 HBM transfer complexity
- `[DIRECT]` Under the lecture's word-access model, naive attention uses

  $$
  \Theta(Nd+N^2)
  $$

  HBM accesses because it moves the $Nd$-sized inputs/output and materializes $N^2$-sized score and probability arrays.

- `[DIRECT]` When $d\le M\le Nd$, the FlashAttention schedule uses

  $$
  \Theta\!\left(\frac{N^2d^2}{M}\right)
  $$

  HBM word accesses.

- `[DERIVATION · Step 1]` Since $B_k=\Theta(M/d)$, the number of key/value blocks is

  $$
  T_k=\Theta\!\left(\frac{N}{B_k}\right)
  =\Theta\!\left(\frac{Nd}{M}\right).
  $$
- `[DERIVATION · Step 2]` For each fixed key/value block, sweeping all query blocks reads and updates $Q$, $Y$, $m$, and $\ell$, costing $\Theta(Nd)$ accesses in the stated model.
- `[DERIVATION · Step 3]` Multiplying cost per sweep by the number of sweeps gives

  $$
  \Theta(NdT_k)
  =\Theta\!\left(\frac{N^2d^2}{M}\right).
  $$
- `[DERIVATION · Step 4]` Loading the base inputs and writing the output adds $\Theta(Nd)$. The assumption $M\le Nd$ ensures the tiled-sweep term is at least the same order, so

  $$
  \Theta\!\left(Nd+\frac{N^2d^2}{M}\right)
  =\Theta\!\left(\frac{N^2d^2}{M}\right).
  $$
- `[INTUITION]` More SRAM permits a larger key/value tile. A larger tile means fewer complete sweeps over the queries and running outputs, so HBM traffic decreases inversely with $M$ in this model.
- `[LIMIT]` These are asymptotic word counts for a particular dense-attention schedule. Actual performance also depends on causal masking, head dimensions, data types, kernel occupancy, cache effects, and hardware-specific tiling.
- ## 4. Autoregressive inference and KV caching
- ### 4.1 Prefill and generation
- `[DIRECT]` Inference includes two phases.
- **Prefill:** process all known prompt tokens in parallel under the causal mask. At every layer, save their key and value vectors. The logits at the final prompt position define the distribution for the first generated token.
- **Generation:** sample one token, process that new token using the existing cache, append its new key/value vectors, and use the new logits to predict the next token.
- `[DERIVATION]` In a causal Transformer, the representation used to compute an earlier token's key and value depends only on that token and its prefix. Appending a later token does not alter that prefix. Earlier keys and values can therefore be reused exactly.
- `[SYNTHESIS]` The KV cache removes repeated projection work for old tokens, but it creates a growing memory object that must be read during every decoding step.
- ### 4.2 Shape notation for the cost model
- `[DIRECT]` Let
- $B$ be the number of sequences in a batch;
- $T$ be the number of query tokens processed per sequence in one forward call;
- $S$ be the number of key/value positions available to each sequence.
- `[DIRECT]` For ordinary prefill, $S=T$ equals the prompt length. For ordinary generation, $T=1$ and $S$ is the current context length including the new token.
- `[DIRECT]` The lecture's byte estimates assume 16-bit values, so every scalar moved to or from HBM contributes two bytes. Non-matrix-multiplication arithmetic is omitted.
- ## 5. Parameter matrix multiplications
- ### 5.1 FLOPs, bytes, and intensity
- `[DIRECT]` Let $W\in\mathbb{R}^{F\times D}$ map model width $D$ to width $F$. Combine the $BT$ token activations into $X\in\mathbb{R}^{D\times BT}$.
- `[DERIVATION]` Each of the $BTF$ output entries is a dot product of length $D$. Counting multiplication and addition separately gives

  $$
  F_{\mathrm{lin}}=2BTDF.
  $$
- `[DIRECT]` Under the optimistic assumption that $X$ and $W$ are each read once and the output is written once, the bytes transferred are

  $$
  D_{\mathrm{lin}}
  =2(BTD+DF+BTF).
  $$
- `[DERIVATION]` The corresponding times and arithmetic intensity are

  $$
  t_{\mathrm{comp}}=\frac{2BTDF}{C},
  \qquad
  t_{\mathrm{mem}}=\frac{2(BTD+DF+BTF)}{\beta},
  $$

  $$
  I_{\mathrm{lin}}
  =\frac{BTDF}{BTD+DF+BTF}.
  $$
- ### 5.2 Why token count amortizes weights
- `[DIRECT]` When $BT\ll D,F$, the $DF$ parameter matrix dominates the byte count. Then

  $$
  I_{\mathrm{lin}}\approx BT.
  $$
- `[EXPLANATION]` The weights are shared across all $BT$ tokens in the call. Loading one weight supports roughly $BT$ uses, so increasing either prompt length $T$ or batch size $B$ amortizes the same parameter read.
- `[DIRECT]` The idealized crossover occurs around

  $$
  BT\gtrsim I^*.
  $$

  With the cited H100 peak balance, this means a few hundred tokens per call.

- #### Concrete dimension check
- `[EXAMPLE]` Take $D=4096$ and $F=11008$.
- `[DERIVATION]` With one token, $BT=1$ and

  $$
  I_{\mathrm{lin}}
  =\frac{4096\cdot11008}{4096+4096\cdot11008+11008}
  \approx1.
  $$
- `[DERIVATION]` With $BT=1024$ tokens,

  $$
  I_{\mathrm{lin}}
  =\frac{1024\cdot4096\cdot11008}
  {1024\cdot4096+4096\cdot11008+1024\cdot11008}
  \approx762.
  $$
- `[SYNTHESIS]` The same matrix multiplication can move from strongly memory-bound during single-token decoding to potentially compute-bound during a long prefill, even though the weights and layer definition are unchanged.
- `[LIMIT]` This calculation assumes ideal reuse and a single read of the weights. Small or awkward shapes, quantization kernels, parallel sharding, cache behavior, and non-matmul operations can move the real crossover.
- ## 6. Attention during inference
- ### 6.1 Ideal fused-attention cost
- `[DIRECT]` For one head of width $d$ in each of $B$ sequences, let

  $$
  Q\in\mathbb{R}^{d\times T},
  \qquad
  K,V\in\mathbb{R}^{d\times S}.
  $$
- `[DERIVATION]` Computing $K^\top Q$ costs approximately $2BSTd$ FLOPs. Multiplying the values by the attention weights costs another $2BSTd$, so

  $$
  F_{\mathrm{attn}}=4BSTd.
  $$
- `[DERIVATION]` In an ideal fused implementation, reading $Q$ and writing $Y$ each move $2BTd$ bytes, while reading both $K$ and $V$ moves $4BSd$ bytes. Hence

  $$
  D_{\mathrm{attn}}=4Bd(S+T).
  $$
- `[DERIVATION]` Dividing gives

  $$
  I_{\mathrm{attn}}
  =\frac{F_{\mathrm{attn}}}{D_{\mathrm{attn}}}
  =\frac{ST}{S+T}.
  $$
- `[EXPLANATION]` Both $B$ and $d$ cancel. In this idealized model, increasing the number of independent sequences adds attention work and KV-cache bytes at the same rate.
- ### 6.2 Prefill
- `[DIRECT]` For prefill, $S=T$, so

  $$
  I_{\mathrm{attn}}=\frac{T}{2}.
  $$
- `[EXPLANATION]` Every key and value can be reused by many query positions. A longer prompt therefore increases potential arithmetic per byte.
- `[DERIVATION]` Comparing with $I^*\approx295$ predicts a compute crossover near

  $$
  T\gtrsim2I^*\approx590,
  $$

  which the lecture rounds to roughly 600 tokens.

- `[EXAMPLE]` At $T=2048$, the rectangular idealization gives $I_{\mathrm{attn}}=1024$ FLOPs/byte, above the cited balance.
- `[LIMIT]` Causal masking removes approximately half of the query-key pairs, finite SRAM can force repeated loads, and real kernels have efficiency losses. The numerical threshold is therefore only a rough guide.
- ### 6.3 Generation
- `[DIRECT]` For ordinary generation, $T=1$ and $S\gg1$, giving

  $$
  I_{\mathrm{attn}}
  =\frac{S}{S+1}
  \approx1\ \text{FLOP/byte}.
  $$
- `[EXPLANATION]` The new query attends over a long, sequence-specific KV cache but supplies only one query position. There is almost no cross-query reuse.
- `[EXAMPLE]` At context length $S=8192$,

  $$
  I_{\mathrm{attn}}=\frac{8192}{8193}\approx0.9999.
  $$
- `[CONCLUSION]` Ordinary decoding attention is strongly memory-bound in the lecture's model.
- `[DIRECT]` Increasing batch size does not improve this attention ratio. Each added sequence brings its own KV cache, so FLOPs and transferred bytes both scale with $B$.
- `[SYNTHESIS]` This differs from parameter matrices. Model weights are shared across the whole batch, so batching amortizes weight reads. KV-cache entries belong to individual sequences, so batching supplies no analogous reuse for ordinary attention.
- ### 6.4 SRAM limits the ideal reuse
- `[DIRECT]` The one-read byte count above assumes enough SRAM to exploit all desired reuse. For dense square attention, the FlashAttention transfer theorem instead gives $\Theta(N^2d^2/M)$ word transfers.
- `[DERIVATION]` Dividing $\Theta(N^2d)$ FLOPs by those transfers gives arithmetic intensity

  $$
  \Theta\!\left(\frac{M}{d}\right)
  $$

  at fixed bytes per word.

- `[SYNTHESIS]` Prompt length creates an opportunity for reuse, but SRAM capacity sets how much reuse one tiling schedule can realize before data must be fetched again.
- ## 7. The central comparison

  | Operation | Prefill behavior | Generation behavior | Why |
  | --- | --- | --- | --- |
  | Parameter matmul | Often high intensity for long prompts | Low intensity at small batch | All tokens share the same weights. |
  | Attention | Reuses keys/values across many queries | Reads a long KV cache for one query | KV state is sequence-specific. |
  | Softmax / pointwise work | Commonly memory-sensitive | Commonly memory-sensitive | Few FLOPs are performed per moved element. |

- `[SYNTHESIS]` “Inference is memory-bound” is too coarse. During prefill, large matrix multiplications and sufficiently long attention can be compute-bound. During generation, weight reads at small batch and KV-cache reads at every batch size are the dominant memory-pressure mechanisms in this simplified model.
- `[SYNTHESIS]` FlashAttention and KV caching solve different reuse problems. FlashAttention keeps temporary score work on-chip; KV caching keeps permanent prefix keys and values across decoding steps. The first avoids materializing an intermediate, while the second avoids recomputing an invariant prefix.
- ## 8. Manual verification checklist
- [ ] Re-derive the online softmax merge without looking at the formula.
- [ ] Explain why $\ell_i y_i$, rather than $y_i$ alone, is needed when two blocks are merged.
- [ ] Draw the key-block outer loop and query-block inner loop, then identify every HBM read and write.
- [ ] Re-derive $T_k=\Theta(Nd/M)$ from the SRAM tile size.
- [ ] Explain why FlashAttention changes IO complexity but not the $O(N^2d)$ dense-attention FLOP count.
- [ ] Derive $I_{\mathrm{lin}}\approx BT$ and state exactly which denominator terms were neglected.
- [ ] Derive $I_{\mathrm{attn}}=ST/(S+T)$ and explain why batch size cancels.
- [ ] Explain, without equations, why batching helps weight reads but not the per-sequence KV-cache ratio.
- ## 9. Questions to carry into study
- `[QUESTION]` How do grouped-query attention and multi-query attention change KV-cache bytes and generation intensity?
- `[QUESTION]` How do quantized weights move the parameter-matmul crossover, especially when dequantization adds arithmetic?
- `[QUESTION]` Which assumptions in the $\Theta(N^2d^2/M)$ transfer result change for causal or sliding-window attention?
- `[QUESTION]` At what sequence lengths do kernel launch overhead and non-matmul operations dominate the simple roofline estimate?
- `[QUESTION]` How should the roofline model be extended when tensor-core compute, CUDA-core reductions, and communication overlap imperfectly?
- `[QUESTION]` Was the sharpening section intentionally omitted from the supplied Lecture 10 PDF, or will it appear in a later version or lecture?
- ## References recorded from the supplied lecture
- `[SOURCE NOTE]` These references are named by the lecture PDF. This first draft uses the lecture's presentation and has not independently checked every source.
- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.” *NeurIPS*, 2022. <https://arxiv.org/abs/2205.14135>
- Jacob Austin, Swapnil Patil, Adam Paszke, and Reiner Pope. “How to Think About GPUs.” *How To Scale Your Model*, 2025. <https://jax-ml.github.io/scaling-book/gpus/>
- *How To Scale Your Model*. “All About Rooflines.” <https://jax-ml.github.io/scaling-book/roofline/>
- *How To Scale Your Model*. “All About Transformer Inference.” <https://jax-ml.github.io/scaling-book/inference/>
- NVIDIA. “NVIDIA H100 GPU: Product Specifications (H100 SXM).” <https://www.nvidia.com/en-us/data-center/h100/>
- Michael Andersch et al. “NVIDIA Hopper Architecture In-Depth.” NVIDIA Technical Blog, 2022. <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>
- `[SOURCE BOUNDARY]` The lecture also lists Huang et al., “Self-Improvement in Language Models: The Sharpening Mechanism,” but the supplied PDF does not contain a substantive sharpening section. It is not used to fill that missing material here.

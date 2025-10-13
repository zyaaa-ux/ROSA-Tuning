# ROSA-Tuning: Make window-attention greater than global-attention

## TL;DR

ROSA-Tuning adds minimal per-layer discrete adapter parameters (W_lm^(ℓ), E^(ℓ)) while keeping the pretrained model frozen, enabling windowed attention models to achieve near-global ultra-long-range memory capabilities.

Practical results: After replacing global attention with windowed attention in Qwen3-0.6B, adding ROSA branch and freezing the original model, training for one epoch on 28,000 samples achieves better PPL on PG-19-16k than the original global attention model.

Resource advantages: ROSA core is parameter-free and runs on CPU, while GPU only handles small matrix projection and representation injection; windowed attention can process arbitrary-length contexts (limited by CPU-side index memory), significantly reducing the quadratic complexity cost of global attention.

---

## Experimental Results

### Setup

- Base model: Qwen3-0.6B with global attention or windowed attention
- Training: 28,000 samples, 1 epoch, original model frozen, only ROSA adapters trained
- Evaluation: PG-19 dataset, sequence length 16k, window size 1024

### Perplexity (PPL) Comparison On Validation Datasets

- Global Attention: 31.96
- Windowed Attention: 465.59
- Windowed Attention + ROSA: 25.96

ROSA enables windowed attention to outperform the global attention baseline.

---

## Method Overview

### Core Formula

For the ℓ-th layer hidden representation $h^{(\ell)} \in \mathbb{R}^{T\times d}$:

$$
h^{(\ell+1)} = h^{(\ell)}
+ \mathrm{Attn}^{(\ell)}_{\text{win}}\big(\mathrm{LN}(h^{(\ell)})\big)
+ v^{(\ell)}
+ \mathrm{MLP}^{(\ell)}\big(\mathrm{LN}(\cdot)\big)
$$

The ROSA branch $v^{(\ell)}$ is computed as:

$$
\begin{aligned}
\textbf{logits}^{(\ell)} &= W_{\rm lm}^{(\ell)}\,\mathrm{LN}(h^{(\ell)}) \in \mathbb{R}^{T\times K_\ell}, \\
z^{(\ell)} &= \arg\max(\textbf{logits}^{(\ell)}) \in \{0,\ldots,K_\ell-1\}^T,\\
y^{(\ell)} &= \mathrm{ROSA}_{\text{collapse}}\big(z^{(\ell)}\big) \in \{-1,0,\ldots,K_\ell-1\}^T,\\
p^{(\ell)} &= \mathrm{softmax}\big(\textbf{logits}^{(\ell)}/\tau_\ell\big),\\
v^{(\ell)}_{\text{hard}} &= \mathrm{Emb}^{(\ell)}\big[y^{(\ell)}+1\big],\quad
v^{(\ell)}_{\text{soft}} = p^{(\ell)}\,\mathrm{Emb}^{(\ell)}[1:],\\
v^{(\ell)} &= v^{(\ell)}_{\text{hard}} + \mathrm{sg}\big(v^{(\ell)}_{\text{soft}} - v^{(\ell)}_{\text{hard}}\big)
\end{aligned}
$$

### Key Design

- Miss representation: CPU-side ROSA returns -1 for misses; GPU-side applies +1 offset uniformly, making Emb[0] = 0
- Straight-Through Estimator (STE): Forward pass injects hard copy (v_hard), backward pass propagates gradients through soft expectation (v_soft)
- Per-layer independent parameters: {W_lm^(ℓ), Emb^(ℓ)} are not shared across layers

### Run-Length Collapse in Index View

For historical discrete sequence $z_{<t}$, apply run-length collapse only in ROSA's index view:

$$\mathcal{C}(1,1,1,2,2,3) = (1,2,3)$$

Maintain an online Suffix Automaton (SAM) on the collapsed string $\mathcal{C}(z_{<t})$. During query, find the longest and most recent historical match ending with the current suffix, and output its next different token as $y_t$.

Retrieve-then-commit: At step t, first retrieve $y_t$ from the collapsed index, then write $z_t$ to the index using collapse policy.

### CPU/GPU Parallelism

- GPU: Windowed attention + W_lm projection + softmax + embedding/residual
- CPU: Parameter-free ROSA (SAM construction and matching) operates on integer sequences
- Data path: logits → argmax(z) placed in pinned memory → async transfer to CPU → CPU computes y → return to GPU for embedding and STE composition; runs concurrently with GPU attention

---

## Core Advantages

- Infinite-range, lossless memory: ROSA performs exact substring matching and copies true historical successors, with retrieval range unlimited by window size
- Minimal GPU overhead: ROSA core is parameter-free and runs on CPU; GPU only performs small vocabulary projection and representation injection, significantly saving the O(T²) cost of global attention
- Windowed attention handles arbitrary-length sequences: Cross-window information passes through ROSA's discrete channel, while windowed attention only performs local fusion
- Maintains trainability: STE achieves both hard copy and soft gradients; per-layer independent {W_lm^(ℓ), E^(ℓ)} enables the network to learn internal discrete language, evolving layer-by-layer into symbol streams efficiently retrievable by ROSA

---

## Citation

Original ROSA discussion: https://x.com/BlinkDL_AI/status/1976912771985146184

Proposes using Suffix Automaton for neurosymbolic infinite-range, lossless information propagation in LLMs as a memory/retrieval channel beyond attention.

> Bo is great, no need to say more

---

## Additional Notes

More detailed experiments, hardware optimizations, more powerful ROSA-Tuning methods, and related papers will be released soon.

Additionally, this project will release new ROSA-Tuning methods daily in the coming days. Stay tuned.

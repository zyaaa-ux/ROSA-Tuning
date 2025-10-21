  # ROSA-Tuning: Tuning Global Attention into Better Window Attention

## What is ROSA-Tuning

ROSA is a non-neural memory mechanism running on CPUs, capable of achieving perfect recall and precise matching over infinitely long contexts.

ROSA-Tuning integrates this mechanism with modern large language models, enabling them to handle arbitrarily long inputs using only a fixed-length attention window, while achieving better performance than full global attention.

During inference, ROSA only needs to cache the rosa_token_id corresponding to the input sequence, instead of the costly kv_cache, achieving an O(1) spatiotemporal complexity per step.

The current implementation already supports multi-GPU, multi-node, and multi-core training, and more efficient methods are under continuous development.

---

## Experimental Results

### Setup

- **Base model:** Qwen3-0.6B with global attention or windowed attention  
- **Training:** 28,000 samples, 1 epoch on PG-19 training set, original model frozen, only ROSA adapters trained  
- **Evaluation:** PG-19 test set, sequence length 16k, window size 1024  

### Perplexity (PPL) Comparison On Validation Datasets

| Model | PPL |
|:------|----:|
| Global Attention | 31.96 |
| Windowed Attention | 465.59 |
| Windowed Attention + ROSA (2025.10.13) | 25.96 |
| Windowed Attention + ROSA (2025.10.14) | 20.01 |
| Windowed Attention + ROSA (2025.10.15) | 19.93 |
| Windowed Attention + ROSA (2025.10.20, 1/8 parameters) | 19.82 |

ROSA enables windowed attention to outperform the global attention baseline.

In our experiments, ROSA-Tuning demonstrates even greater advantages on 32k and longer sequences. Here, we only present the 16k test results, but you are welcome to explore other configurations if interested.

---

## Core Advantages

- **Infinite-range, lossless memory:** ROSA performs exact substring matching and copies true historical successors, with retrieval range unlimited by window size.  
- **Minimal GPU overhead:** ROSA core is parameter-free and runs on CPU; GPU only performs small vocabulary projection and representation injection, significantly saving the $O(T^2)$ cost of global attention.  
- **Windowed attention handles arbitrary-length sequences:** Cross-window information passes through ROSA's discrete channel, while windowed attention only performs local fusion.  


---

## Method Overview

### Core Formula

For the $\ell$-th layer hidden representation $h^{(\ell)} \in \mathbb{R}^{T\times d}$:

$$
h^{(\ell+1)} = h^{(\ell)} + \mathrm{Attn}^{(\ell)}_{\text{win}}(\mathrm{LN}(h^{(\ell)})) + v^{(\ell)} + \mathrm{MLP}^{(\ell)}(\mathrm{LN}(\cdot))
$$

The ROSA branch $v^{(\ell)}$ is computed as:

$$
\begin{aligned}
\textbf{logits}^{(\ell)} &= W_{\rm lm}^{(\ell)}\,\mathrm{LN}(h^{(\ell)}) \in \mathbb{R}^{T\times K_\ell}, \\
z^{(\ell)} &= \arg\max(\textbf{logits}^{(\ell)}) \in \{0,\ldots,K_\ell-1\}^T,\\
y^{(\ell)} &= \mathrm{ROSA}_{\text{collapse}}(z^{(\ell)}) \in \{-1,0,\ldots,K_\ell-1\}^T,\\
p^{(\ell)} &= \mathrm{softmax}(\textbf{logits}^{(\ell)}/\tau_\ell),\\
v^{(\ell)}_{\text{hard}} &= \mathrm{Emb}^{(\ell)}[y^{(\ell)}+1],\\
v^{(\ell)}_{\text{soft}} &= p^{(\ell)}\,\mathrm{Emb}^{(\ell)}[1:],\\
v^{(\ell)} &= v^{(\ell)}_{\text{hard}} + \mathrm{sg}(v^{(\ell)}_{\text{soft}} - v^{(\ell)}_{\text{hard}})
\end{aligned}
$$

---

## Update · 2025-10-20

- The fusion method of ROSA has been finalized as “pre-attn”, and a time_shift mechanism has been introduced to assist feature fusion.

- The logic related to the quantization head has been revised: each classification head now processes only part of the dimensions, and the output of the ROSA module is formed by concatenating the embeddings corresponding to the IDs output from all classification heads. This approach reduces the number of additional parameters introduced by ROSA-Tuning to one-eighth of the original, while achieving better performance.

- The QKV-ROSA module is currently under development. Most of the core code has been completed, and we are now addressing the backpropagation issue for the K head. The update is expected to be released tomorrow or the day after.

---

## Update · 2025-10-16
- Added a new ROSA fusion method `pre_attn`, which injects ROSA representations before the attention layer, allowing the window attention to operate directly in the $(h + v)$ space.  Currently, its performance is slightly worse than that of `post_attn`.  
- Added code to enable the model to load ROSA-related files and perform inference.
- Added a high-speed C++ kernel and several other optimizations, achieving a 2× speedup.

---

## Update · 2025-10-15
- **LCG moved to embedding level** — gradients are computed on the injected embedding branch $v^{(\ell)}$ instead of token-level edits.  
- **10× faster** — event-gated CPU/GPU overlap, pinned memory, vectorized top-k, Numba kernels.  
- **ROSA fix** — strict *retrieve-then-commit* SAM with rightmost tracking ensures **longest and latest** matches.


---

## Update · 2025-10-14

- Added **multi-route ROSA**: each layer has M independent routes $\{W_{lm}^{(\ell,m)}, E^{(\ell,m)}\}$; injected as the mean of all routes.  
- Removed all temperature and scaling factors; fully hard forward path.  
- Replaced STE with **Local Counterfactual Gradient (LCG)**; CPU computes $\Delta L_i(k)$ by "change-one-token" simulation and writes position-wise contrastive gradients to logits.  


---

## Citation

Original ROSA discussion: [https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8)


Proposes using Suffix Automaton for neurosymbolic infinite-range, lossless information propagation in LLMs as a memory/retrieval channel beyond attention.

> Bo is great, no need to say more.

---

## Additional Notes

More detailed experiments, hardware optimizations, more powerful ROSA-Tuning methods, and related papers will be released soon.  

Additionally, this project will release new ROSA-Tuning methods daily in the coming days. Stay tuned.

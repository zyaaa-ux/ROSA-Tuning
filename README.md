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

ROSA enables windowed attention to outperform the global attention baseline.

---

## Core Advantages

- **Infinite-range, lossless memory:** ROSA performs exact substring matching and copies true historical successors, with retrieval range unlimited by window size.  
- **Minimal GPU overhead:** ROSA core is parameter-free and runs on CPU; GPU only performs small vocabulary projection and representation injection, significantly saving the $O(T^2)$ cost of global attention.  
- **Windowed attention handles arbitrary-length sequences:** Cross-window information passes through ROSA's discrete channel, while windowed attention only performs local fusion.  
- **Maintains trainability:** STE achieves both hard copy and soft gradients; per-layer independent $\{W_{lm}^{(\ell)}, E^{(\ell)}\}$ enables the network to learn internal discrete language, evolving layer-by-layer into symbol streams efficiently retrievable by ROSA.  

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

## Update · 2025-10-14

- Added **multi-route ROSA**: each layer has M independent routes $\{W_{lm}^{(\ell,m)}, E^{(\ell,m)}\}$; injected as the mean of all routes.  
- Removed all temperature and scaling factors; fully hard forward path.  
- Replaced STE with **Local Counterfactual Gradient (LCG)**; CPU computes $\Delta L_i(k)$ by "change-one-token" simulation and writes position-wise contrastive gradients to logits.  



$$
g^{(\ell)}_t = \frac{\partial \mathcal{L}}{\partial v^{(\ell)}_t}.
$$

$$
\Delta L_i^{(\ell,m)}(k) \approx \sum_{t \in S_i^{(\ell,m)}(k)} (g^{(\ell)}_t)^\top (E^{(\ell,m)}[\hat y^{(\ell,m)}_t(i \!\leftarrow\! k)] - E^{(\ell,m)}[\hat y^{(\ell,m)}_t]).
$$

$$
\frac{\partial \mathbb{E}[\mathcal{L}]}{\partial \mathbf{logits}^{(\ell,m)}_{i,v}} = p^{(\ell,m)}_{i,v} \Big( \Delta L_i^{(\ell,m)}(v) - \sum_{k} p^{(\ell,m)}_{i,k} \Delta L_i^{(\ell,m)}(k) \Big).
$$

$$
\frac{\partial \mathcal{L}}{\partial W_{\rm lm}^{(\ell,m)}} = (u^{(\ell)})^\top \frac{\partial \mathcal{L}}{\partial \mathbf{logits}^{(\ell,m)}}, \quad \frac{\partial \mathcal{L}}{\partial E^{(\ell,m)}[r]} = \frac{1}{M} \sum_t \mathbf{1}\{\hat y^{(\ell,m)}_t = r\} g^{(\ell)}_t.
$$

---

## Update · 2025-10-15
- **LCG moved to embedding level** — gradients are computed on the injected embedding branch $v^{(\ell)}$ instead of token-level edits.  
- **10× faster** — event-gated CPU/GPU overlap, pinned memory, vectorized top-k, Numba kernels.  
- **ROSA fix** — strict *retrieve-then-commit* SAM with rightmost tracking ensures **longest and latest** matches.



$$
\mathbf{logits}^{(\ell,m)} = W_{\mathrm{lm}}^{(\ell,m)}\,u^{(\ell)}, \quad
z^{(\ell,m)}=\arg\max \mathbf{logits}^{(\ell,m)}, \quad
y^{(\ell,m)}=\mathrm{ROSA}_{\mathrm{collapse}}(z^{(\ell,m)})
$$

$$
v^{(\ell)}=\frac{1}{M}\sum_{m=1}^{M} E^{(\ell,m)}[y^{(\ell,m)}+1], \quad
g^{(\ell)}=\frac{\partial\mathcal{L}}{\partial v^{(\ell)}}
$$

$$
S^{(\ell,m)}_{t,k}=(g^{(\ell)}_t)^\top E^{(\ell,m)}[k], \quad
\Delta L^{(\ell,m)}_{i}(c)=
\sum_{t\in S^{(\ell,m)}_{i}(c)} \left(
S^{(\ell,m)}_{t,\,\hat{y}(i\leftarrow c)+1}-S^{(\ell,m)}_{t,\,\hat{y}+1}
\right)
$$

$$
\frac{\partial\mathcal{L}}{\partial \mathbf{logits}^{(\ell,m)}_{i,c}}
= p^{(\ell,m)}_{i,c}\left(
\Delta L^{(\ell,m)}_{i}(c)
-\sum_{k} p^{(\ell,m)}_{i,k}\,\Delta L^{(\ell,m)}_{i}(k)
\right)
$$

$$
\frac{\partial\mathcal{L}}{\partial W_{\mathrm{lm}}^{(\ell,m)}}
= (u^{(\ell)})^\top
\frac{\partial\mathcal{L}}{\partial \mathbf{logits}^{(\ell,m)}}, \quad
\frac{\partial\mathcal{L}}{\partial E^{(\ell,m)}[r]}
= \frac{1}{M}\sum_{t}\mathbb{1}\{\hat{y}^{(\ell,m)}_t=r\}\,g^{(\ell)}_t
$$

$$
\mathcal{C}(1,1,1,2,2,3)=(1,2,3), \quad
y_t=\mathrm{nextdiff}(\mathrm{SAM}(\mathcal{C}(z_{<t>})))
$$

---

## Update · 2025-10-16
- Added a new ROSA fusion method `pre_attn`, which injects ROSA representations before the attention layer, allowing the window attention to operate directly in the $(h + v)$ space.  Currently, its performance is slightly worse than that of `post_attn`.  
- Also added code to enable the model to load ROSA-related files and perform inference.
- Fix some bugs.
- 2× faster — Added a high-performance C++ kernel and several other optimizations. 



$$u = \mathrm{LN}(h + v)$$
$$\tilde{h} = h + \mathrm{Attn}_{\text{win}}(u)$$
$$h^{+} = \tilde{h} + \mathrm{MLP}\big(\mathrm{LN}(\tilde{h})\big)$$

---

## Citation

Original ROSA discussion: [https://x.com/BlinkDL_AI/status/1976912771985146184](https://x.com/BlinkDL_AI/status/1976912771985146184)

Proposes using Suffix Automaton for neurosymbolic infinite-range, lossless information propagation in LLMs as a memory/retrieval channel beyond attention.

> Bo is great, no need to say more.

---

## Additional Notes

More detailed experiments, hardware optimizations, more powerful ROSA-Tuning methods, and related papers will be released soon.  

Additionally, this project will release new ROSA-Tuning methods daily in the coming days. Stay tuned.

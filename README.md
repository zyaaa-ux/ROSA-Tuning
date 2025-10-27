  # ROSA-Tuning: Tuning Global Attention into Better Window Attention

## What is ROSA-Tuning

ROSA(RWKV Online Suffix Automation) is a non-neural memory mechanism running on CPUs, capable of achieving perfect recall and precise matching over infinitely long contexts.

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
| Windowed Attention + ROSA (2025.10.22) | 19.63 |

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

## Update · 2025-10-27

- The implementation of QKV-ROSA has been further optimized, the computational logic remains exactly the same, but the execution speed is now faster.

- A pure GPU version of QKV-ROSA has been added; however, it runs significantly slower than the CPU version on long sequences. This is because ROSA itself involves almost no arithmetic computation but relies heavily on multi-core sequence matching. The GPU version is mainly intended to help illustrate the underlying principles of ROSA.

- Due to limited computational resources, long-text experiments have not yet started. They are expected to begin after this week’s training, with evaluations on LongBench. I’m also collecting your suggested training–testing dataset pairs (please provide both together) in the issue thread — they can be from text, image, or any other modality. If time permits, I’ll run tests using ROSA-Tuning on them.

---

## Update · 2025-10-22

- We have fixed several critical bugs and logical errors in QKV-ROSA released yesterday, along with optimizing the overall codebase. The latest version of QKV-ROSA now runs faster while delivering significantly improved performance.

- The vocab sizes of QK and V have been decoupled, allowing you to configure them independently.

- The core logic of ROSA-Tuning is now relatively complete, and we’ll begin conducting more rigorous and diverse experimental validations. Some results and related code will continue to be updated.


---

## Update · 2025-10-21

- We attempted to add rosa_emb(token_id) to the first layer of the model and tested two schemes: one where it is added to the original word embedding before participating in attention computation (emb_sun), and another where it is added to the residual connection after attention (attn_plus). Currently, attn_plus performs better.

- We use an improved QKV-ROSA method to avoid the “matching being matched” issue and enhance flexibility. This method matches the q sequence with the k sequence to generate the v sequence, respectively focusing on “which query token is most useful at the moment”, “which token can carry more valuable routing”, and “which embedding row should be selected when being routed”. Gradients are obtained through perturbation.

- On the PG-19 task, QKV-ROSA does not significantly outperform the original ROSA. However, from a theoretical perspective, QKV-ROSA possesses greater expressive power, as the original ROSA can be viewed as a constrained version of QKV-ROSA where Q=K=V.

- This project mainly focuses on reproducing and exploring the application of the Bo's method. We express our gratitude to Bo for their open-source spirit and contributions to the AI field.


---

## Update · 2025-10-20

- The fusion method of ROSA has been finalized as “pre-attn”, and a time_shift mechanism has been introduced to assist feature fusion.

- The logic related to the quantization head has been revised: each classification head now processes only part of the dimensions, and the output of the ROSA module is formed by concatenating the embeddings corresponding to the IDs output from all classification heads. This approach reduces the number of additional parameters introduced by ROSA-Tuning to one-eighth of the original, while achieving better performance.


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

This project mainly focuses on reproducing and exploring the application of the Bo's method. We express our gratitude to Bo for their open-source spirit and contributions to the AI field.

> Bo is great, no need to say more.

---

## Additional Notes

More detailed experiments, hardware optimizations, more powerful ROSA-Tuning methods, and related papers will be released soon.  

Additionally, this project will release new ROSA-Tuning methods daily in the coming days. Stay tuned.


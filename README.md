  # ROSA-Tuning: Tuning Global Attention into Better Window Attention

## What is ROSA-Tuning

ROSA(RWKV Online Suffix Automation) is a non-neural memory mechanism running on CPUs, capable of achieving perfect recall and precise matching over infinitely long contexts.

ROSA-Tuning integrates this mechanism with modern large language models, enabling them to handle arbitrarily long inputs using only a fixed-length attention window, while achieving better performance than full global attention.

During inference, ROSA only needs to cache the rosa_token_id corresponding to the input sequence, instead of the costly kv_cache, achieving an O(1) spatiotemporal complexity per step.

The current implementation already supports multi-GPU, multi-node, and multi-core training, and more efficient methods are under continuous development.

---

## Experimental 2 (2025.11.26)

### Setup

- **Base model:** Qwen3-0.6B with global attention or windowed attention  
- **Training:** 3B tokens from prolong-52K, original model frozen, only ROSA adapters trained  
- **Evaluation:** lm-eval  

### Results

| Model                           | HellaSwag (acc_norm) | LAMBADA-OAI (acc) | MMLU (acc) | PIQA (acc) | SciQ (acc) | Winogrande (acc) | niah_single_1-64k (acc) |
|:-------------------------------|----------------------:|--------------------:|------------:|------------:|------------:|-------------------:|-------------------------:|
| Qwen3-0.6B                     |               0.4737  |             0.4013  |     0.4017  |     0.6736  |     0.8730  |            0.5659  |                  1.0000  |
| Qwen3-0.6B (window_attn + rosa)  |               0.4716  |             0.4005  |     0.4013  |     0.6763  |     0.8680  |            0.5635  |                  1.0000  |


The experimental results show that with only a small amount of training, ROSA-Tuning enables a baseline global-attention model to switch to windowed attention while maintaining performance comparable to that of full global attention.

## Experimental 1

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
| Windowed Attention + ROSA (2025.10.20) | 19.82 |
| Windowed Attention + ROSA (2025.10.22) | 19.63 |
| Windowed Attention + ROSA (2025.10.31) | 19.06 |

ROSA enables windowed attention to outperform the global attention baseline.

In our experiments, ROSA-Tuning demonstrates even greater advantages on 32k and longer sequences. Here, we only present the 16k test results, but you are welcome to explore other configurations if interested.

---

## Core Advantages

- **Infinite-range, lossless memory:** ROSA performs exact substring matching and copies true historical successors, with retrieval range unlimited by window size.  
- **Minimal GPU overhead:** ROSA core is parameter-free and runs on CPU; GPU only performs small vocabulary projection and representation injection, significantly saving the $O(T^2)$ cost of global attention.  
- **Windowed attention handles arbitrary-length sequences:** Cross-window information passes through ROSA's discrete channel, while windowed attention only performs local fusion.  


---

## Method Overview · 2025-10-31

For layer $l$ with hidden $h^{(l)} \in \mathbb{R}^{T \times C}$ where $C = R \cdot M$:

$$
h^{(l+1)} = h^{(l)} + \text{Attn}^{(l)}_{\text{win}}(\text{LN}(h^{(l)})) + \text{ROSA}^{(l)}(h^{(l)}) + \text{MLP}^{(l)}(\text{LN}(\cdot))
$$

$$
R = \frac{C}{M}, \quad K = 2^M, \quad b^{X}_{b,t,c} = \mathbf{1}[x^{X}_{b,t,c} > 0], \quad a^{X}_{b,t,r} = \sum_{m=0}^{M-1} b^{X}_{b,t,(r,m)} \cdot 2^m, \quad X \in \{Q, K, V\}
$$

$$
s_0 = 0, \; \text{sym}_0 = a^{K}_{b,0,r}, \qquad a^{K}_{b,t,r} \neq a^{K}_{b,t-1,r} \Rightarrow s_{l+1} = t, \; \text{sym}_{l+1} = a^{K}_{b,t,r}
$$

$$
\text{rcap}(t) = \max \{ l \mid s_l \le t \}
$$

$$
ns = \mathrm{match\_next}(s, x), \quad rpos = e[ns], \quad nxt = rpos + 1
$$

$$
\tau_{b,r,t} = \begin{cases}
s_{nxt}, & \text{if match success and } nxt \le \text{rcap}(t), \\
-1, & \text{otherwise}
\end{cases}
$$

For Q-run first symbol $a$ and bit $j$:

$$
a^{(j,0)} = a \wedge \neg(1 \ll j), \qquad a^{(j,1)} = a \vee (1 \ll j)
$$

$$
\tau^{(j,b)} = \begin{cases}
s_{nxt^{(j,b)}}, & \text{if match success}, \\
-1, & \text{otherwise}
\end{cases}
$$

$$
\Delta = \text{Emb}_1 - \text{Emb}_0, \qquad y_{b,t,c} = \mathbf{1}[\tau \ge 0] \cdot \left(\text{Emb}_0[c] + \Delta[c] \cdot \mathbf{1}[v_{b,\tau,r,m} > 0]\right)
$$

$$
\text{ROSA}^{(l)}(h) = \text{Linear}(y)
$$

$$
p^{Q} = \sigma(T_Q q), \qquad p^{K} = \sigma(T_K k), \qquad p^{V} = \sigma(T_V v)
$$

$$
\theta_{b,t,c} = \frac{\partial \mathcal{L}}{\partial y_{b,t,c}} \cdot \Delta[c], \quad \theta_{b,t,r,m} \text{ is reshape of } \theta_{b,t,c}
$$

$$
S^{V}_{b,r,\tau,m} = \sum_{t: \tau_{b,r,t} = \tau} \theta_{b,t,r,m}, \qquad \frac{\partial \mathcal{L}}{\partial v_{b,t,r,m}} = p^{V}_{b,t,r,m}(1 - p^{V}_{b,t,r,m}) S^{V}_{b,r,t,m}
$$

Let $\mathcal{V}^{Q}_{b,r,\tau,m} \in \{\mathbf{1}[v > 0], p^{V}\}$:

$$
d^{(j)}_{b,t,r} = \sum_{m} \theta_{b,t,r,m} \left(\mathcal{V}^{Q}_{b,r,\tau^{(j,1)},m} - \mathcal{V}^{Q}_{b,r,\tau^{(j,0)},m}\right)
$$

$$
\frac{\partial \mathcal{L}}{\partial q_{b,t,r,j}} = p^{Q}_{b,t,r,j}(1 - p^{Q}_{b,t,r,j}) d^{(j)}_{b,t,r}
$$


$$
U^{(b)}_{b,r,l,j} = \sum_{t} \sum_{m} \theta_{b,t,r,m} \mathcal{V}^{K,(b)}_{b,r,s_l,m}, \quad \Delta U_{b,r,l,j} = U^{(1)}_{b,r,l,j} - U^{(0)}_{b,r,l,j}
$$

$$
\frac{\partial \mathcal{L}}{\partial k_{b,s_l,r,j}} = p^{K}_{b,s_l,r,j}(1 - p^{K}_{b,s_l,r,j}) \Delta U_{b,r,l,j}, \qquad \frac{\partial \mathcal{L}}{\partial k_{b,t \neq s_l,r,j}} = 0
$$


---

## Update · 2025-11-4

- Fixed a minor bug in the previous implementation of gradient backpropagation.

- More detailed and comprehensive experiments are currently underway.

---

## Update · 2025-10-31

- In this week’s comprehensive experiments, we found that the previous ROSA-Tuning performed exceptionally well on seen data categories (e.g., training and testing on the same dataset). However, its performance dropped significantly on unseen tasks (such as LongBench), showing only a marginal advantage over window attention. After thorough analysis and discussion, we attributed this issue to the inherent generalization limitation of the discretization structure.

- Through theoretical analysis and extensive exploration of different discretization strategies, we identified a method that achieves a strong balance between performance and generalization — Bo’s bit-based discretization approach.

- Some have compared ROSA to Transformer-VQ, but this is incorrect. ROSA is built upon three core components: discretization (bitization), matching (suffix automaton), and training (perturbation). All three are fundamentally different from those in Transformer-VQ. Moreover, ROSA’s technology is significantly more advanced, with its bitization method demonstrating far superior performance and generalization compared to Transformer-VQ.

---

## Update · 2025-10-27

- The implementation of QKV-ROSA has been further optimized, the computational logic remains exactly the same, but the execution speed is now faster.

- A pure GPU version of QKV-ROSA(Referenced https://github.com/wjie98/rosa_soft) has been added; however, it runs significantly slower than the CPU version on long sequences. This is because ROSA itself involves almost no arithmetic computation but relies heavily on multi-core sequence matching. The GPU version is mainly intended to help illustrate the underlying principles of ROSA.

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


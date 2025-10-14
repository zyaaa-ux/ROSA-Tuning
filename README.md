# ROSA-Tuning: Tuning Global Attention into Better Window Attention

## TL;DR

**ROSA-Tuning** adds minimal per-layer discrete adapter parameters $W_{lm}^{(\ell)}, E^{(\ell)}$ while keeping the pretrained model frozen, enabling windowed attention models to achieve even better ultra-long-range memory capabilities than global attention.

**Practical results:** After replacing global attention with windowed attention in Qwen3-0.6B, adding ROSA branch and freezing the original model, training for one epoch on 28,000 samples achieves better PPL on PG-19-16k than the original global attention model.

**Resource advantages:** ROSA core is parameter-free and runs on CPU, while GPU only handles small matrix projection and representation injection; windowed attention can process arbitrary-length contexts (limited by CPU-side index memory), significantly reducing the quadratic complexity cost of global attention.

---

## Experimental Results

### Setup

- **Base model:** Qwen3-0.6B with global attention or windowed attention  
- **Training:** 28,000 samples, 1 epoch, original model frozen, only ROSA adapters trained  
- **Evaluation:** PG-19 dataset, sequence length 16k, window size 1024  

### Perplexity (PPL) Comparison On Validation Datasets

| Model | PPL |
|:------|----:|
| Global Attention | 31.96 |
| Windowed Attention | 465.59 |
| Windowed Attention + ROSA (2025.10.13) | **25.96** |
| Windowed Attention + ROSA (2025.10.14) | **20.01** |

ROSA enables windowed attention to outperform the global attention baseline.

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

### Layer $\ell \ge 1$ (window attention + multi-route ROSA)

$$
u^{(\ell)} = \mathrm{LN}_1(h^{(\ell)}), \quad a^{(\ell)} = \mathrm{Attn}^{(\ell)}_{\mathrm{win}}(u^{(\ell)}).
$$

$$
\mathbf{logits}^{(\ell,m)} = W_{\rm lm}^{(\ell,m)}u^{(\ell)}, \quad p^{(\ell,m)} = \mathrm{softmax}(\mathbf{logits}^{(\ell,m)}), \quad z^{(\ell,m)} = \arg\max(\mathbf{logits}^{(\ell,m)}).
$$

### Index-view RLE and ROSA Retrieval

$$
c^{(\ell,m)}_{<t} = \mathcal{C}(z^{(\ell,m)}_{<t}), \quad \mathcal{C}(1,1,1,2,2,3) = (1,2,3).
$$

$$
y^{(\ell,m)}_t = \begin{cases} c^{(\ell,m)}_{j_{\text{last}}(z^{(\ell,m)}_t)+1}, & j_{\text{last}}(z^{(\ell,m)}_t)+1 < |c^{(\ell,m)}_{<t}|,\\ -1, & \text{otherwise}. \end{cases}
$$

$$
\hat y^{(\ell,m)}_t = y^{(\ell,m)}_t + 1, \quad E^{(\ell,m)}[0] = 0.
$$

### Multi-route Injection and Output

$$
v^{(\ell)} = \frac{1}{M}\sum_{m=1}^{M} E^{(\ell,m)}[\hat y^{(\ell,m)}],
$$

$$
h^{(\ell+1)} = h^{(\ell)} + a^{(\ell)} + v^{(\ell)} + \mathrm{MLP}^{(\ell)}(\mathrm{LN}_2(h^{(\ell)} + a^{(\ell)} + v^{(\ell)})).
$$

### LCG (Local Counterfactual Gradient)

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

### Run-Length Collapse in Index View

For historical discrete sequence $z_{<t}$, apply run-length collapse only in ROSA's index view:

$$
\mathcal{C}(1,1,1,2,2,3) = (1,2,3)
$$

Maintain an online **Suffix Automaton (SAM)** on the collapsed string $\mathcal{C}(z_{<t})$.  
During query, find the longest and most recent historical match ending with the current suffix, and output its next different token as $y_t$.

**Retrieve-then-commit:**  
At step *t*, first retrieve $y_t$ from the collapsed index, then write $z_t$ to the index using collapse policy.

---

### CPU/GPU Parallelism

- **GPU:** Windowed attention + $W_{lm}$ projection + softmax + embedding/residual  
- **CPU:** Parameter-free ROSA (SAM construction and matching) operates on integer sequences  

---

## Core Advantages

- **Infinite-range, lossless memory:** ROSA performs exact substring matching and copies true historical successors, with retrieval range unlimited by window size.  
- **Minimal GPU overhead:** ROSA core is parameter-free and runs on CPU; GPU only performs small vocabulary projection and representation injection, significantly saving the $O(T^2)$ cost of global attention.  
- **Windowed attention handles arbitrary-length sequences:** Cross-window information passes through ROSA's discrete channel, while windowed attention only performs local fusion.  
- **Maintains trainability:** STE achieves both hard copy and soft gradients; per-layer independent $\{W_{lm}^{(\ell)}, E^{(\ell)}\}$ enables the network to learn internal discrete language, evolving layer-by-layer into symbol streams efficiently retrievable by ROSA.  

---

## Citation

Original ROSA discussion: [https://x.com/BlinkDL_AI/status/1976912771985146184](https://x.com/BlinkDL_AI/status/1976912771985146184)

Proposes using Suffix Automaton for neurosymbolic infinite-range, lossless information propagation in LLMs as a memory/retrieval channel beyond attention.

> Bo is great, no need to say more.

---

## Additional Notes

More detailed experiments, hardware optimizations, more powerful ROSA-Tuning methods, and related papers will be released soon.  

Additionally, this project will release new ROSA-Tuning methods daily in the coming days. Stay tuned.

# Rigorous Mathematical Derivation of Rotary Positional Embedding (RoPE)

## Abstract

Rotary Positional Embedding (RoPE) is a novel approach to encoding positional information in transformer-based language models. Unlike traditional absolute or relative positional encodings, RoPE encodes position by rotating the query and key vectors in their embedding space through a rotation matrix that depends on position. This document provides a rigorous mathematical derivation of RoPE, exploring its theoretical foundations, properties, and connections to other positional encoding methods.

---

## 1. Problem Statement and Motivation

### 1.1 The Position Problem in Transformers

The self-attention mechanism in transformers, originally introduced by Vaswani et al. (2017), computes attention scores as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q$, $K$, and $V$ are the query, key, and value matrices, respectively.

**Critical Limitation:** Self-attention is *permutation-equivariant* — it has no inherent notion of token order. If we permute the input tokens, the output is permuted in the same way without any change in the attention computation. This means the model has no built-in understanding that "the cat sat on the mat" differs from "mat the on sat cat the."

### 1.2 What Positional Encoding Must Achieve

For a language model to understand sequence order, positional encoding must satisfy:

1. **Uniqueness:** Each position must have a distinct representation
2. **Deterministic:** The encoding for a given position must be consistent
3. **Bounded:** The encoding should not grow unboundedly with position
4. **Relative Information:** It should facilitate the computation of relative positions
5. **Extrapolation:** The encoding should generalize to longer sequences than seen during training

### 1.3 Motivation for RoPE

Traditional approaches have limitations:

- **Absolute Positional Encoding (APE):** Sinusoidal embeddings (Vaswani et al., 2017) or learned embeddings add position information to token embeddings. However, they don't directly encode relative positions in the attention computation.

- **Relative Positional Encoding (RPE):** Methods like Shaw et al. (2018) modify attention to incorporate relative positions, but often introduce computational overhead or memory complexity.

**RoPE's Innovation:** Encode position by *rotating* query and key vectors by a position-dependent angle. This naturally encodes relative positions through the property that the dot product of two rotated vectors depends only on their relative rotation angle.

---

## 2. Mathematical Notation and Definitions

### 2.1 Basic Notation

| Symbol | Description |
|--------|-------------|
| $d$ | Model dimension (embedding size) |
| $d_k$ | Key/query dimension, typically $d_k = d/h$ where $h$ is heads |
| $n$ | Sequence length |
| $m, i$ | Position indices in a sequence |
| $\mathbf{x}_m \in \mathbb{R}^d$ | Token embedding at position $m$ |
| $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$ | Linear projection matrices |
| $\mathbf{q}_m, \mathbf{k}_m, \mathbf{v}_m$ | Query, key, value vectors at position $m$ |
| $\theta$ | Base angle/frequency parameter |
| $\theta_i$ | Rotation frequency for dimension pair $i$ |

### 2.2 Vector Partitioning

For RoPE, we partition each $d$-dimensional vector into $d/2$ two-dimensional subspaces:

$$\mathbf{x} = (x^{(0)}, x^{(1)}, \ldots, x^{(d/2-1)})$$

where each $x^{(i)} \in \mathbb{R}^2$ represents a pair of dimensions.

### 2.3 Rotation Matrix Definition

**Definition 2.1 (2D Rotation Matrix).** For angle $\theta$, the 2D rotation matrix is:

$$\mathbf{R}(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

**Definition 2.4 (Position-Dependent Rotation Matrix).** For position $m$ and base frequency $\theta$, the position-specific rotation matrix for dimension pair indexed by $i$ is:

$$\mathbf{R}(m, \theta_i) = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

where $\theta_i = \theta^{-2i/d}$ is the exponentially decaying frequency.

### 2.4 Frequency Selection

**Definition 2.5 (RoPE Frequencies).** The rotation frequencies follow a geometric progression:

$$\theta_i = \theta^{-2i/d} = \frac{1}{\theta^{2i/d}}$$

for $i \in \{0, 1, \ldots, d/2 - 1\}$, where $\theta$ is typically set to $10000$ (following the original transformer paper).

---

## 3. Step-by-Step Derivation of RoPE

### 3.1 Goal: Position-Aware Attention

We want the attention score between positions $m$ (query) and $n$ (key) to depend only on the relative distance $m - n$, not on absolute positions.

**Desired Property:**

$$f(\mathbf{q}, \mathbf{k}, m, n) = g(\mathbf{q}, \mathbf{k}, m - n)$$

where $f$ is the attention scoring function.

### 3.2 The Rotation Approach

Consider applying a position-dependent transformation $\mathbf{R}(m)$ to queries and $\mathbf{R}(n)$ to keys. The dot product becomes:

$$(\mathbf{R}(m)\mathbf{q})^T (\mathbf{R}(n)\mathbf{k}) = \mathbf{q}^T \mathbf{R}(m)^T \mathbf{R}(n) \mathbf{k}$$

If $\mathbf{R}(m)$ is orthogonal (i.e., $\mathbf{R}(m)^T = \mathbf{R}(m)^{-1}$), and if:

$$\mathbf{R}(m)^T \mathbf{R}(n) = \mathbf{R}(n - m)$$

then we achieve our goal.

### 3.3 Deriving the Rotation-Based Attention

**Step 1: Assume Rotation by Position**

Let each 2D subspace of dimension pair $(2i, 2i+1)$ be rotated by angle $m \cdot \theta_i$ for the query at position $m$:

$$\begin{pmatrix} q_{2i}^{(m)} \\ q_{2i+1}^{(m)} \end{pmatrix} = \mathbf{R}(m\theta_i) \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

**Step 2: Express Rotated Components**

Explicitly:

$$q_{2i}^{(m)} = q_{2i}\cos(m\theta_i) - q_{2i+1}\sin(m\theta_i)$$
$$q_{2i+1}^{(m)} = q_{2i}\sin(m\theta_i) + q_{2i+1}\cos(m\theta_i)$$

Similarly for keys at position $n$:

$$k_{2i}^{(n)} = k_{2i}\cos(n\theta_i) - k_{2i+1}\sin(n\theta_i)$$
$$k_{2i+1}^{(n)} = k_{2i}\sin(n\theta_i) + k_{2i+1}\cos(n\theta_i)$$

**Step 3: Compute the Dot Product**

The attention score contribution from dimension pair $i$:

$$s_i = q_{2i}^{(m)}k_{2i}^{(n)} + q_{2i+1}^{(m)}k_{2i+1}^{(n)}$$

**Step 4: Expand and Simplify**

Substituting the rotated values:

$$\begin{aligned}
s_i &= [q_{2i}\cos(m\theta_i) - q_{2i+1}\sin(m\theta_i)][k_{2i}\cos(n\theta_i) - k_{2i+1}\sin(n\theta_i)] \\
&\quad + [q_{2i}\sin(m\theta_i) + q_{2i+1}\cos(m\theta_i)][k_{2i}\sin(n\theta_i) + k_{2i+1}\cos(n\theta_i)]
\end{aligned}$$

**Step 5: Expand and Group Terms**

$$\begin{aligned}
s_i &= q_{2i}k_{2i}\cos(m\theta_i)\cos(n\theta_i) - q_{2i}k_{2i+1}\cos(m\theta_i)\sin(n\theta_i) \\
&\quad - q_{2i+1}k_{2i}\sin(m\theta_i)\cos(n\theta_i) + q_{2i+1}k_{2i+1}\sin(m\theta_i)\sin(n\theta_i) \\
&\quad + q_{2i}k_{2i}\sin(m\theta_i)\sin(n\theta_i) + q_{2i}k_{2i+1}\sin(m\theta_i)\cos(n\theta_i) \\
&\quad + q_{2i+1}k_{2i}\cos(m\theta_i)\sin(n\theta_i) + q_{2i+1}k_{2i+1}\cos(m\theta_i)\cos(n\theta_i)
\end{aligned}$$

**Step 6: Apply Trigonometric Identities**

Group like terms and use:
- $\cos(a)\cos(b) + \sin(a)\sin(b) = \cos(a - b)$
- $\sin(a)\cos(b) - \cos(a)\sin(b) = \sin(a - b)$

$$\begin{aligned}
s_i &= q_{2i}k_{2i}[\cos(m\theta_i)\cos(n\theta_i) + \sin(m\theta_i)\sin(n\theta_i)] \\
&\quad + q_{2i+1}k_{2i+1}[\sin(m\theta_i)\sin(n\theta_i) + \cos(m\theta_i)\cos(n\theta_i)] \\
&\quad + q_{2i}k_{2i+1}[\sin(m\theta_i)\cos(n\theta_i) - \cos(m\theta_i)\sin(n\theta_i)] \\
&\quad + q_{2i+1}k_{2i}[\cos(m\theta_i)\sin(n\theta_i) - \sin(m\theta_i)\cos(n\theta_i)]
\end{aligned}$$

$$\begin{aligned}
s_i &= q_{2i}k_{2i}\cos((m-n)\theta_i) + q_{2i+1}k_{2i+1}\cos((m-n)\theta_i) \\
&\quad + q_{2i}k_{2i+1}\sin((m-n)\theta_i) - q_{2i+1}k_{2i}\sin((m-n)\theta_i)
\end{aligned}$$

**Step 7: Final Form**

$$s_i = (q_{2i}k_{2i} + q_{2i+1}k_{2i+1})\cos((m-n)\theta_i) + (q_{2i}k_{2i+1} - q_{2i+1}k_{2i})\sin((m-n)\theta_i)$$

Or equivalently:

$$s_i = \mathbf{q}_{[2i:2i+2]}^T \mathbf{R}((m-n)\theta_i) \mathbf{k}_{[2i:2i+2]}$$

**Key Result:** The attention score depends only on the **relative position** $m - n$ and the rotation frequency $\theta_i$, not on absolute positions $m$ and $n$ individually.

---

## 4. The Rotation Matrix Formulation

### 4.1 Block-Diagonal Structure

RoPE applies different rotation angles to each dimension pair. This can be represented as a block-diagonal matrix:

$$\mathbf{R}(m) = \begin{pmatrix} 
\mathbf{R}(m\theta_0) & \mathbf{0} & \cdots & \mathbf{0} \\
\mathbf{0} & \mathbf{R}(m\theta_1) & \cdots & \mathbf{0} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{0} & \mathbf{0} & \cdots & \mathbf{R}(m\theta_{d/2-1})
\end{pmatrix}$$

where each block is a $2 \times 2$ rotation matrix.

### 4.2 Compact Representation Using Complex Numbers

RoPE has an elegant formulation using complex arithmetic. Represent each dimension pair as a complex number:

$$\tilde{q}_i = q_{2i} + i \cdot q_{2i+1}$$
$$\tilde{k}_i = k_{2i} + i \cdot k_{2i+1}$$

The position encoding becomes complex multiplication:

$$\tilde{q}_i^{(m)} = \tilde{q}_i \cdot e^{im\theta_i}$$
$$\tilde{k}_i^{(n)} = \tilde{k}_i \cdot e^{in\theta_i}$$

**Theorem 4.1 (Complex Formulation).** The dot product of rotated vectors equals the real part of the product of one vector with the complex conjugate of the other:

$$\langle \mathbf{q}^{(m)}, \mathbf{k}^{(n)} \rangle = \sum_{i=0}^{d/2-1} \text{Re}\left(\tilde{q}_i^{(m)} \cdot \overline{\tilde{k}_i^{(n)}}\right)$$

where $\overline{\tilde{k}}$ denotes the complex conjugate.

**Proof:** By direct computation:

$$\tilde{q}_i^{(m)} \cdot \overline{\tilde{k}_i^{(n)}} = \tilde{q}_i e^{im\theta_i} \cdot \overline{\tilde{k}_i} e^{-in\theta_i} = \tilde{q}_i \cdot \overline{\tilde{k}_i} \cdot e^{i(m-n)\theta_i}$$

Taking the real part:

$$\text{Re}\left(\tilde{q}_i \cdot \overline{\tilde{k}_i} \cdot e^{i(m-n)\theta_i}\right) = \text{Re}\left(\tilde{q}_i \cdot \overline{\tilde{k}_i}\right)\cos((m-n)\theta_i) - \text{Im}\left(\tilde{q}_i \cdot \overline{\tilde{k}_i}\right)\sin((m-n)\theta_i)$$

which matches our earlier derivation.

### 4.3 Efficient Computation

The rotation can be computed efficiently without explicit matrix multiplication:

$$\text{RoPE}(\mathbf{x}, m) = \mathbf{x} \odot \cos(m\boldsymbol{\theta}) + \text{rotate}_{\frac{\pi}{2}}(\mathbf{x}) \odot \sin(m\boldsymbol{\theta})$$

where:
- $\boldsymbol{\theta} = (\theta_0, \theta_0, \theta_1, \theta_1, \ldots, \theta_{d/2-1}, \theta_{d/2-1})^T$ (repeated twice for each dimension pair)
- $\text{rotate}_{\frac{\pi}{2}}(\mathbf{x})$ rotates each 2D subspace by $90°$: $(x_1, -x_0, x_3, -x_2, \ldots)$
- $\odot$ denotes element-wise multiplication

---

## 5. Properties and Advantages of RoPE

### 5.1 Relative Position Encoding (RPE) Property

**Theorem 5.1 (Relative Position).** RoPE encodes relative positions:

$$\langle \text{RoPE}(\mathbf{q}, m), \text{RoPE}(\mathbf{k}, n) \rangle = \langle \mathbf{q}, \mathbf{R}(m-n)\mathbf{k} \rangle$$

**Proof:** This follows from the group property of rotation matrices: $\mathbf{R}(m)\mathbf{R}(n)^T = \mathbf{R}(m-n)$. $\square$

### 5.2 Long-Distance Decay

**Theorem 5.2 (Long-Distance Decay).** For typical frequency choices, the contribution of dimension $i$ to attention decays with relative distance $|m - n|$.

**Proof Sketch:** The dot product contribution from dimension pair $i$ involves $\cos((m-n)\theta_i)$. For large $|m-n|$, the rapidly oscillating cosine effectively averages out, reducing the effective attention weight. This creates an implicit locality bias. $\square$

### 5.3 Extrapolation Capability

**Theorem 5.3 (Length Extrapolation).** RoPE naturally extrapolates to sequence lengths longer than seen during training.

Unlike learned positional embeddings which have a fixed-size lookup table, RoPE computes position encoding on-the-fly using the rotation formula. For position $m > L_{\text{max}}^{\text{train}}$, the encoding is well-defined and follows the same mathematical structure.

### 5.4 Group Properties

**Theorem 5.4 (Group Structure).** The set of rotation matrices $\{\mathbf{R}(m) : m \in \mathbb{Z}\}$ forms an abelian group under matrix multiplication.

**Properties:**
- **Closure:** $\mathbf{R}(m)\mathbf{R}(n) = \mathbf{R}(m+n)$
- **Identity:** $\mathbf{R}(0) = \mathbf{I}$
- **Inverse:** $\mathbf{R}(m)^{-1} = \mathbf{R}(-m) = \mathbf{R}(m)^T$
- **Commutativity:** $\mathbf{R}(m)\mathbf{R}(n) = \mathbf{R}(n)\mathbf{R}(m)$

These properties ensure consistency in how positions are composed.

### 5.5 Computational Efficiency

RoPE has the same computational complexity as standard attention:

| Method | Additional Parameters | Extra Computation |
|--------|----------------------|-------------------|
| Learned APE | $O(L \cdot d)$ | $O(1)$ (lookup) |
| Sinusoidal APE | $0$ | $O(1)$ (precompute) |
| Relative RPE | $O(L \cdot d)$ or $O(L^2)$ | $O(L^2 \cdot d)$ |
| **RoPE** | $0$ | $O(L \cdot d)$ |

---

## 6. Connection to Other Positional Encoding Methods

### 6.1 Absolute Positional Encoding (APE)

**Sinusoidal Encoding (Vaswani et al., 2017):**

$$\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$\text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

**Connection to RoPE:**
RoPE uses the same frequency schedule but applies it differently:
- APE: Add positional encoding to token embeddings
- RoPE: Rotate query/key vectors by these angles

**Mathematical relationship:** The frequencies $\theta_i$ in RoPE match the sinusoidal frequencies in the original transformer.

### 6.2 Relative Positional Encoding

**Shaw et al. (2018) Approach:**

Modifies attention to:

$$e_{ij} = \frac{\mathbf{q}_i^T \mathbf{k}_j + \mathbf{q}_i^T \mathbf{r}_{j-i}}{\sqrt{d_k}}$$

where $\mathbf{r}_{j-i}$ is a learned relative position embedding.

**Comparison with RoPE:**

| Aspect | Shaw RPE | RoPE |
|--------|----------|------|
| Parameters | $O(L \cdot d)$ | $0$ (fixed) |
| Relative info | Explicit additive | Implicit in rotation |
| Computation | Additional lookup | Same complexity |
| Extrapolation | Limited | Better |

**Theorem 6.1 (RoPE as Implicit RPE).** RoPE encodes relative positions implicitly through the rotation property, eliminating the need for explicit relative position embeddings.

### 6.3 ALiBi and Other Alternatives

**ALiBi (Press et al., 2021):** Adds a penalty to attention scores based on distance:

$$e_{ij} = \mathbf{q}_i^T \mathbf{k}_j - m \cdot |i - j|$$

**Comparison:**
- ALiBi uses a linear distance penalty
- RoPE uses sinusoidal rotations
- Both achieve length extrapolation but through different mechanisms

### 6.4 Unified View

RoPE can be seen as a unification of APE and RPE:

$$\underbrace{\text{Rotation by } m}_{\text{Absolute aspect}} \xrightarrow{\text{Dot Product}} \underbrace{\text{Function of } m-n}_{\text{Relative aspect}}$$

The rotation encodes absolute position, but the attention mechanism (dot product) extracts only relative information.

---

## 7. Theoretical Analysis

### 7.1 Information Theoretic Perspective

RoPE preserves the information content of the original embeddings while adding position information through an invertible transformation (rotation).

**Theorem 7.1 (Information Preservation).** The mapping $\mathbf{x} \mapsto \text{RoPE}(\mathbf{x}, m)$ is information-preserving up to the precision of the representation.

**Proof:** Rotation matrices have determinant 1 and are invertible, preserving information theoretic content. $\square$

### 7.2 Geometric Interpretation

In the high-dimensional embedding space, RoPE rotates vectors along $d/2$ orthogonal 2D planes. Each plane rotates at a different frequency $\theta_i$, creating a helix-like structure in the high-dimensional space.

The frequencies are chosen so that:
- Low indices (early dimensions): Slow rotation (long wavelength)
- High indices (later dimensions): Fast rotation (short wavelength)

This allows the model to capture both coarse and fine-grained positional information.

### 7.3 Frequency Selection Analysis

**Theorem 7.2 (Frequency Coverage).** The geometric progression of frequencies $\theta_i = \theta^{-2i/d}$ provides coverage across multiple scales of position differences.

The wavelength for dimension pair $i$ is:

$$\lambda_i = \frac{2\pi}{\theta_i} = 2\pi \cdot \theta^{2i/d}$$

For $\theta = 10000$ and $d = 512$:
- $i = 0$: $\lambda_0 \approx 2\pi \approx 6.28$
- $i = d/4$: $\lambda_{128} \approx 2\pi \cdot 100 \approx 628$
- $i = d/2 - 1$: $\lambda_{255} \approx 2\pi \cdot 10000 \approx 62832$

This spans wavelengths from single-digit positions to tens of thousands.

---

## 8. Implementation Considerations

### 8.1 Numerical Stability

When computing $\sin$ and $\cos$ for large positions ($m \gg 0$), numerical precision must be considered. Modern implementations typically:
1. Precompute $\cos$ and $\sin$ values for a reasonable range
2. Use periodicity properties for very large positions

### 8.2 Interpolation and Extrapolation

**NTK-aware Scaling (2023):** When extrapolating to much longer sequences, methods like "NTK-aware" scaling modify the base $\theta$:

$$\theta' = \theta \cdot s^{d/(d-2)}$$

where $s$ is the scaling factor.

This modifies the frequency distribution to better handle extended contexts.

### 8.3 Flash Attention Compatibility

RoPE can be fused into Flash Attention kernels for efficient computation. Since RoPE is an element-wise transformation of queries and keys before the attention computation, it fits well into the tiled computation pattern of Flash Attention.

---

## 9. References

1. **Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y.** (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. *Neurocomputing*, 568, 127063. https://doi.org/10.1016/j.neucom.2023.127063

2. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017). Attention is All You Need. *Advances in Neural Information Processing Systems*, 30.

3. **Shaw, P., Uszkoreit, J., & Vaswani, A.** (2018). Self-Attention with Relative Position Representations. *Proceedings of NAACL-HLT*, 464-468.

4. **Press, O., Smith, N. A., & Lewis, M.** (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. *arXiv preprint arXiv:2108.12409*.

5. **Xiao, S., et al.** (2023). Training Large Language Models Efficiently with Sparsely Activated Transformer. https://kaiokendev.github.io/til

6. **Chen, J.** (2023). NTK-Aware Scaled RoPE. https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope/

---

## 10. Summary

RoPE provides an elegant mathematical solution to the positional encoding problem:

1. **Theoretical Soundness:** Built on rotation matrices with well-understood group properties
2. **Relative Position Encoding:** Naturally emerges from the dot product of rotated vectors
3. **Efficiency:** No additional parameters, minimal computational overhead
4. **Extrapolation:** Works for sequences longer than training data
5. **Flexibility:** Compatible with various attention optimizations

The key insight of RoPE is that by rotating query and key vectors by position-dependent angles, the attention score naturally depends only on the relative positions, solving the fundamental challenge of encoding sequential order in permutation-equivariant attention mechanisms.

---

*Document generated for theoretical analysis of Rotary Positional Embedding.*

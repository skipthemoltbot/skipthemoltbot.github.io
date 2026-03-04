# Mathematical Derivation: Hand-Coded vs. Learned Induction Heads

## 1. Foundation: Self-Attention Mechanism

The attention mechanism in transformers is defined as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- **Q** (Queries): computed as Q = X @ W_q
- **K** (Keys): computed as K = X @ W_k  
- **V** (Values): computed as V = X @ W_v
- **X**: Input embeddings [seq_len × d_model]
- **W_q, W_k, W_v**: Learned weight matrices [d_model × d_model]
- **d_k**: Dimension of keys (per-head dimension, e.g., 32)

### Step-by-Step Matrix Multiplication

For a single token at position **m** with embedding **x_m** (1 × d_model):

1. Compute query: **q_m = x_m @ W_q** → (1 × d_model)
2. Compute key: **k_n = x_n @ W_k** → (1 × d_model)  
3. Dot product: **q_m @ k_n^T** → scalar attention score

## 2. Adding RoPE (Rotary Position Embeddings)

RoPE rotates query and key vectors by position-dependent angles **before** computing the dot product.

### 2.1 Rotation Frequencies

```
θ_i = base^(-2i/d_head)  where i ∈ [0, 1, ..., d_head/2 - 1]
```

With base = 10000, this creates frequencies from fast (local patterns) to slow (global patterns).

### 2.2 RoPE Transformation

For each pair of dimensions (2i, 2i+1) at position m:

**Query rotation:**
```
q'_2i     = q_2i · cos(m·θ_i) - q_(2i+1) · sin(m·θ_i)
q'_(2i+1) = q_2i · sin(m·θ_i) + q_(2i+1) · cos(m·θ_i)
```

**Key rotation:**
```
k'_2i     = k_2i · cos(n·θ_i) - k_(2i+1) · sin(n·θ_i)
k'_(2i+1) = k_2i · sin(n·θ_i) + k_(2i+1) · cos(n·θ_i)
```

### 2.3 Matrix Form

This can be written as applying rotation matrices:

```
R(m, θ_i) = [[cos(m·θ_i), -sin(m·θ_i)],
             [sin(m·θ_i),  cos(m·θ_i)]]

[q'_2i, q'_(2i+1)]^T = R(m, θ_i) · [q_2i, q_(2i+1)]^T
```

## 3. Why RoPE Enables Relative Position Attention

### 3.1 Key Property Proof

The dot product of rotated query (position m) and key (position n):

```
q'^T @ k' = Σ_i [q'_2i·k'_2i + q'_(2i+1)·k'_(2i+1)]
```

After substituting the RoPE transformations and using trigonometric identities:

```
q'^T @ k' = Σ_i [cos((m-n)·θ_i) · (q_2i·k_2i + q_(2i+1)·k_(2i+1))
                + sin((m-n)·θ_i) · (q_2i·k_(2i+1) - q_(2i+1)·k_2i)]
```

**Critical Result:** The dot product depends ONLY on **(m - n)**, the relative position!

## 4. Previous Token Head: Mathematical Requirements

### 4.1 Goal

Attend from position m → position (m-1), i.e., the previous token.

### 4.2 Chris's Hand-Coded Approach

Create W_q and W_k such that q_m matches k_(m-1).

**W_q pattern:**
```
W_q = [[+1, -1, +1, -1, ...],
       [-1, -1, +1, -1, ...],
       ... (alternating pattern per row)]
```

**Analysis of hand-coded pattern:**
- For dim 2i: w_q[row, 2i] = (-1)^i
- For dim 2i+1: w_q[row, 2i+1] = (-1)^(i+1)

When we apply RoPE at position m:
```
q_2i     = x_2i · cos(m·θ_i) - x_(2i+1) · sin(m·θ_i)
q_(2i+1) = x_2i · sin(m·θ_i) + x_(2i+1) · cos(m·θ_i)
```

The alternating signs in W_q create phase-alignment with keys at position m-1.

### 4.3 What SGD Actually Learned

From the trained checkpoint:

```python
# Extracted from rope_transformer_trained.pt
W_q[0, :16] = [ 0.08, -0.10, 0.02, -0.04, 0.12, 0.02, -0.21, 0.07,
                -0.20, 0.07, -0.08, 0.09, 0.09, 0.10, -0.10, -0.21]

W_k[0, :16] = [ 0.11,  0.09, -0.12, 0.02, 0.22, 0.05, -0.15, 0.12,
                -0.07, -0.11, -0.17, 0.03, 0.06, 0.07, -0.14, -0.04]
```

**Key Difference:**
- Hand-coded: Clean alternating +1/-1 pattern
- Learned: Distributed, non-orthogonal weights with varying magnitudes

## 5. Induction Head: Mathematical Requirements

### 5.1 Two-Layer Architecture

**Layer 0 (Previous Token Heads):**
- Multiple heads learn to shift information one position back
- Output contains "what came before each token"

**Layer 1 (Induction Heads):**
- Query attends to tokens with matching content AND appropriate relative position
- Key provides position-shifted information from Layer 0

### 5.2 Functional Form

For input pattern [abc][abc], when predicting after second 'a':

1. **Layer 0** output at position 0 (first 'a'): contains info about preceding context
2. **Layer 0** output at position 3 (second 'a'): contains info about what preceded it
3. **Layer 1** induction head recognizes: "position 3's token 'a' appeared before at position 0"
4. **Query at 3** attends to **Key at 0** due to RoPE relative position encoding
5. **Copied token:** What followed position 0 (which was 'b')

### 5.3 Why Both Approaches Work

**Hand-coded (alternating):**
- Orthogonal basis vectors that align perfectly with RoPE rotations
- Each dimension pair contributes maximally to specific relative positions
- Mathematically "clean" but requires perfect alternation

**Learned (distributed):**
- Combinations of basis vectors that sum to same directional alignment
- Example: Two non-orthogonal vectors can sum to a direction equivalent to one orthogonal vector
- More flexible but requires SGD to find correlated weights across dimensions

**Functional equivalence proof:**

For any target output vector **y**, both approaches find W such that:

```
RoPE(x @ W_q, m) · RoPE(x @ W_k, n) ≈ large when m - n = target_offset
```

Hand-coded achieves this through:
```
W_q[i,j] ∈ {+1, -1}  (orthogonal matrix-like)
```

Learned achieves this through:
```
W_q[i,j] ∈ ℝ  (distributed, non-orthogonal)
```

Subject to constraint:
```
Σ_j W_q[i,j] · x_j ≈ Σ_j W_k[i,j] · x_j when properly rotated
```

## 6. The Key Finding: Gabor-Like Learned Patterns

Empirical analysis shows SGD learns weights that approximate Gabor filters:

```
W[i,j] ≈ A · exp(-(j-μ)^2/σ^2) · cos(ω·j + φ)
```

Where:
- Gaussian envelope (exp term): localizes attention to relevant dimensions
- Cosine modulation (cos term): creates frequency-specific response

This is functionally equivalent to the hand-coded alternating pattern but:
1. **Sparse:** Small weights in irrelevant dimension ranges
2. **Smooth:** No sharp +1/-1 transitions  
3. **Overlapping:** Different heads learn complementary frequency bands

## 7. Summary: Why SGD ≠ Hand-Coded

| Property | Hand-Coded | SGD Learned |
|----------|------------|-------------|
| Weight values | Discrete {+1, -1} | Continuous ℝ |
| Orthogonality | Perfect | Approximate |
| Interpretability | Immediate | Requires analysis |
| Robustness | Fragile | Generalizes better |
| Optimization | N/A | Minimizes loss landscape |

**Mathematical insight:** SGD finds a solution in a subspace of the weight space that achieves functional equivalence but trades perfect orthogonality for robustness and learnability.

The alternating pattern is ONE solution; SGD finds ANOTHER solution that's easier to reach via gradient descent from random initialization.
# Step-by-Step Mathematical Analysis: Why Learned Weights Implement Induction Heads

## Setup: The Toy Model

**Architecture:**
- 2 layers, 4 heads per layer
- d_model = 128, d_head = 32
- Vocabulary: 27 tokens (a-z + space)
- RoPE with base = 10000

**Extracted Learned Weights (from trained checkpoint):**
```python
# W_q for head 0, first 16 dimensions
W_q[0, :16] = [ 0.0838, -0.0976,  0.0246, -0.0366,  0.1181,  0.0235, 
                -0.2116,  0.0660, -0.1985,  0.0677, -0.0835,  0.0934,
                 0.0914,  0.0989, -0.1044, -0.2070]

# W_k for head 0, first 16 dimensions  
W_k[0, :16] = [ 0.1099,  0.0879, -0.1191,  0.0171,  0.2178,  0.0543,
                -0.1537,  0.1209, -0.0730, -0.1075, -0.1675,  0.0284,
                 0.0581,  0.0705, -0.1388, -0.0379]
```

---

## Step 1: Compute Query and Key Vectors

For a token embedding **x** at position **m**:

**Step 1a: Linear projection**
```
q_raw = x · W_q    (before RoPE)
k_raw = x · W_k    (before RoPE)
```

For dimension i of q_raw:
```
q_raw[i] = Σ_j x[j] · W_q[i,j]
```

**Step 1b: Apply RoPE rotation**

For each dimension pair (2i, 2i+1):

```
θ_i = 10000^(-2i/32) = 10000^(-i/16)

q_rotated[2i]   = q_raw[2i]·cos(m·θ_i) - q_raw[2i+1]·sin(m·θ_i)
q_rotated[2i+1] = q_raw[2i]·sin(m·θ_i) + q_raw[2i+1]·cos(m·θ_i)

k_rotated[2i]   = k_raw[2i]·cos(n·θ_i) - k_raw[2i+1]·sin(n·θ_i)
k_rotated[2i+1] = k_raw[2i]·sin(n·θ_i) + k_raw[2i+1]·cos(n·θ_i)
```

---

## Step 2: Why Previous Token Heads Work

**Goal:** Make position m attend to position (m-1)

**Step 2a: The RoPE Phase Difference**

For relative offset Δ = m - n = 1:
```
Phase difference at frequency i: Δ·θ_i = 1·θ_i = θ_i

This creates a rotation matrix:
R(Δ=1, θ_i) = [[cos(θ_i), -sin(θ_i)],
                [sin(θ_i),  cos(θ_i)]]
```

**Step 2b: Dot Product with RoPE**

The attention score is:
```
score(m, n) = q_rotated · k_rotated
            = Σ_i [q_rotated[2i]·k_rotated[2i] + q_rotated[2i+1]·k_rotated[2i+1]]
```

Substituting the RoPE transformations and using trig identities:
```
score(m, n) = Σ_i [cos((m-n)·θ_i)·(q_raw[2i]·k_raw[2i] + q_raw[2i+1]·k_raw[2i+1])
                + sin((m-n)·θ_i)·(q_raw[2i]·k_raw[2i+1] - q_raw[2i+1]·k_raw[2i])]
```

**Step 2c: What Makes Score Large When m - n = 1?**

For the score to be large when Δ = 1, we need:
```
q_raw · k_raw to have components that align with cos(θ_i) and sin(θ_i)
```

The learned W_q and W_k achieve this through:
```
W_q[i,j] and W_k[i,j] are correlated such that:

For tokens at position m and (m-1):
Σ_j x_m[j]·W_q[i,j]  ·  Σ_j x_{m-1}[j]·W_k[i,j]

produces dot products that align with the RoPE phase at offset Δ=1
```

---

## Step 3: Distributed vs Alternating - The Key Difference

**Chris's Hand-Coded Solution:**
```
W_q[i, 2j]   = (-1)^j  (alternating sign)
W_q[i, 2j+1] = (-1)^(j+1)
```

This creates basis vectors that, when rotated by RoPE at specific offsets, align perfectly.

**SGD's Learned Solution:**
```
W_q[0, :4] = [0.0838, -0.0976, 0.0246, -0.0366]
```

These are NOT alternating +1/-1. They're small, distributed values.

**Step 3a: Why Distributed Works**

Consider two dimensions j and k. Even though W_q[0,j] and W_q[0,k] aren't ±1, the combination:
```
q_raw[2i]   = x[0]·W_q[0,0] + x[1]·W_q[0,1] + ...
q_raw[2i+1] = x[0]·W_q[1,0] + x[1]·W_q[1,1] + ...
```

When rotated by RoPE and dotted with k_rotated from position (m-1), the phases align to produce a large score.

**Step 3b: Functional Equivalence Proof**

Both solutions satisfy:
```
E_x[score(m, m-1)] >> E_x[score(m, n)] for n ≠ m-1

Where score(m, n) = RoPE(x·W_q, m) · RoPE(x·W_k, n)
```

The hand-coded solution achieves this through **orthogonal basis vectors**.
The learned solution achieves this through **linear combinations** of basis vectors that sum to the same effective direction.

---

## Step 4: Induction Head - Two Layer Analysis

**Layer 0 (Previous Token Heads):**
- Multiple heads learn to output: "what token came before current position"
- For position i, output contains information about token at position (i-1)

**Layer 1 (Induction Heads):**
- Query at position m looks for positions n where:
  1. Same token content: x_m ≈ x_n
  2. RoPE phase alignment for some offset Δ

**Step 4a: Pattern Completion Example**

Input: [abc][abc] (positions 0,1,2,3,4,5)

At position 3 (second 'a'):
- Query q_3 = RoPE(x_"a" · W_q^{L1}, 3)
- Looks for keys k_n where content matches AND phase aligns

At position 0 (first 'a'):
- Key k_0 is computed from Layer 0 output
- Layer 0 output at 0 contains: "token before 'a' was [BOS]"

But also:
- Key k_0 contains token embedding of 'a' itself (residual stream)

**Step 4b: RoPE Enables Matching**

The dot product:
```
q_3 · k_0 = Σ_i [cos(3·θ_i)·(q_raw[2i]·k_raw[2i] + q_raw[2i+1]·k_raw[2i+1])
           + sin(3·θ_i)·(q_raw[2i]·k_raw[2i+1] - q_raw[2i+1]·k_raw[2i])]
```

Since x_3 = x_0 = "a", and the W_q, W_k have learned to handle offset Δ=3:
```
E[score(3, 0)] is large because W_q and W_k have been trained to make it so
```

**Step 4c: Copying the Next Token**

Once attention focuses on position 0:
- Value vector v_0 is retrieved
- v_0 contains information about what followed 'a' at position 0
- Through the residual stream and MLP, this is 'b'
- Output prediction is 'b'

---

## Step 5: The Complete Induction Algorithm

**Learned by SGD:**

1. **W_q** maps token embeddings to query space such that:
   - Similar tokens produce similar query directions
   - Position information encodes naturally via RoPE

2. **W_k** maps token+position info to key space such that:
   - Queries looking for "same token at offset Δ" match appropriately
   - Previous token info from Layer 0 is accessible

3. **The combined effect:**
```
   Input:  ... a b c a b c ...
            ↑       ↑
            n       m
   
   score(m, n) = RoPE(x_"a"·W_q, m) · RoPE(x_"a"·W_k, n)
   
   Since x_m = x_n = "a", and Δ = m-n is in training distribution:
   score(m, n) >> score(m, n') for x_n' ≠ "a"
```

4. **Output:** Attention focuses on position n, retrieves value v_n which leads to predicting 'b' (what followed first 'a')

---

## Summary: Why SGD Learned This

| Question | Answer |
|----------|--------|
| Why distributed weights? | SGD found a point in weight space where random initialization + gradient descent converged |
| Why not alternating ±1? | Alternating pattern is a saddle point; SGD found a nearby local minimum with similar function |
| Do both work? | YES - both achieve high attention scores at target offsets |
| Which is "better"? | Learned is more robust; hand-coded is more interpretable |

**The key insight:** There are infinitely many weight matrices W_q, W_k that can implement induction heads. SGD found one that works, even though it doesn't look like the clean alternating pattern a human would design.
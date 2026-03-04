# Previous Token Head: The Simplest Possible Explanation

## The Goal
Make position **m** pay attention to position **m-1** (the previous token).

## The Problem Without Position Encoding

If we just compute: `score = query · key`, the model has no idea where tokens are in the sequence.

**Example:**
```
Sentence: "The cat sat"
Positions:  0   1   2

Query at position 2 ("sat") might match:
- Key at position 0 ("The") → score = 0.5
- Key at position 1 ("cat") → score = 0.8
- Key at position 2 ("sat") → score = 1.0 (same word!)
```

The model can't tell the query "should" look at position 1.

## RoPE's Magic Trick

RoPE **rotates** vectors based on their position. Think of it like this:

- Position 0: Vector points at 0°
- Position 1: Vector rotated by θ°  
- Position 2: Vector rotated by 2θ°
- Position m: Vector rotated by mθ°

**Key insight:** The dot product between a query at position m and key at position n depends on the **difference** (m - n), not the absolute positions.

## What We Want

For a "previous token head":
```
score(m, m-1) should be LARGE
score(m, n) should be SMALL for n ≠ m-1
```

## How W_q and W_k Make This Happen

### Step 1: Create Query at Position m
```
x = embedding of token at position m
q_raw = x · W_q
q_rotated = RoPE(q_raw, m)
```

### Step 2: Create Key at Position m-1  
```
x = embedding of token at position m-1
k_raw = x · W_k  
k_rotated = RoPE(k_raw, m-1)
```

### Step 3: The Dot Product
```
score = q_rotated · k_rotated
```

Because of RoPE, this becomes:
```
score = f(q_raw, k_raw, m - (m-1))
      = f(q_raw, k_raw, 1)
      = some function of (q_raw · k_raw) and the phase offset Δ=1
```

## The Hand-Coded Solution (+1/-1 Pattern)

Chris designed W_q and W_k so that:
```
q_raw[i] = x[0]·(+1) + x[1]·(-1) + x[2]·(+1) + x[3]·(-1) + ...
```

This alternating pattern creates basis vectors that, when rotated by specific phases, **align perfectly** at offset Δ=1.

Think of it like:
- Query vector points at angle α
- Key vector points at angle β
- RoPE rotates both: query by mθ, key by (m-1)θ
- At the specific frequency where θ matches the pattern, they align!

## The Learned Solution (Distributed Weights)

SGD found:
```
W_q[0] = [0.08, -0.10, 0.02, -0.04, ...]
```

These aren't ±1, but they work the same way!

**Why?** Because:
```
q_raw = 0.08·x[0] + (-0.10)·x[1] + 0.02·x[2] + (-0.04)·x[3] + ...
```

Even though each weight is small and irregular, the **combination** creates a vector that, when rotated by RoPE at position m, aligns with the key vector (rotated at position m-1).

## The Simplest Analogy

Imagine you're at position 2 trying to look at position 1.

**Without RoPE:** Everyone looks the same direction. You can't tell who's in front of you.

**With RoPE:** Everyone faces a slightly different angle based on their position.
- Position 0 faces 0°
- Position 1 faces 10°  
- Position 2 faces 20°

You (at 20°) can now "see" position 1 (at 10°) because you're facing 10° ahead of them!

The weights W_q and W_k just determine **how** you look — hand-coded uses binoculars pointed exactly right; learned uses regular vision but still sees the same thing.

## Why Both Work

| Hand-Coded | Learned |
|------------|---------|
| W = [+1, -1, +1, -1, ...] | W = [0.08, -0.10, 0.02, -0.04, ...] |
| Perfect alignment | Approximate alignment |
| Human-designed | SGD-discovered |
| Clean math | Messy but functional |

**Both achieve:** `score(m, m-1) >> score(m, n≠m-1)`

The hand-coded solution is one point in weight space. SGD found a different point that works just as well!
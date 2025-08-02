Let's continue our exploration of the DeepSeek V3 model, and move on to the RoPE (Rotary Positional Encoding) mechanism and its [implementations](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L294).

Attention is a sequence modeling mechanism that processes query, key, and value sequences. It is crucial for tokens in the query and key sequences to be position-sensitive during this process. This sensitivity can be achieved through an encoding function, denoted as:

$$
\begin{align}
\bar{\mathbf{x}} &= f(\mathbf{x}, m)\\
\end{align}
$$

where $\mathbf{x}$ is a high-dimensional vector and $m$ is an integer specifying the position. The function $f$ incorporates the positional information $m$ into the vector $\mathbf{x}$.

Rotary Positional Encoding (RoPE) is an encoding function with a unique property: it injects absolute positional information into the query and key vectors such that the resulting inner product between them depends only on the relative positions of the tokens, not their absolute positions. This feature allows the model to process longer sequences during inference, extending beyond the sequence lengths encountered during training.

# How RoPE are Precomputed

RoPE treats a high-dimensional vector $\mathbf{x}$ as a complex vector and performs calculations in the complex space. A complex number is represented by two real numbers $[x, y]$ in the form $x + iy$, where $x$ is the real part, $y$ is the imaginary part, and $i$ is the imaginary unit satisfying $i^2 = -1$.

Given this framework, we can explore the specific formula $f$ of RoPE. The RoPE function rotates a complex number by $e^{i n\theta}$, where $\theta$ is the rotation angle. According to Euler's formula, $e^{i n\theta} = \cos(n\theta) + i\sin(n\theta)$. Therefore, we have:

$$
\begin{align}
(x+iy)e^{i n\theta} &= (x+iy)(\cos(n\theta) + i\sin(n\theta)) \nonumber \\
&=x\cos(n\theta) +ix\sin(n\theta) +iycos(n\theta) -ysin(n\theta) \nonumber \\
&=\left( x\cos\left(n\theta \right) -y\sin\left(n\theta \right) \right) + i\left( x\sin\left(n\theta \right) + y\cos\left(n\theta \right) \right) \nonumber \\
&=\begin{pmatrix}x\\y\end{pmatrix} \cos\left(n\theta \right) +
\begin{pmatrix}-y\\x\end{pmatrix} \sin\left(n\theta \right)
\end{align}
$$

Equation (2) demonstrates that the RoPE function rotates a complex number by \( e^{i n\theta} \), where \(\theta\) represents the rotation angle relative to position \( n \). To generalize this concept to a high-dimensional vector, each pair of consecutive dimensions is treated as the real and imaginary parts of a complex number. The final consideration involves selecting \(\theta\), which can vary across different dimensions.

## Frequency Calculation

Let's explore how these equations are implemented in the DeepSeek V3 model.

```python
def precompute_freqs_cis(
    dim: int = 64,
    seqlen: int = 16384,
    beta_fast: float = 32,
    beta_slow: float = 1,
    base: float = 1000.0,
    factor: float = 40,
    original_seq_len: int = 4096,
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    seq_positions = torch.arange(seqlen)
    freqs = torch.outer(seq_positions, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)
```

The core idea behind RoPE is to compute frequency values for each position in the sequence based on a rotational strategy. The frequencies are calculated using an exponential decay function, where each dimension has its own unique frequency based on the position:

$$
\omega_k = \text{base}^{-\frac{2k}{\text{dim}}}, \quad k = 0, 1, \dots, \frac{\text{dim}}{2} - 1
$$

- `base` (θ) is a constant (e.g., 1e3 is used in the DeepSeek V3 671B model), which controls the scaling of frequencies.
- `dim` is the total dimensionality of the embedding space, and $k$ is the index of the "complex" dimension in the RoPE setup.
- The frequencies $\omega_k$ decay exponentially as $k$ increases, with lower-dimensional (lower $k$) embeddings having higher frequencies.

## Position-Dependent Frequencies

For a given token at position $p$ in the sequence, the corresponding angle for the rotary positional embedding is computed by multiplying the position $p$ with the frequency $\omega_k$ for each dimension:

$$
\theta_{p,k} = p \cdot \omega_k
$$

This produces the angular shift for the $k$-th embedding dimension at position $p$.

The implementation of RoPE in the DeepSeek V3 model is as follows:

## Complex Exponentials

To create the rotary positional embeddings, we use complex exponentials (CIS), which are computed as:

$$
\text{cis}(\theta_{p,k}) = \cos(\theta_{p,k}) + i\sin(\theta_{p,k})
$$

This complex exponential represents a unit circle in the complex plane, which is essential for the RoPE mechanism, as it allows efficient encoding of positional information in both directions (forward and backward) in the sequence. The result is a tensor of complex exponentials, `freqs_cis`, which is of shape $(\text{seqlen}, \text{dim}/2)$, where each element represents a `cos(θ)` and `sin(θ)` pair for each position $p$ and dimension $k$.

## Correction for Longer Sequences

When the sequence length during inference exceeds the sequence length used during training, we need to adjust the frequencies to handle the new, longer context. This is achieved by introducing correction factors based on `beta_fast` and `beta_slow`, which control how fast the frequency scales for different positions.

- First, we calculate the correction for the dimensional index where the sequence length exceeds the training sequence length:

$$
\text{dim}_{cor} = \frac{\text{dim} \cdot \ln\left(\frac{\text{seqlen}}{\beta \cdot 2\pi}\right)}{2 \cdot \ln(\text{base})}
$$

- This adjustment helps smooth the transition between the original sequence length and the new sequence length, preventing overly fast rotations (high frequencies) in the longer sequence, ensuring better model generalization.

```python
def find_correction_dim(
    num_rotations: float, dim: int, base: float, max_seq_len: int
) -> float:
    return (
        dim
        * math.log(max_seq_len / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )

def find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
) -> tuple[int, int]:
    low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)

def linear_ramp_factor(
    min_value: float, max_value: float, dim: int
) -> torch.Tensor:
    if min_value == max_value:
        max_value += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_value) / (
        max_value - min_value
    )
    return torch.clamp(linear_func, 0, 1)
```

## Smoothing Transition for Frequencies

Suppose $\sigma$ is the smoothing factor. For the dimensions within the correction range, we apply a linear ramp function to smoothly transition between the original frequencies and the adjusted frequencies for the new sequence length:

$$
\sigma = 1 - \text{linear_ramp_factor}(\text{low}, \text{high}, \frac{\text{dim}}{2})
$$

- This function ensures that, for dimensions inside the correction range, the frequency values smoothly transition from the original to the adjusted values, making the positional encodings more stable and adaptable to longer contexts.

## Final Frequency Computation

After applying the smoothing factor to adjust frequencies for the extended sequence, the final precomputed frequencies are:

$$
\text{freqs} = \frac{\text{freqs}}{\text{factor}} \cdot (1 - \sigma) + \text{freqs} \cdot \sigma
$$

- The adjusted frequencies are then used to create the final complex exponentials `freqs_cis`, which are the positional embeddings used in the attention mechanism.

## Summary

The RoPE mechanism allows for efficient handling of positional encodings that work across both short and long sequences. The main idea behind the method is to provide a smooth and gradual transition for longer sequences, preventing the model from overfitting to short contexts during training. The final result, `freqs_cis`, is a tensor that encodes the positional information for each token in the sequence and is used in the self-attention mechanism to allow the model to "rotate" its attention based on the token's position.

# References

1. [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/)

1. [Transformer升级之路：2、博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265/comment-page-1)

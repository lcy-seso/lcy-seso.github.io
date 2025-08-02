In my last post, I attempted to derive an online update formula for the attention mechanism. However, the resulting formula isn't optimal for high-performance real-world implementations. There are two issues left undiscussed:

1. **Granularity of the Computation**: The formula treats the query as a vector, which is too granular. In practice, it’s better to work with matrices, as this allows leveraging specialized hardware like tensor cores in GPUs to speed up matrix multiplication.
2. **Numerical Stability**: There are smarter ways to reduce computational load and enhance numerical stability.

Let's address these issues in this post.

<div class="toc-container" style="background-color: #f0f7ff; border-radius: 8px; padding: 15px; margin-bottom: 20px; border-left: 4px solid #9c27b0;">
  <h4 style="margin-top: 0; margin-bottom: 10px; color: #9c27b0;">Table of Contents</h4>
  <ul style="list-style-type: none; padding-left: 10px;">
    <li style="margin-bottom: 8px;"><a href="#recap-the-three-stage-online-updated-attention" style="color: rgba(234, 67, 53, 0.9); text-decoration: none; font-weight: bold;">Recap the Three-Stage Online Updated Attention</a></li>
    <li style="margin-bottom: 8px;"><a href="#the-log-sum-exp-trick-and-numerical-stability" style="color: rgba(66, 133, 244, 0.9); text-decoration: none; font-weight: bold;">The Log-Sum-Exp Trick and Numerical Stability</a></li>
    <li style="margin-bottom: 8px;"><a href="#a-new-matrix-based-online-update-formula" style="color: rgba(52, 168, 83, 0.9); text-decoration: none; font-weight: bold;">A New Matrix-Based Online Update Formula</a></li>
  </ul>
</div>

## Recap the Three-Stage Online Updated Attention

Let's recap the attention formula for calculating $\vec{o_i}$. Figure 1 shows a chunk of the query, key, mask, and value matrices, along with the notation used.

<p align="center"><img src="/images/query-key-mask-value.png" width="100%"/><br>Fig.1 Query-Key-Mask-Value Matrix Visualization and Notation</p>

In Fig 1, the $j$-annotated dimension highlighted in red represents sequence length and carries data dependencies over the entire sequence. Below we give the formula for the attention mechanism corresponding to Fig 1.

$$
\begin{align*}
T[i,j] &= \sum_k \left(Q[i, k] * K [k,j]\right) & [tM, tN] &= \sum_k \left([tM, d]\ *\ [d, tN] \tag{1}\right)\\
---&-----------&---------&-----------\\
S &=T + M & [tM, tN] &= [tM, tN] + [tM, tN] \tag{2}\\
---&-----------&---------&-----------\\
\color{#1E90FF}{\vec{C}_i} &= \max_{\color{#FF0000}{j}} \left(S[i,\color{#FF0000}{j}] \right) & [tM,1]&=[tM, tN] \tag{3}\\
\bar{S} &= \exp \left( S - \vec{C}_i \right)&[tM, tN] &= [tM, tN] - [tM,1]\\
\color{#1E90FF}{\vec{L}_i} &= \sum_{\color{#FF0000}{j}} \bar{S}[i, \color{#FF0000}{j}] &[tM, 1] &= [tM, tN]\\
\vec{P}_i &= \frac{\bar{S}}{\color{#1E90FF}{\vec{L_i}}} & [tM, tN] &=\frac{[tM, tN]}{[tM, 1]}\\
---&-----------&---------&-----------\\
\color{#1E90FF}{O}[i,d] &= \sum_{\color{#FF0000}{j}} \left( P[i,\color{#FF0000}{j}]\ * \ V[\color{#FF0000}{j},d] \right )& [tM, d] &= [tM, tN]\ @ \ [tN, d] \tag{4}
\end{align*}
$$

Flash attention refactors this computation into three stages, motivated by the fact that the sequence-wise normalized softmax operation has an online updated equivalent. This allows computation to occur in a three-stage streaming manner by maintaining running statistics. The three stages are:

1. **Local Compute Stage**: The attention function is computed locally within each chunk, independent of others, with rescaling based on running statistics.

2. **Statistics Update Stage**: Running statistics are updated for numerical stability across the sequence.

3. **Global Rescaling Stage**: After all chunks are processed, the final output is rescaled based on the accumulated running statistics.

This streaming approach enables efficient computation of attention without the need to store large intermediate matrices back in slow external memory. The online update formula for the running statistics must be derived based on the original attention formula for three values:

1. The running maximum values $\color{#1E90FF}{\vec{C}_i}$.
2. The running normalization denominator values in the softmax operation $\color{#1E90FF}{\vec{L}_i}$.
3. Based on $\color{#1E90FF}{\vec{C}_i}$ and $\color{#1E90FF}{\vec{L}_i}$, the formula for the output $\color{#1E90FF}{O}$ also needs to be derived.

Here's how we derived them in the last post:

$$
\begin{align*}
&\vec{C}_i^{\ t} = \max \left( \vec{C}_i,\ \color{#1E90FF}{\vec{C}_i^{\ t-1}} \right) \\
---&-------------------\\
&\Delta \vec{C}_i^{\ t} \triangleq \vec{C}_i - \vec{C}_i^{\ t} \\
&\Delta \vec{C}_i^{\ t-1} \triangleq \color{#1E90FF}{\vec{C}_i^{\ t-1}} - \vec{C}_i^{\ t} \\
&\vec{L}_i^{\ t} = \vec{L}_i \exp \left(\Delta \vec{C}_i^{\ t} \right) + \color{#1E90FF}{\vec{L}_i^{\ t-1}} \exp \left(\Delta \vec{C}_i^{\ t-1} \right) \\
---&-------------------\\
&O^{t} = \frac{\vec{L}_i \exp \left(\Delta \vec{C}_i^{\ t} \right)O + \color{#1E90FF}{\vec{L}_i^{\ t-1}} \exp \left(\Delta \vec{C}_i^{\ t-1} \right) \color{#1E90FF}{O^{t-1}}}{\vec{L}_i^{\ t}}
\end{align*}
$$

## The Log-Sum-Exp Trick and Numerical Stability

Before we proceed, let's take a moment to discuss the log-sum-exp (LSE) trick. The softmax function normalizes an $N$-dimensional vector $x$ into a probability distribution $\vec{p}=\left[p_1, p_2, \cdots, p_N\right]$ as follows:

$$
p_i = \frac{\exp \left(x_i\right)}{\sum_{j=1}^N \exp \left(x_j\right)}, \quad \sum_{i=1}^N p_i = 1 \tag{eq.1}
$$

Dealing with exponential operations can be prone to numerical underflow or overflow. To address this, let's consider computing the log probability $\log p_i$ instead:

$$
\begin{align*}
\log p_i &= \log \left(\frac{\exp \left(x_i\right)}{\sum_{j=1}^N \exp \left(x_j\right)}\right) \\
&=x_i - \log \left(\sum_{j=1}^N \exp \left(x_j\right)\right)
\end{align*}
$$

Then we have $p_i$ as:

$$
\begin{align*}
p_i &= \exp \left(x_i - \underbrace{\log \left(\sum_{j=1}^N \exp \left(x_j\right)\right)}_{LSE} \right) \tag{eq.2}
\end{align*}
$$

From $\text{eq}.1$ and $\text{eq}.2$ we can see that the LSE operation can be used to normalize the values in the denominator of the softmax function. However, you might wonder if this is really helpful for numerical stability since we are still computing the exponential operation.

To address this concern, let's consider normalizing the input values for LSE first. By doing so, we effectively reduce the range of the exponentials before performing the exponential operation. This helps prevent the exponential terms from becoming too large (leading to overflow) or too small (leading to underflow), thereby ensuring numerical stability.

$$
\begin{align*}
L &= \log \left(\sum_{j=1}^N \exp \left(x_j\right)\right) \\
e^{L} &= \sum_{j=1}^N \exp \left(x_j\right) \\
e^{L} &= e^c\sum_{j=1}^N \exp \left(x_j - c\right) \\
L &= c + \log \left(\sum_{j=1}^N \exp \left(x_j - c\right)\right)
\end{align*}
$$

From the above formula, we can see that the LSE operation is advantageous because we can shift the values of $x_j$ by a constant $c$ while still computing the same final result. If we set $c$ to the maximum value of $x_j$, we ensure that the largest positive exponential value is $\exp(0)=1$, and the largest negative exponential value is $\exp(-\infty) = 0$. This helps prevent overflow and underflow in the exponential operation, thus enhancing the numerical stability of the softmax operation's normalization factor.

## A New Matrix-Based Online Update Formula

$$
\begin{align*}
\vec{C}^{\ 0} &= - \inf,\ \overrightarrow{lse}^0 = -\inf, \ O^0 = \mathbf{0} \\
\forall \color{#FF0000}{t} \in &\ [1,\ \frac{N}{tN}] \\
&S= Q\ @\ K^t + M\tag{1}\\
&\color{#FF0000}{\vec{C}^{\ t}} = \max \left( \max_j \left(S\right),\ \color{#FF0000}{\overrightarrow{\text{lse}}^{t-1}} \right)\\
&\bar{S} = \exp \left(S - \color{#FF0000}{\vec{C}^{\ t}} \right) \\
&\color{#FF0000}{O_t} =  \exp \left(\color{#FF0000}{\vec{C}^{\ t-1}} - \color{#FF0000}{\vec{C}^{\ t}} \right)* \color{#FF0000}{O^{t-1}} + \bar{S}\ @\ \color{#FF0000}{V^t} \\
&--------------- \\
&L = \exp \left(\color{#FF0000}{\overrightarrow{\text{lse}}^{t-1}} - \color{#FF0000}{\vec{C}^{t-1}}\right) + \sum_j \bar{S} \tag{2} \\
&\color{#FF0000}{\overrightarrow{\text{lse}}^{\ t}} = \color{#FF0000}{\vec{C}^{t}} + \log \left(L \right) \\
&--------------- \\
O=&\ \color{#FF0000}{O_t} *\exp \left( \color{#FF0000}{\vec{C}^{\ t}} - \color{#FF0000}{\overrightarrow{\text{lse}}^{\ t}}\right) \tag{3}
\end{align*}
$$

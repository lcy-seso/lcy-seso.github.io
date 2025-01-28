Recently, while handling some everyday tasks, I stumbled upon the derivation of the online updated formula in Flash Attention. Even though I've worked it out before, it's easy to forget over time. So, I thought it would be a good idea to jot down some notes about it.

Let's take a look at the original formula for single-head attention (multiple heads are just computed in parallel):

$$
\begin{align*}
\text{Attention}(Q, K, V) = \text{softmax}(QK^T\odot M)V \tag{1}
\end{align*}
$$

This formula is written in a tensorized form to be efficiently computed using pre-optimized linear algebra routines. Here, $Q$, $K$, and $V$ are sequences of vectors with dimensions $\mathbb{R}^{\color{#FF0000}{L} \times d}$. We consider a scenario where $L$ can become extremely large, requiring the input to be stored on slower but larger external memory. The main idea is <span style="color: blue;">to break the computation into chunks. By choosing the chunk size correctly, we can cache the computation of each chunk within high-speed memory, thus improving performance</span>.

## Transitioning from Full Batch to Online Updates

Let's consider a vector $q \in \mathbb{R}^{1 \times d}$, which can be thought of as the smallest unit of $Q$. Next, we break down $K$ and $V$ into smaller parts, labeled as $\{ ks \}_i \in \mathbb{R}^{B \times d}$ and $\{ vs \}_i \in \mathbb{R}^{B \times d}$ respectively, where $B$ is the chunk size. Now, let's apply the Attention formula (1) to two individual chunks of $K$ and $V$, namely $ks_1$, $ks_2$, $vs_1$, and $vs_2$. For now, we won't worry about the final results being correct. We'll break down formula (1) into the following detailed steps, performed on two separate chunks:

$$
\begin{align*}
  &a_1 = \text{dot}(q, ks_1) &a_2 &= \text{dot}(q, ks_2) \\
  &b_1 = \color{#FF0000}{\max(-\inf, a_1)} &b_2 &= \color{#FF0000}{\max(-\inf, a_2)} \\
  &c_1 = a_1 - b_1 &c_2&= a_2 - b_2 \\
  &d_1 = \exp(c_1) &d_2&=\exp(c_2) \\
  &e_1 = \color{#FF0000}{\text{sum}(0, d_1)}&e_2&=\color{#FF0000}{\text{sum}(0,d_2)} \\
  &f_1 = \frac{d_1}{e_1}&f_2&=\frac{d_1}{e_1} \\
  &g_1 = f_1 *vs_1 &g_2&=f_2 * vs_2 \\
  &o_1 = \color{#FF0000}{\text{sum}(0, g_1)} &o_2 &= \color{#FF0000}{\text{sum}(0, g_2)} \\
  &o_{\text{new}} = \color{#1E90FF}{\text{Combiner}(o_1, o_2)}
\end{align*}
$$

In the first stage (all steps except the last line), we compute a partial result for each individual chunk, without worrying about correctness at this point. In the second stage (the last line), we combine these partial results to get the final correct result, ensuring that the final result matches formula $(1)$. So, the main question is: <span style="color: blue;">how do we combine the partial results using the $\color{#1E90FF}{\text{Combiner}}$ ?</span>

### The Combiner Function

To answer this, let's explore why computing on individual chunks independently is incorrect. What causes this incorrectness? We can categorize all the equations above into two groups: the red-highlighted ones and the rest. The first group involves element-wise operations, whereas the outputs of the second group depend on all their inputs.

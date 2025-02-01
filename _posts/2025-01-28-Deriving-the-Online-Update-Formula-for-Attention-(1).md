Recently, while handling some everyday tasks, I stumbled upon the derivation of the online update formula in Flash Attention[<sup>[1]</sup>](#flash-attention). Even though I've worked it out before, it's easy to forget over time. So, I thought it would be a good idea to jot down some notes about it.

In this post, I’ll start with a simple procedure that relies on basic operations and a mechanical, repetitive derivation process. This is different from the method in the original paper for deriving the online update formula for attention. My hope is to potentially automate the process. I’ve been thinking about this problem for quite some time in the past.

## The Batch Computation of Attention

Let's take a look at the original formula for single-head attention (multiple heads are just computed in parallel):

$$
\begin{align}
\text{Attention}(Q, K, V) = \text{softmax}(QK^T\odot M)V \tag{1}
\end{align}
$$

This formula is written in a tensorized form to be efficiently computed using pre-optimized linear algebra routines. Here, $Q$, $K$, and $V$ are sequences of vectors with dimensions $\mathbb{R}^{\color{#FF0000}{L} \times d}$. We consider a scenario where $L$ can become extremely large, requiring the input to be stored on slower but larger external memory. The main idea is <span style="color: blue;">to break the computation into chunks. By choosing the chunk size correctly, we can cache the computation of each chunk within high-speed memory, thus improving performance</span>.

# Transitioning from Batch to Online Updates

Let's consider a vector $q \in \mathbb{R}^{1 \times d}$, which can be thought of as the smallest unit of $Q$. Next, we break down $K$ and $V$ into smaller parts, labeled as $\{ ks \}_i \in \mathbb{R}^{B \times d}$ and $\{ vs \}_i \in \mathbb{R}^{B \times d}$ respectively, where $B$ is the chunk size. Now, let's apply the Attention formula (1) to two individual chunks of $K$ and $V$, namely $ks_1$, $ks_2$, $vs_1$, and $vs_2$. For now, we won't worry about the final results being correct. We'll break down formula (1) into the following detailed steps, performed on two separate chunks (the thrid column gives the shape of the inputs and output):

$$
\begin{align*}
&a_1 = q@ks_1^T &a_2 &= q@ks_2^T &[B,] =& [1,d]@[d,B] \tag{2}\\
&b_1 = \color{#FF0000}{\max(-\inf, a_1)} &b_2 &= \color{#FF0000}{\max(-\inf, a_2)} &[1,]=&\max \left([B,] \right)\tag{3}\\
&c_1 = a_1 - b_1 &c_2&= a_2 - b_2 &[B,]=&[B,]-[1,]\tag{4}\\
&d_1 = \exp(c_1) &d_2&=\exp(c_2) &[B,]=&[B,]\tag{5}\\
&e_1 = \color{#FF0000}{\text{sum}(0, d_1)}&e_2&=\color{#FF0000}{\text{sum}(0,d_2)}&[1,]=&\text{sum}\left([B,]\right)\tag{6}\\
&f_1 = \frac{d_1}{e_1}&f_2&=\frac{d_2}{e_2} &[B,]=&\frac{[B,]}{[1,]}\tag{7}\\
&g_1 = f_1 *vs_1 &g_2&=f_2 * vs_2 &[B,d]=&[B,]*[B,d]\tag{8}\\
&o_1 = \color{#FF0000}{\text{sum}(0, g_1)} &o_2 &= \color{#FF0000}{\text{sum}(0, g_2)} &[1,d]=&\text{sum}\left([B,d] \right) \tag{9}\\
&o_{\text{new}} = \color{#1E90FF}{\otimes(o_1, o_2)} \notag
\end{align*}
$$

In the first stage $($equation $(2)$ to $(9))$, we compute a partial result for each individual chunk, without worrying about correctness at this point. In the second stage (the last line), we combine these partial results to get the final correct result using the combiner function $\color{#1E90FF}{\otimes}$, ensuring that the final result matches formula $(1)$. So, the main question is: <span style="color: blue;">how do we combine the partial results using the $\otimes$ ?</span>

## Background: Element-wise Operators and Reduction Operators

Before diving deeper, let's take a moment to review two fundamental list processing operations: the $\textbf{reduce}$ operator and the $\textbf{element-wise}$ operator.

The reduce operator is a higher-order function that aggregates a list down to a single value. It works as follows:

$$
\begin{align*}
&\textit{reduce} :: (\alpha \rightarrow \beta \rightarrow \beta) \rightarrow \beta \rightarrow ([\alpha] \rightarrow \beta) \\
&\textit{reduce} \ \oplus\ v \ [] = v \\
&\textit{reduce}\ \oplus\ v\ (x:xs) = \oplus \ x\ (\textit{reduce}\ \oplus\ v\ xs)
\end{align*}
$$

The reduce function takes three inputs: a binary operator $\oplus$ (of type $\alpha \rightarrow \beta \rightarrow \beta$), an initial value $v$ (of type $\beta$), and a list $xs$ (of type $[\alpha]$). It then returns a single value of type $\beta$. The process starts by applying the operator $\oplus$ to the initial value and the first element of the list, then to the result of that operation and the second element, and continues this way until all elements are processed.

The element-wise operation applies a unary function $f$ (of type $\alpha \rightarrow \beta$) to each element of a list. This can be described as:

$$
\begin{align*}
\textit{element-wise} &:: (\alpha \rightarrow \beta) \rightarrow ([\alpha] \rightarrow [\beta]) \\
\textit{element-wise}\ f \ xs &= \left[ f \ x_1, \ f \ x_2, \ \ldots, \ f \ x_n \right]
\end{align*}
$$

Since there's no need for communication between the evaluations of $f$ for different elements, the underlying implementation can execute these evaluations in any order it chooses.

## Deriving the Combiner Function

Now, let's examine equations $(2)$ to $(9)$ to understand why computing on individual chunks independently can lead to incorrect results. What causes this issue? We can categorize these equations into two groups: the reduction operations (highlighted in red) and the element-wise operations. In reduction operations, the outputs depend on all their inputs to produce the correct result.

In the full batch formula $(1)$, the reduction operation is applied to the entire sequence $L$. This means that all subsequent operations must wait until the reduction operation has completed its computation on the whole sequence $L$. When we break this dependency, we need to address two key questions to ensure we obtain a final result that is mathematically equivalent to formula $(1)$:

1. How do we combine the partial results obtained in each individual step to get the correct final result?
2. How do we propagate the combined results from one step to the next and correct any inaccuracies?

If we can do this efficiently, we can turn the original batch computation into an online update computation. The answer to question 1 is straightforward:

Suppose $a$ and $b$ are two partial results in one compute step:

- For combining the partial results of element-wise operations, we use the juxtaposition operator (or concatenation): $\otimes(a, b) = [a : b]$, where $:$ denotes the juxtaposition operator.
- For reduction operations, the combiner function remains the same: $\otimes(a, b) = \text{reduce}(0, \oplus, a, b)$.

Now, let's create a simple combiner by recomputing steps $(2)$ through $(9)$. At each step, we merge the partial results for that particular step.

For updating $(2)$: $$a = \left[ a_1: a_2 \right]$$

For updating $(3)$: $$\color{#FF0000}{b = \max(b_1, b_2)}$$

For updating $(4)$:
$$
\begin{align*}
c &=\left[ c_1' : c_2' \right] \\
&= \left[ a_1 - b : a_2 - b \right] \\
&= \left[a_1 - \max(b_1, b_2) : a_2 - \max(b_1, b_2) \right] \\
&= \left[\left(a_1 - b_1 \right) + b_1 - b : \left(a_2 - b_2 \right) + b_2 - b \right] \\
&= \left[c_1 + \Delta c_1 : c_2 + \Delta c_2 \right]
\end{align*}
$$

where we denote $\Delta c_1 := b_1 - b$ and $\Delta c_2 := b_2 - b$.

For updating $(5)$:

$$
\begin{align*}
d &= \left[ d_1' : d_2'\right] \\
&= [\exp\left(c_1 + \Delta c_1 \right): \exp\left( c_2 + \Delta c_2\right)] \\
&= \left[\exp(c_1) \exp(\Delta c_1) : \exp(c_2) \exp(\Delta c_2)\right]
\end{align*}
$$

For updating $(6)$:

$$
\begin{align*}
e &= \text{sum}(e_1', e_2') \\
&= \exp(c_1) \exp(\Delta c_1)+\exp(c_2) \exp(\Delta c_2)\\
&= \color{#FF0000}{e_1 \exp(\Delta c_1) + e_2 \exp(\Delta c_2)}
\end{align*}
$$

For updating $(7)$:

$$
\begin{align*}
f &= \left[ f_1' : f_2' \right] \\
&= \left[\frac{d_1}{e} : \frac{d_2}{e} \right] \\
&= \left[\frac{\exp(c_1) \exp(\Delta c_1)}{e_1 \exp(\Delta c_1) + e_2 \exp(\Delta c_2)} : \frac{\exp(c_2) \exp(\Delta c_2)}{e_1 \exp(\Delta c_1) + e_2 \exp(\Delta c_2)} \right]
\end{align*}
$$

For updating $(8)$:

$$
\begin{align*}
g &= \left[ g_1' : g_2' \right] \\
&= \left[f_1'*vs_1 : f_2'*vs_2 \right] \\
&= \left[\frac{\exp(c_1) \exp(\Delta c_1)*vs_1}{e_1 \exp(\Delta c_1) + e_2 \exp(\Delta c_2)} : \frac{\exp(c_2) \exp(\Delta c_2)*vs_2}{e_1 \exp(\Delta c_1) + e_2 \exp(\Delta c_2)} \right]
\end{align*}
$$

For updating $(9)$:

$$
\begin{align*}
o &= \text{sum}(o_1', o_2') \\
&= \frac{\exp(c_1) \exp(\Delta c_1)*vs_1}{e_1 \exp(\Delta c_1) + e_2 \exp(\Delta c_2)} + \frac{\exp(c_2) \exp(\Delta c_2)*vs_2}{e_1 \exp(\Delta c_1) + e_2 \exp(\Delta c_2)} \\
&= \frac{\exp(\Delta c_1) \color{#0000FF}{\exp(c_1)*vs_1} +\exp(\Delta c_2) \color{#0000FF}{\exp(c_2)*vs_2}}{e_1 \exp(\Delta c_1) + e_2 \exp(\Delta c_2)} \tag{10}
\end{align*}
$$

Now we've got the final combiner function for the online update formula, which is equation $(10)$. However, it still looks a bit messy and isn't very insightful. Let's perform some transformations to simplify it.

$$
\begin{align*}
o_1 &= \text{sum} \left(f_1 *vs_1 \right) \\
o_1 * e_1 & = \text{sum} \left(f_1 * e_1 * vs_1 \right) \\
o_1 *e_1&=\text{sum} \left(d_1 * vs_1 \right)\\
o_1 *e_1 &= \text{sum} \left( \color{#0000FF}{\exp(c_1) * vs_1} \right)
\end{align*}
$$

Therefore, we can rewrite equation $(10)$ as:

$$
\color{#FF0000}{o= \frac{\exp(\Delta c_1)e_1 * o_1+\exp (\Delta c_2)e_2*o_2}{e_1 \exp(\Delta c_1)+ e_2 \exp(\Delta c_2)}} \tag{11}
$$

When using an element-wise operator, the output's shape does not change. However, with a reduction operator, the output becomes smaller. A great feature of equation $(11)$ is that we don't need to recompute everything from scratch. Instead, we can simply save the results of the reduction operator (which are much smaller) and adjust the existing partial results based on them. This makes the combiner function more efficient to implement, and we will discuss this further in the next post.

# References

<div id="flash-attention"></div>

1. Dao, Tri, et al. "[Flashattention: Fast and memory-efficient exact attention with io-awareness](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)." Advances in Neural Information Processing Systems 35 (2022): 16344-16359.

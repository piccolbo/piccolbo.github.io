---
layout: post
title: "The softmedian"
date: "2019-05-14 16:26:45 -0700"
---


Why is there a `softmax` function, but not a `softmedian`? Let's create not one,
but  a few of them.

<!--more-->


$$
\newcommand{\f}[1]{\mathrm{#1}}
\newcommand{\x}{ {\bf x} }
\newcommand{\soft}[1]{\f{soft#1}}
\newcommand{\softmax}{\soft{max}}
\newcommand{\softmin}{\soft{min}}
\newcommand{\softargmin}{\soft{argmin}}
\newcommand{\softargmax}{\soft{argmax}}
\newcommand{\softmedian}{\soft{median}}
\newcommand{\softargmedian}{\soft{argmedian}}
\newcommand{\softabs}{\soft{abs}}
\newcommand{\softsign}{\soft{sign}}
\newcommand{\softmedianrank}{\soft{medianrank}}
\newcommand{\softremedian}{\soft{remedian}}
\newcommand{\softargremedian}{\soft{argremedian}}
\newcommand{\R}{\mathbb{R}}$$

## Basics

People who are familiar with neural networks will likely have heard of operations named
$$\softmax$$ and $$\softargmax$$. There is a tendency to use the shorter of the two
names for both concepts, which would be confusing for this article, so we will keep them
distinct. A $$\softmax$$ function typically need to satisfy two properties:

1.  approximating the max;
2.  being differentiable everywhere.

In this context, by "approximating" we mean that there is an additional parameter
$$\alpha$$ such that, in the limit $$\alpha \to \infty$$, the approximation is equal to
the target function (for the mathematically inclined, we are talking about *pointwise
convergence*). For simplicity, in the following, the dependence on $$\alpha$$ will be
left implicit: consider any "soft" function to have $$\alpha$$ as the last argument.
Here is a possible definition for $$\softmax$$:

$$\softmax(\x) = \frac{1}{\alpha}\log \sum_k{e^{\alpha x_k}}$$


The $$\softargmax$$ function is a function $$\R^n \to (0,1)^n$$ and each element is 1 or
near 1 when the corresponding element in the argument is the maximum of its argument, 0
or near 0 otherwise. In other words, it's an approximation of the indicator function for
the maximum --- a mask identifying the maximum (I will ignore the case of ties for
simplicity). It also needs to be differentiable everywhere. An additional property of
the $$\softargmax$$, useful but perhaps not fundamental to its definition, is to be
normalized to 1, meaning that the sum of the elements of $$\softargmax(x)$$ is always 1.

Its typical use in the context of neural networks is to model a discrete distribution in
a classification problem, owing to its elements' range between 0 and 1 and its
normalization to 1, like a multiclass equivalent of the $$\mathrm{logit}$$ function.
Here is a possible definition:

$$\softargmax(\x) = \left(\frac{e^{\alpha x_i}}{\sum_k e^{\alpha x_k}}\right)_i$$

Considering that the max is one particular quantile, and that differentiable operations
are important in machine learning, let's pose the following question: are there "soft"
versions of other quantiles?

First let's clarify what we are talking about and focus on the median, for starters. As
the reader might have guessed already, these are the necessary properties of a
$$\softmedian$$:

1.  Approximating the median.
2.  Being differentiable everywhere.

Naturally, we can also define a $$\softargmedian$$ function with the following
properties:

1.  Approximates the indicator function of the median
2.  Is differentiable everywhere

## Applications?

If mathematical curiosity is not a sufficient motivation for this inquiry, let me try to
suggest some possible applications of the $$\softmedian$$  in machine learning. If the
$$\softmax$$ is a soft version of a winner-take-all type operation, the $$\softmedian$$
is a consensus one. It reduces to the majority rule in the special case of binary
classification. Consensus is very important in ensemble learning and has applications to
privacy, for instance in the PATE algorithm, whereby models trained on a partition of a
the training set need to mostly agree on the output --- otherwise it has to be somehow
obfuscated, lest it reveals too much about the training set. Of course one could take
the average as consensus, but it is less robust than the median. As a pooling operation,
it doesn't seem that everywhere-differentiability is a necessity, and the max-pool
operation is widely used. But there is an intuitive argument for using soft operations
instead. When backpropagating through max-pool, only one unit in input to that operation
(more only in case of perfect ties) receives a non-zero signal. If different units are
almost tied for the max, then they may alternate, batch after batch, in receiving a
non-zero backpropagation signal, which seems wasteful. In fact, a recently published
[paper](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/mcfee_autopool_taslp_2018.pdf)
defines a soft-max-pool operation with $$\alpha$$ an adaptive parameter. This gives me
more confidence that the $$\softmedian$$ or $$\softargmedian$$ will also find
applications.


## Technicalities

In addition to satisfying the basic properties of the $$\softmedian$$ or
$$\softargmedian$$, I considered simplicity of the definition and low computational
complexity as additional goals. In our quest we will use two mathematical properties of
the composition of functions: it preserves differentiability and pointwise convergence
under mild assumptions. Therefore, if we find a way to express the median as composition
of functions $$f$$ (must be continuous) and $$g$$ which have soft versions $$f_\alpha$$
and $$g_\alpha$$ then  $$f_\alpha \circ g_\alpha$$ is a soft version of $$f \circ g$$
(but pointwise convergence doesn't preserve continuity or differentiability, hence the
limit function $$f$$, $$g$$, and $$f\circ g$$ need not be differentiable or continuous).
And with this, as my probability teacher used to say, "The proof is left as an easy
exercise for the student".

## The median as minmax of subsets

The median, by definition, splits a set in half: elements smaller than it and the rest
(with some details to be filled in for even sized sets and in the case of ties).
Therefore, if one considers all the subsets of size $$\lceil \frac{n}{2}\rceil$$, where
$$n$$ is the sample size, one of them must be the set of elements less than or equal to
the median. Ignoring ties, it is also the only such set, because, if there were two
distinct ones, their union would also consist of elements smaller than the median but
would be larger than half the size of the set, which is a contradiction. So if we take
the maximum for each such subset and consider the minimum of all such maxima, that value
is the median (or *a* median, if multiple values are possible). Replacing the maximum
and minimum operations with their soft counterparts, we have a soft median! More
formally, for a sample $$\x$$:

$$\softmedian(\x) = \underset{s \in 2^x:\,|s| = \lceil|\x|/2\rceil
}{\softmin}\left(\softmax(s)\right)$$

$$\softmin$$ has not been defined but the reader will have no trouble coming up with a
definition based on $$\softmax$$ and the equality $$\min(\x) = -\max(-\x)$$.
Unfortunately, the computation of this incarnation of the $$\softmedian$$ requires an
exponential number of steps. Unless some clever simplification is possible, beyond what
I could come up with, it's not very practical. Conceptually, though, it's very simple.
As a bonus, the generalization to other quantiles is trivial.

## The median as minimizer of the total absolute deviation

It is well known that the median $$m$$ minimizes the total absolute deviation:


$$\underset{m}{\min}\sum_i \lvert x_i-m \rvert$$.

Since the absolute value is not differentiable in 0, it looks like an obstacle on our
quest. But we can replace it with a soft absolute value such as

$$\softabs(x) = \softmax(x, -x)$$.

One could try to find stationary points by differentiation, but it doesn't seem to lead
to a simple solution. Instead, let's try all values in $$\x$$, one of which must be a
median:

$$\softargmedian(\x) = \underset{j}{\softargmin} \left( \sum_i
\softabs(x_i-x_j)\right)$$.

This approach requires $$O(N^2)$$ steps. It is better than our first attempt and, in the
context of neural networks, not any worse than the matrix-vector multiplication that
dominate their computational complexity. On the other hand, compared to the $$O(N)$$
complexity for the regular median, it seems quite a penalty, considering also that no
such penalty is incurred to put the "soft" in $$\softmax$$.

To define the $$\softmedian$$, we just need use the former as weights in a weighted sum
of the sample $$\x$$:

$$\softmedian(\x) = \langle\x,\softargmedian(\x)\rangle$$.


To generalize to other quantiles, one can use the property that a $$\gamma$$-quantile
minimizes

$$\underset{q}{\min}\sum_{i:x_i-q>0} \gamma(x_i- q) + \sum_{i:x_i-q<0}
(1-\gamma)(q-x_i)$$

which can be "softened" to

$$\underset{q}{\min}\sum_{i} \left(\softabs(x_i - q)\ \langle\softargmax(x_i, q) ,
\left(\gamma,  (1-\gamma)\right)\rangle\right)$$

and then proceed as before.

## The median using ranks

The first step is defining a form of soft ranking. We need a soft version of comparing a
pair of elements or, equivalently, a sign function, for instance:

$$\softsign(x) = 2* \arctan(\alpha x)/\pi$$.

This is also known as a sigmoid function, and here I picked one of the many options and
scaled it to the desired range. From this, I am going to calculate a sort of rank, named
$$\softmedianrank$$, which is translated so that the median of a sample has
$$\softmedianrank$$ 0 in it --- this makes the following definitions simpler:

$$\softmedianrank$$ $$(x_i, x)=\sum_j \softsign(x_i-x_j)$$.

The idea is taking a sum of all the signs (positive or negative) of all the differences
between one element and each of the others. For a median, the positive and negative
differences should balance out. By taking the square of this quantity, we have 0, the
$$\softmedianrank$$ of the median, be the smallest possible value. From there, we just
need to take the $$\softargmin$$:

$$\softargmedian(\x)=\underset{j}{\softargmin}(\softmedianrank(x_j, x)^2)$$.

Substituting the definition of $$\softmedianrank$$, we obtain:

$$\softargmedian(\x)=\underset{j}{\softargmin}\left(\left(\sum_i
\softsign(x_i-x_j)\right)^2\right)$$.

While we followed a different thought process, the result is not very different from the
one in the previous section. Unfortunately, that also implies we have made no progress
on the computational complexity side, which still is $$O(N^2)$$. One can also recognize
the similarity with the trivial parallel sorting algorithm that consists of computing
all the pairwise comparisons, organize them in a matrix and take the row sums to compute
the ranks. This suggests that either of the last two $$\softargmedian$$. definitions is
parallelizable along the same lines.

A definition for the $$\softmedian$$ can be obtained by following the same steps as in
the previous section. The extension to any quantile is possible by re-centering the
$$\softmedianrank$$ to be 0 for the desired quantile.


## Approximating the median

If looking at parallel algorithms could have provided some inspiration, why not look at
streaming algorithms as well? An approximate median algorithm for streaming settings is
the *remedian*, whose origins can be traced back to Tukey's *ninether*. The idea is to
split a vector into smaller ones and take the median of those and so on recursively,
until we are left with a single number. As to approximating other quantiles, the same
recursive approach to any quantile other than max and min results in biased estimates.
It seems that extreme quantiles admit specialized algorithms (top-k-type algorithms,
with k independent of sample size, unlike, say, a quartile). It's also possible to
reduce other quantile calculations to the median by padding. For instance, if we want to
calculate the first quartile of a sample of size N, we can pad the sample with $$a$$
elements smaller than the min, for a total of $$a+N$$ elements. The median of the padded
sample is larger than $$\frac{a+N}{2}$$ elements, of which $$a$$ are padding and
$$\frac{N-a}{2}$$ are from the original sample. For the latter number to match a first
quartile we just need to set $$a=\frac{N}{2}$$. More in general, for a quantile $$q$$
smaller than the median, the required padding is of size $$(1-2q)N$$. For quantiles
above the median, we can pad on the max side and follow the same calculation.  So let's
focus on the median. If we take the remedian algorithm and replace any calculation of
the median with the $$\softmedian$$, we have soft version of the remedian, or
$$\softremedian$$. As a bonus, we get a streaming algorithm and one with $$O(N)$$ time
complexity, and $$O(\log N)$$ parallel time complexity. The $$\softmedian$$ complexity
is still quadratic, but it is applied only to small input vectors, so it doesn't impact
the overall cost, in the asymptotic sense.

So now what about the $$\softargremedian$$? The two ingredients we need are a notion of
distance from the median and the $$\softargmin$$. I am sure the astute reader can take
it from here. And that's all for the theory. Stay tuned for some implementations in
future posts.

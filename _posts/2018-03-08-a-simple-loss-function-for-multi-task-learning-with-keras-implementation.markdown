---
layout: "post"
title: "A Simple Loss Function for Multi-Task learning with Keras implementation, part&nbsp;1"
date: "2018-03-08 11:51"
---

In this post I walk through a recent paper about multi-task learning and fill
in some mathematical details. Implementation and  experiments will follow in a later post.

<!--more-->



$$
\newcommand{\vx}{\mathbf{x}}
\newcommand{\vw}{\mathbf{w}}
$$

## Multi-task Learning

[Multi-task
learning](http://papers.nips.cc/paper/959-learning-many-related-tasks-at-the-same-time-with-backpropagation.pdf)
is generally defined as solving two or more ML problems in an integrated way
that results in synergies and better performance on each of the independent
tasks. For example, object recognition and depth perception in artificial
vision. Knowing their distance from an observer helps telling a mouse from an
elephant or a rook (a chess piece) from a medieval tower, for sure. There are
many reasons why this may be helpful and there are even more open questions, but
the approach has taken off and is widely adopted at least in AI research
circles. A nice [overview](http://ruder.io/multi-task/) is available.

One common variant of multi-task learning consists of co-training a single model
to perform several tasks at once. In statistical parlance, this would be
multivariate (multiple) regression. Since training is achieved in most cases by
optimizing a loss function, varying the free parameters of the model, one is
left with the problem of formulating a unified loss function that captures the
performance on all tasks simultaneously. Unfortunately, this is not so simple.
In the image processing example, the output of object recognition is typically a
label, whereas the output of depth perception is a depth map (pixel-by-pixel
depth prediction). The output is in completely different units and the
performance on each task is measured in ways that are not comparable.  How much
depth error is equivalent to one mislabeled example? Maybe one way would be to
let the application be our guide, and convert everything into a "dollar value".
But even if the different tasks consist of predictions in the same units, there
are statistical issues to consider. The tasks can differ in their complexity and
signal to noise ratio. Imagine predicting multiple stock prices, or the weather
at different locations. This can have a negative effect on the balancing act
between bias and variance that needs to be achieved for optimal performance.
There is a risk to overfit the noisier tasks in an attempt to reduce the error
while underfitting the less noisy ones.

## The Paper

This problem and a possible solution, with an application to image processing,
are the subject of a recent and interesting paper: [Multi-Task Learning Using
Uncertainty to Weigh Losses for Scene Geometry and
Semantics](https://arxiv.org/abs/1705.07115). If I have a problem with that
paper, is that they use the word *homoscedastic* (constant variance) all over
the place when their main contribution is to recognize and take into account the
*heteroscedasticity* (variable variance) of multitask learning. If it's lucky to
have constant variance within a task, it would be downright miraculous to have
it across tasks. Therefore it is advisable to model the heterogeneity of
different tasks with the addition of per-task variance parameter.

Another slight disappointment is that this paper doesn't explain in full detail
how it goes about estimating those per-task variances. Given a well-behaved loss
function, which they derive in detail, it's easy to imagine that they use SGD or
equivalent to simultaneously fit the weights of a neural network and the
per-task variances.  But how this is accomplished is not clear, nor an
implementation is provided &mdash; email to the first author has not been
returned yet. One would think that in this day and age of open science and
reproducibility crisis we should do better than that. While trying to fill in
the missing details, I derived a closed form solution to the problem of per-task
variance fitting, which I present below. In a follow up post, I will
provide a commented example and Keras implementation of the loss function thus
derived.


## Mathematical Derivation of the Loss function

We largely follow the notation in the aforementioned paper. Let $$i$$ index the
training set and $$j$$ the dependent variables (the "tasks" in multi-task
learning). Under the assumptions that the such variables, with realizations $$y_{ij}$$, are independent, conditional to the prediction returned by a model $$f$$, with adjustable parameters $$\vw$$, on input $$\vx_i$$; and that the
error is normally distributed and zero-mean, with variance $$\sigma_j^2$$, which
depends only on $$j$$ (the task), we can write the log-likelihood function as
follows:

$$\sum_{ij}\log\left(\frac{1}{\sqrt{2\pi}\sigma_j}
\exp\left(-\frac{\left(y_{ji}-f_j(\vx_i;\vw)\right)^2}{2\sigma_j^2}\right)\right)$$

By the basic properties of the $$\log$$ function this can be rewritten as:

$$\sum_{ij}\left(-\log(\sqrt{2\pi}) -\frac{1}{2}\log\sigma_j^2 -
\frac{\left(y_{ji}-f_j(\vx_i;\vw)\right)^2}{2\sigma_j^2}\right)
$$

Looking for stationary points of this loss w.r.t. the variance $$\sigma_j^2$$,
considered a variable with slight abuse of notation &mdash; the substitution is
immaterial to finding a minimum &mdash; and, dropping a constant additive term,
we have:

$$\frac{\partial}{\partial\sigma_j^2}\sum_i\left( -\frac{1}{2}\log\sigma_j^2 -
\frac{\left(y_{ji}-f_j(\vx_i;\vw)\right)^2}{2\sigma_j^2}\right) = 0$$

Applying the linearity of partial derivatives, and calculating the derivatives
for each term we have:

$$\sum_i\left(-\frac{1}{2\sigma_j^2}+
  \frac{\left(y_{ji}-f_j(\vx_i;\vw)\right)^2}{2\sigma_j^4}\right) = 0$$

The first term is independent of $$i$$ so it can be extracted from the summation:

$$-\frac{N}{2\sigma_j^2}+\sum_i
  \frac{\left(y_{ji}-f_j(\vx_i;\vw)\right)^2}{2\sigma_j^4}= 0$$

where $$N$$ is the size of the training set.
We can simplify a common $$1/2\sigma_j^2$$ factor:

$$-N+\sum_i
    \frac{\left(y_{ji}-f_j(\vx_i;\vw)\right)^2}{\sigma_j^2} = 0$$

And, finally, solving for $$\sigma_j^2$$, we have:

$$\sigma_j^2 = \frac{1}{N}\sum_i\left(y_{ji}-f_j(\vx_i;\vw)\right)^2$$

We can now substitute this into the likelihood, neglecting a constant term:

$$\sum_{ij}\left( -\frac{1}{2}\log\left(\frac{1}{N}\sum_i
    \left(y_{ji}-f_j(\vx_i;\vw)\right)^2\right)-
\frac{\left(y_{ji}-f_j(\vx_i;\vw)\right)^2}{\frac{2}{N}\sum_i
    \left(y_{ji}-f_j(\vx_i;\vw)\right)^2}\right)$$

We observe the first term is independent of $$i$$ and can be thus pulled out of
 the summation over that variable:

$$\sum_j\left(-\frac{N}{2}\log\left(\frac{1}{N}\sum_i
    \left(y_{ji}-f_j(\vx_i;\vw)\right)^2\right)-
\frac{\sum_i\left(y_{ji}-f_j(\vx_i;\vw)\right)^2}{\frac{2}{N}\sum_i
    \left(y_{ji}-f_j(\vx_i;\vw)\right)^2}\right)$$

The second term allows some drastic simplification:

$$ \sum_j\left(-\frac{N}{2}\log\left(\frac{1}{N}\sum_i
    \left(y_{ji}-f_j(\vx_i;\vw)\right)^2\right)-\frac{N}{2}\right)$$

Then by dropping constant additive and multiplicative terms:

$$ -\sum_j\log\left(\sum_i
    \left(y_{ji}-f_j(\vx_i;\vw)\right)^2\right)$$

To obtain a loss function, we need to flip the sign and, optionally,
exponentiate to return to the original scale of the quadratic loss:

$$\prod_j\left(\sum_i
    \left(y_{ji}-f_j(\vx_i;\vw)\right)^2\right)$$



Now, discounting the possibility of errors, this is an interesting result: it
says that when variances are unknown we can't average the losses among different
tasks, not matter how weighted, which would be possible in the case of known
variances; we have instead to switch to a geometric average of the losses.
Otherwise stated, it says that the doubling of the loss on one task cancels out
the halving of it on another one, which would not be the case when variances are
known. It's quite neat, and I wouldn't be surprised if it had been observed
before in a different context.

This result has a very practical consequence: we do not need to fit the
variances during the training of the neural network. Leveraging the  closed form
solution we can train in the multitask case with no increase in complexity and
by simply implementing this new loss function. And that's exactly what we are
going to do in the next post (link will appear in the footer on the right), together with a couple of experiments. Stay tuned.

Update: Thanks to João Paulo Lima for pointing out an error in the math. Luckily it doesn't affect the final expression. It has now been corrected in the main text.

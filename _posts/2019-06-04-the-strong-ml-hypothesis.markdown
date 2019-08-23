---
layout: post
title: "The Strong ML Hypothesis"
date: "2019-06-04 12:09:00 -0700"
---

Data and compute power availability are important in the resurgence of ML and AI, but two of the biggest innovations in neural networks (NN), convolutional and deep networks (CN and DN), are data- and compute-efficient ideas, which allow practitioners to do more with fewer resources. I think this observation deserves more attention.

<!--more-->

Recently, a post entitled "Bitter Lesson" by Rich Sutton, a founding father of the field, was making the rounds of the twitter-verse:

> The biggest lesson that can be read from 70 years of AI research is that general **methods that leverage computation are ultimately the most effective**, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation.

The emphasis on "methods that leverage computation" has morphed into a belief that **data and computation are the only thing that matters**.
At the recent scaled ML meeting, Ilya Sutskever, fresh from the success of the GPT-2
text model, declared that, while this model [doesn't pass yet the Turing
test](https://youtu.be/T0I88NhR_9M?list=PLRM2gQVaW_wWXoUnSfZTxpgDmNaAS1RtG&t=1785), it may well do so with additional scaling. GPT-2 is remarkable in that it uses two well-known architectural ideas, deep networks and *attention*, and scales them to some of the largest training sets and network sizes ever attempted in NLP. No new ML ideas were needed to build GPT-2; the challenging part was the underlying infrastructure to scale to its size.

Dr. Sutskever's position is not new. Top Google researchers
[claim](https://research.google.com/pubs/archive/35179.pdf) "Invariably, simple models
and a lot of data trump more elaborate models based on less data". I could trace back
this position all the way to [2001](www.aclweb.org/anthology/P01-1005). If the original
strong AI hypothesis stated that all that mattered was the input-output relation
implemented by an AI system, its modern, ML-based version, which I hereby name the
**strong ML hypothesis**, is that progress in **ML systems should be judged exclusively
on their generalization performance**. Since scaling improves generalization, scaling is
what we need to pursue. [Listen](https://youtu.be/T0I88NhR_9M?t=535) to Dr. Sutskever
formulate a NLP-specific version in no uncertain terms, asserting that excellent text
prediction (a.k.a. test set performance) can only be explained by intelligence.

The original strong AI hypothesis placed no restriction on the system or algorithm
implementing the AI. If it has to come down to a hash table with all possible I/O pairs,
so be it --- notwithstanding the practical impossibility of doing so. The strong ML
version of this is, if my interpretation is correct, that any learning model is
equivalent as long as its generalization performance is the same.

**I don't think the strong ML hypothesis will stand the test of time** and in the
following I will present four different, albeit related, arguments to support this
prediction. While they don't constitute a proof, I hope you will find them compelling
and worth considering. I will also add two reasons why we shouldn't be putting faith in
more scaling, beyond the short term, even if the strong ML hypothesis turned out to be
correct.
<!-- TOC -->

- [Data Efficiency and model size](#data-efficiency-and-model-size)
- [Performance outside the test set](#performance-outside-the-test-set)
- [Architecture matters](#architecture-matters)
- [Misspecified models](#misspecified-models)
  - [If scale is all that matters, we are in trouble](#if-scale-is-all-that-matters-we-are-in-trouble)
- [Scaling is getting harder](#scaling-is-getting-harder)
- [Conclusion](#conclusion)

<!-- /TOC -->

## Data Efficiency and model size

If bigger models trained on bigger data sets are better, nothing is bigger than a hash
table with all possible I/O pairs. For instance, a table of all pairs of English and
corresponding Spanish sentences up to a maximum length. You may object that it wouldn't
be actually generalizing because it contains all possible I/O pairs; there is no data
left for generalization. So let's move away gradually from the hash table, as a thought
experiment. What if one stored only half of those sentences and translated the rest by
looking for the closest match? If that system achieved excellent test set performance,
would it be displaying intelligence? Some people would say that it is rote memorization and therefore not intelligence. Others would claim that the intelligence is in not having to store half of the sentences.

Now compare that with NNs as used in tackling modern AI benchmarks. Is a NN like GPT-2,
with a billon adjustable parameters used to learn 40 billion bytes of text --- similar
in size to its training set --- more like a brain or a hash table? Award-winning deep
network architectures have been [shown](https://arxiv.org/abs/1611.03530) to be able to
learn random data (in training only, of course), both in the sense of randomly generated
or with randomly shuffled labels. They are equally capable to learn nonsense as they are
real data, like hash tables.

Writing in the magazine Wired, Gary Marcus, a virulent critic of machine learning in general, [trains its criticism](https://www.wired.com/story/deepminds-losses-future-artificial-intelligence/amp) on deep reinforcement learning, but there is nothing specific to the "reinforcement" part in it:

> In some ways, deep reinforcement learning is a kind of turbocharged memorization; systems that use it are capable of awesome feats, but they have only a shallow understanding of what they are doing

If I may make a comparison with my students of a past experience as a lecturer, the
average ones needed lots of examples, and could only solve problems similar to one of
the examples. My best students needed fewer examples and could generalize to much more
original problems. They seemed much more data-efficient and able to extrapolate. They
did better with less data. They didn't scale with more data. Most of them worked off the same book, the same lessons.

This is to suggest that we should incorporate model size and data efficiency in the
evaluation of models. Right now, state-of-the-art is determined purely in terms of test
set performance. But there are encouraging signs that at least the Reinforcement
Learning community is realizing they can not just scale the field forward (they call it
*sample efficiency*).


## Performance outside the test set

Like students, ML models live and die by their test performance. But there are signs
that it isn't enough, that test performance doesn't capture important aspects of
learning, for students as well machines. Here are four signs that this is the case for
machines:

1.  Researchers discovered *[adversarial examples](https://arxiv.org/abs/1905.02175)*, generated examples that fall just
outside the training or test sets by some metric. There is a consensus that they are not
just oddities but that they reveal something important about the brittleness of a model and its lack of robustness. They also represent a concrete barrier to the deployment of
ML systems when malicious input is a possibility. Yet, as far as test set performance is
concerned, they don't matter. Same goes with fuzz or nonsense examples. Absent from
the training set, one would like the system to throw up its hands in those cases, not
make a random guess.
1.  In some applications, models are being trained preferentially on *hard examples*,
such as cases when an autonomous vehicle has to be taken over by a safety driver. This
means that the training set distribution is diverging from that of the test set, downplaying test set performance.
1.  In other applications, models are expected to achieve near-0 error (e.g. autonomous
driving). That is more akin to worst case (CS style) than statistical analysis and requires ever growing test sets just to evaluate, let alone train.
1.  Practitioners monitor the performance of ML systems over time, in case shifts in the
environment compromise performance. Yet test set performance doesn't say much about a
system's resilience to such shifts. An initially high performing system may need more frequent refreshes than a more mediocre one.
1.  Some high-performing models have been [shown](https://arxiv.org/abs/1905.02175) to use brittle features, e.g. using the
background to decide if a bear is polar, or skin color to detect basketball players,  Larry Bird notwithstanding. That fueled doubts as to whether the model really "understands"
what a polar bear or a basketball player is or just has a way to detect them that
doesn't involve understanding, a question that is not helpful until we have a better definition of it. Yet, I've never seen non-white curling players, but I would
be able to recognize them (update: I found a picture of a [Japanese team](https://upload.wikimedia.org/wikipedia/commons/4/49/Team_Aomori_2006.jpg)).
These observations point to limitations of test set performance as the only criterion
for ML system quality. Teachers knew all along that studying for the test is never a
long term learning strategy. OpenAI just [released](https://openai.com/blog/testing-robustness/) research on a general metric to quantify model resilience to adversarial attacks, a step in the right direction.

# Architecture matters

If more data and more compute were all that mattered, then single-hidden-layer networks
should suffice because of universal approximation theorems and because they are easier
to train --- they are not prone to *gradient explosion and vanishing*. Just
expanding the hidden layer can absorb all the data and all the compute power that is
available. There is a variety of universal approximation theorem to support that.
Instead deep networks and convolutional networks and other advanced models took over.
Deep nets can simulate shallow networks with the same number of adjustable parameters,
but not the other way around. A standard example is parity, but there are also general
theorems about this. If the same amount of data and computing resources had been spent
on shallow architectures, few people would disagree the field wouldn't be where it is
now. Another milestone in AI progress were convolutional networks, which are also
parsimonious in the number of adjustable parameters, compared to dense alternatives.
Research centers spend quite some effort on architecture and algorithms, and AI
competitions often center around a fixed data set, making architecture the only degree
of freedom. The aforementioned Ilya Sutskever, when introducing a learning architecture
called *attention*, mentioned LSTMs, deep networks and convolutional networks as the
three most noteworthy architecture advancements in neural nets. Dr. Sutskever is a
staunch supporter of all-out scaling. Nonetheless, he highlighted attention as essential
to the performance of GPT-2. I am looking forward to the next major data-efficient
architectural idea.



# Misspecified models

When Galileo set out to study the laws of dynamics,  he performed a number of
experiments, and proposed a linear relationship between force and acceleration that
closely fit the data. If Galileo had been an ML algorithm, say a random forest, things
could have played out very differently. A random forest is an ensemble of decision
trees: each node in a tree holds a variable name and a threshold, each leaf a constant.
No matter how many or how large trees one generates, the learned function is a stepwise
constant function -- instead of the linear relation that Galileo found. With more data,
ML-Galileo could have generated an ever finer staircase that followed the data ever more
closely, but would have still been unable to discover the linear relation or to
extrapolate outside the range of available data. It is true that an ensemble of very
elementary learning models can reach arbitrarily good test set performance, but the
resulting model can still be very different from the underlying ground truth --- and if
that doesn't concern you at an aesthetic or epistemological level, extrapolation also seems to take a toll.
Test set performance doesn't quite measure whether the model is the correct
one -- in statistical parlance, if the model was *well specified*. Many have said that
intelligence likely doesn't admit simple models like some physical systems do but, among
imperfect models, some may be more misspecified than others.

## If scale is all that matters, we are in trouble

In this and the next section I will consider a consequence of the strong ML hypothesis,
which is that scaling is a valid way to achieve intelligence. The problem is that we may have already been through the best part of the scaling phase. This doesn't mean the strong ML hypothesis is wrong, my main contention, but that it is unworkable as a way to achieve the next level in AI.

Consider again GPT-2, with all of its billion adjustable parameters, and a human brain.
It's hard to compare artificial neurons and biological ones, with some people
emphasizing the complexity of a single neuron and others claiming that the noisiness of
biological information processing reduces the effective precision of the calculations. I
will cut the argument in the middle, aware that this is an unsettled question, and
consider one synapsis equivalent to one adjustable weight. The brain has about half a
million times more synapses than GPT-2 has adjustable weights. When people propose
current NNs actually display intelligence or are on the verge thereof, it must mean that
this complexity gap has been bridged somehow, or that a very specialized,
domain-specific version of intelligence is possible. For instance, it must mean that
when I am driving I am using only a minuscule portion of my brain, its processing power
or its memory capacity, hence a NN of comparably small complexity can achieve autonomous
driving. Alternately, it must mean that artificial NNs, maybe with further architecture
improvements, will become remarkably more powerful than their biological  brethren for a
given complexity. Finally, it could mean that a 5-6 order of magnitude increase in
processing power will become  available to NNs in the near future, which I will touch
upon in the next section. Of all of these, I think that the identification of
specialized tasks is the only credible one, with precedents such as speech and object
recognition. Andrew Ng tentatively identified "reactive" tasks, that take less than a
few seconds for a human to accomplish, as reasonable targets for current NN technology.
The problem with modularizing intelligence is that it's never clear where one task
starts and another ends: parsing English requires physical world knowledge ("Can't fit
the guitar into the suitcase. It is too {big|small}" -- what does "it" refer to?),
driving requires answering ethical questions ("Should I hit the cyclist or swerve into a
ditch?"). The brain shows modularity, but it doesn't mean each module can work in complete isolation.


# Scaling is getting harder

Scaling is based on decades-long, relentless hardware progress. I lived through most of
my life and career under the relentless driving force that hardware improvement was, and
it is hard to adapt to the new reality that we are approaching its limits. *Dennard
scaling*, the predictable parallel improvement of several qualities of computer
hardware, has stopped working years ago and *Moore's law*, about the miniaturization of
circuits,  seems to be reaching its limits. To increase performance, one approach is
designing task-specific processors, which have already been exploited for ML by first
co-opting GPUs and then creating TPUs, processors expressly designed to run NNs. As Bill
Dally of Nvidia said: ["Eventually we'll reach the point of diminishing returns [from
hardware architecture
innovation]"](https://youtu.be/EkHnyuW_U7o?list=PLRM2gQVaW_wWXoUnSfZTxpgDmNaAS1RtG&t=2823).
He then reassured the shaken audience that they had enough ideas in the pipeline to
produce a new generation of GPUs --- just one. So why not horizontal scaling? We've done
quite a bit of that too: GPT-2 was trained on [100 Volta
GPUs](https://youtu.be/T0I88NhR_9M?t=986), which make for a $1M, 30kW system, ignoring
the PCs they are hosted by and the necessary networking. The training took a week and
could cost around $50K at current AWS EC2 prices (others quote $43K and report OpenAI
used Google's TPU v3). Sure one can go bigger than that: there are supercomputers that
use 300 times the power. While neural networks are amenable to parallelization, one
needs to upgrade the weights at regular intervals, which results in network overhead and
Amdahl's law capping performance gains. A few research centers and tech behemoths will
be able to go up to supercomputer size in their experiments, and maybe they are already,
but there's a limit to what can be deployed more generally in the industry and on
limited power devices. Barring radical, unpredictable developments in hardware, scaling won't dominate the next phase of AI, which will make algorithms and
architectures even more important.



# Conclusion

Scaling AI models has resulted in remarkable progress, which has led to what I dubbed
the *strong ML hypothesis*, which calls for improving test set performance by all
available means over any other metric. On the other hand, there are multiple indicators
that model size, data efficiency, resilience to environment changes and adversarial
examples, extrapolation performance and, more subjectively, "quality" are also
important. In addition, comparisons with biological brains and hardware trends place
limits on how far this scaling can go. The deep message underlying this may be that empirical risk minimization, of which test error is a simple estimate, needs to be replaced by some other measure that captures more closely what we mean by "learning". While some steps have been taken --- measuring data efficiency and resilience to adversarial examples --- they are not integrated into one measure of learning performance and they don't enjoy widespread adoption yet. A slow down in scaling may just be the thing that focuses the community on these issues.

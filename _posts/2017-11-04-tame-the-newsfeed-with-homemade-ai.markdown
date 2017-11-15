---
layout: "post"
title: "Tame the newsfeed with homemade AI"
date: "2017-11-04 14:09"
---


Back in the 60s it was called [information
overload](https://books.google.com/books?id=boSFAAAAMAAJ&cd=1&dq=%22the+managing+of+organizations%22&q=information+overload#search_anchor)
and it affected so-called *decision makers*. Fast forward to today and the
situation hasn't improved.

<!-- more info -->

Now it's not only decision makers who are
overloaded: it's everyone. The information is not just too much: it's trivial or
factually incorrect or deliberately crafted to distort our beliefs, influence
our actions and manipulate our feelings to the benefit of others. Check out the
tamely named but strongly worded [timewellspent.io](http://timewellspent.io) and
the strongly named and worded [deathtobullshit.com](http://deathtobullshit.com)
to read more about the Faustian bargain whereby we access free services in
exchange for being manipulated.


## The solutions

Some people propose internet-free days, a modern day version of sober nights,
and we know how those work. Others propose to boycott the economics of this
system: pay for your services, reject the ad-supported, conflict-of-interest
ridden model. Another remedy, and the main subject of this post, is to regain
control of the context, pace and selection of the information we consume. To
that end, we need to get out of the apps designed for the benefit of the
corporations that develop them and use apps that are on our side. Apparently,
there isn't a market for that, so I decided to build my own.

## Feeds

The first step is to do my reading in a feed reader. Remember
[feeds](https://en.wikipedia.org/wiki/Web_feed)? I am talking about RSS and Atom
feeds, the open formats that web sites use to broadcast updates and news, a
model later adopted, in a proprietary, siloed version, by most social networks.
The popular press has decreed that Facebook and Twitter [killed
RSS](http://mashable.com/2010/09/11/bloglines-discontinued/#iMaXBuYAoSq2). In
reality, you'd be surprised how many sites still support it, or its cousin Atom:
this [article](http://www.makeuseof.com/tag/rss-dead-look-numbers/) provides
some numbers. In my personal experience, almost everything I need is available
as a feed, with one glaring exception: twitter. Unfortunately many writers,
including technologists that should know better, have chosen to use Twitter as
their only update channel, and twitter withdrew their support for RSS half a
decade ago. To the rescue comes a third party service,
[twitrss.me](http://twitrss.me), which provides RSS feeds for users and searches
on Twitter. Well done guys! Now that we have feeds, we need a reader. I use
[Vienna](https://github.com/ViennaRSS/vienna-rss), an open source software, but
any will do. [Newsblur](http://newsblur.com) is also a great choice.


## The AI proxy

The next step is to control the quantity and selection of the information
traveling on these feeds.  Some feeds are very noisy; some twitter users even
worse. But one still has to follow them: I really, really need to hear the
latest about that amazing neural net model, but I really, really don't need to
know what the researcher ate for breakfast before working on said  model; or
should I try eating the same stuff? I love the New York Times, but their *top
news* feed is contaminated with local New York news, in contrast to their
declared global mission. Yet other feeds mix in sponsored content and other
poisons. How can we get access to most of what we need while cutting out most of
the crud? How do we pick the signal from the noise? I started thinking about a
solution some 10 years ago: a filtering or ranking proxy which reads the
original feeds on one side, analyzes them, and delivers them cleaned up and
otherwise enhanced on the other. To make an example, if you wanted to subscribe
to `http://example.com/somefeed.xml`, you'd point your feed reader instead to
`http://friendly.ai/feed/http://example.com/somefeed.xml`. The software powering
`friendly.ai` would perform the magic of sifting through the content on  your
behalf and surface the content you are most likely to be interested in. The
magic in question would be implemented using machine learning.


## Machine Learning take #1

My first take on this was based on a
[bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) feature
extraction method, which equates a document with the multiset of its words.
"This story talks about smoke, not fire" and "This story talks about fire, not
smoke" map to exactly the same feature vector, suggesting this method has some
limitations, but it has been applied  successfully to a number of Natural
Language Processing tasks. Since a feature vector in the bag of words approach
has the size of the vocabulary covered by the documents being analyzed, a
kernelized, regularized method, such as SVMs, is a top candidate to perform
Machine Learning on bag of words. The feedback or supervision was provided by
the user itself through links embedded in each feed entry. The system worked,
but not well and not data-efficiently enough. One had to provide feedback
thousands of times to start to get some benefit out of it. And rich vocabularies
were hard to generalize over. It's amazing how many ways the New York Times has
to refer to local news, and this system had to learn them one by one. Even a
very motivated user such as myself found the task  of teaching it frustrating.
My first attempt kind of fizzled out.



## Machine Learning take #2

Fast forward 10 years and the state of the art has drastically moved. We have
[word vector mappings](https://nlp.stanford.edu/pubs/glove.pdf) that map words
to vectors with interesting semantic properties and significantly reduce the
difficulty of NLP problems. When Facebook's AI research arm, FAIR, made
available not just a word vector mapping but a [sentence vector
mapping](https://github.com/facebookresearch/InferSent), I knew I wanted to take
another stab at this project. A sentence vector mapping is a function that maps
any sentence to a vector, in such a way that the vector alone is sufficient to
solve a variety of semantic tasks related to  the original sentence. FAIR
[demonstrated](https://github.com/facebookresearch/SentEval) that these vectors
could be used as features or input to simple ML models, namely logistic
regression, to solve a variety of interesting ML problems. Therefore my new
system is comprised of the following components:

*   content extraction, since feeds are often abridged versions of the original
    content or event just links, using [boilerpipe](https://github.com/kohlschutter/boilerpipe)
*   sentence splitting, using [NLTK](http://www.nltk.org/)
*   sentence-vector mapping, using the aforementioned FAIR project,  [InferSent](https://github.com/facebookresearch/InferSent)
*   a classification model, linear regression, with those vectors as input
    and previously gathered user feedback as output, using [Sklearn](http://scikit-learn.org/stable/)
*   a web proxy to fetch feeds, process them and serve them to the clients
   enriched with a predicted interestingness score per each entry and UI
   elements for the user to provide feedback, implemented as a [Flask](http://flask.pocoo.org/) app and with [feedcache](https://pypi.python.org/pypi/feedcache/1.4.1) to retrieve feeds.
*   a subsystem to collect and store such feedback also based on Flask and
    [SQlite](https://www.sqlite.org/)

The implementation, in Python, is
[available](https://github.com/piccolbo/rightload) on github.

One important detail is that the feedback is provided at the entry or article
level, whereas the classification happens at the sentence level. The sentence
scores are aggregated to obtain an entry score by the simplest of methods, an
average. It is possible to imagine much more sophisticated algorithms whereby
the sentence vectors are fed one by one to a LSTM NN or equivalent. But we need
to walk before we can run.

## The app

Here is how an article index looks like in Vienna:

![Index pane.](/assets/Screen_Shot_2017-11-08_at_7.14.34_AM.png)

The titles are prefixed with a score which is 100 times the class probability
according to the logistic regression. This way it's easy to sort by descending
score, and read the good stuff first, or exclusively.

Opening an article, this is what you see:

![Article pane.](/assets/Screen Shot 2017-11-08 at 7.18.30 AM.png)

From the top, you see the title of the article, then the feed's name and the
date it was published. The grey bar that follows is the UI element inserted by
the proxy. The UI contains only one of two options, either *Time Wasted* or
*Time Well Spent*, names inspired by
[timewellspent.io](http://timewellspent.io). Only one of the two is available:
if the model got it right, based on a score threshold of 50, the feedback is
implicit and there is no need to take further action. I thought that if the ML
is successful this approach would put the lowest burden on the user. Below you
see a presentation of the article that is optimized for development &mdash; call
it a debug version. The HTML has been stripped and the raw text is highlighted
in pink with intensity proportional to the score. The numbers interleaved with
the sentences are class probabilities. An interesting problem that I am still
thinking about is how to do the highlighting directly on the HTML. Serving the
HTML without any highlighting would be the default option for a normal user. But
the highlighting not only helps with ML development, it also helps scanning long
form &mdash; a.k.a. low information density &mdash; articles for the most
important sentences.

Installing this app is somewhat involved, because Infersent forces you to go
through some hoops. It is also very slow and CPU-intensive and Vienna will
sometime timeout waiting for the proxy to respond. This is mitigated using
persistent caching. Infersent should really run on the GPU, which is in theory
supported but so far has proven an elusive goal. Making the proxy available on
line would face the same issues.

Thinking more long term, a multi-user version would allow to add elements of
collaborative filtering. The idea is applicable in other contexts. Imagine a
browser extension that scores pages before we read them, or highlights the most
interesting links in a page, based on where they land; or an email proxy that scores messages.

But in its current form, this system, named *[rightload](https://github.com/piccolbo/rightload)*, like its predecessor,
is already very useful to me. I hope some of you will find it useful as well,
and help me improve it. Or at least will be encouraged to take steps and regain
control of your time.

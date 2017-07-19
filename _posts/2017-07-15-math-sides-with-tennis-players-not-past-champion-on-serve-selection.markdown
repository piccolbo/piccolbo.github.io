---
layout: post
title: "Mathematical model sides with tennis players, not pundits, on serve selection"
date: "2017-07-15 20:02:52 -0700"
---

I was watching the current Wimbledon tennis tournament when I heard a comment by former champion and coach Boris Becker that got my attention. He complained that Canadian player Milos Raonic was not using the *body serve*, a shot aimed directly at the opponent that allegedly results in a weak return. Moreover, the BBC commentators found that to be true for his other matches this year, and a marked departure from his tactics during the previous edition of the tournament. Later on, they had a graphics showing how Swiss uber-champion, arguably greatest of all time Roger Federer placed his serve. The goal was to highlight his great accuracy, but it didn't escape my eye that he was using the body serve only very occasionally (throughout this article we are talking about the first serve only). See a similar [graphics](https://i.imgur.com/ly65BUH.png) from a different event a few years ago. The commentators did not remember their previous observation about Raonic or decided not to bring it up, as criticizing Federer requires a much higher standard of evidence than is necessary for Raonic, given their different statuses. That left me with a question. Is it possible that a tactics that was advantageous in the times of Boris Becker, the 80s, and still taught at all levels is no longer a good idea? What could have changed? The truth is that everything has changed since the 80s: serves are faster, players are stronger etc. But no change jumps out that seems to specifically blunt the body serve ... until I thought of *returner placement*. In the old days, most returners picked a position and waited there. Moving around was considered bad manners, a sneaky yet legal way to interfere with the server. Better players would adjust their position slightly over the length of a match, but wait in position after the server started his preparation.  In the last decade, it seems players have thrown this dogma out of the window, together with many others. I am not sure who started it, but current ATP tour number one Andy Murray seems to always [start behind](https://www.youtube.com/watch?v=HfN88NJ0j6c) where he wants to be and then take a *last minute* step or two forward, left or right or a combination thereof. Many players seem to take at least a last minute half step  left or right, without an apparent rationale. Watch [here](https://www.youtube.com/watch?v=a_hn4VFHz80) an impressive collage of returns by former number one Novak Djokovic, one of the best returners of his generation, using this technique. Guessing the direction of the serve from small changes in the ball toss and racquet angle may be part of it, but the best servers are [masters of disguise](https://youtu.be/wFm98S5IAFU?t=21). The "last minute" detail is essential. By waiting until the server has tossed the ball in the air, the returner prevents him from tailoring his serve direction to the position of the returner. In fact, a [slow motion of the Federer's serve](https://www.youtube.com/watch?v=FBkFgS3e4QY) shows that he keeps his eyes trained on his opponent halfway through his ball toss movement, with the ball already in the air. But he needs a final glance at the ball before hitting it; he is human after all. A last minute return position adjustment, for the vast majority of players, can not be observed in time by the server. What does that have to deal with the decline of the body serve? Let's build a simple model. Let's say that the returner steps left with probability \\(p_1\\), stays put with probability \\(p_2\\) and steps right with probability \\(p_3\\). Let's assume he is returning from the deuce side. The server has three basic options available: serving down the middle, at the body, or out wide. If he misses and hits in between these options, he's going to go straight at the forehand or backend of the opponent. This is usually frowned upon as the returner has his best chance to hit a highly competitive return in these two cases. But if the returner moves, then these options are scrambled. A step left means defusing the serve to the middle and the body serve, but a disadvantage in the other three cases. Likewise a step to the right defuses the serve out wide and the body serve, but concedes the other three options. So from left to right (or the reverse if talking about the ad side) let's say that a server strategy is to hit each of these directions with probabilities \\(q_0 \ldots q_4\\). Let's adopt an all or nothing model and write down the expected payoff for the returner as follows

$$
S = p_1(q_0+q_2) + p_2(q_1 + q_3) + p_3(q_2+q_4)
$$

The returner has observed the server's selections and wants to optimize his payoff. So he's going to formulate an optimization problem as follows:

$$
\max_{p_1,\, p_2,\, p_3} S
$$

subject to:

$$
\sum_i p_i = 1
$$


The probabilities \\(q_i\\) are considered fixed for now, but satisfy the same constraint:

$$
\sum_i q_i = 1
$$

The variables \\(p_i\\)  are bounded between 0 and 1 which would make this problem an inequality constrained one, but it's easy to show that if the returner gives up completely on one option, the server has a perfect strategy winning every time. So an optimal strategy for the returner must be bounded away from the boundaries of the feasible region. This observation allows us to apply a very simple technique, known as Lagrange multipliers. This consists of forming an expression called lagrangian:

$$
L = S + \lambda\left(\sum_i p_i - 1\right)
$$

As you can see it is a combination of the goal and the (only in this case) constraint weighed with an additional variable lambda. We now need to make the gradient of this function of \\(p_1 \ldots p_3\\) and \\(\lambda\\) equal to 0 to obtain a necessary condition for a stationary point of S in the feasible region. This gives, for each \\(i \in {1,2,3}\\)

$$
q_{i-1} + q_{i+1} = \lambda
$$

Let's stop to understand these equations. They are telling us that there is no stationary point if the server does not "balance" his serve strategy so that each of these three pairs of options is selected with probability \\(\lambda\\). If he fails to do so, the returner will abandon one of the three available options, focusing all his efforts on the other two, with an improved payoff. Therefore the server has to "balance" his strategy. The acute reader may have already noticed that \\(q_2\\), the probability of the body serve, plays a special role in the above equations: it's the only variable to appear twice! We may have our clue to why the body serve has fallen out of fashion. Let's take the sum of the three equations above and remember that

$$
\sum_i q_i = 1
$$

to obtain

$$
q_2 = 3\lambda -1
$$

On the other hand, if we rewrite the return goal S in terms of \\(\lambda \\), we get \\(S = \lambda\\). Therefore by substitution

$$
q_2 = 3S-1
$$

The returner payoff is proportional to the probability of the body serve. No surprise it's fallen out of fashion! Since \\(q_2\\) can't be negative the mathematically-minded server will give up the body serve to limit the returner payoff to 1/3. That may not sound so great, but on the first serve good servers score routinely 70% of their points, up to 90% and, in exceptional cases on fast surfaces, even 100% for the duration of a set (at least three serving rounds or 12 serves). Additional calculations show that the complete solution for the server satisfies:

$$
\begin{eqnarray}
q_0 = q_4 &=& 1/3\\
q_1 + q_3 &=& 1/3\\
q_2 &=& 0
\end{eqnarray}
$$

Which seems to fit what we see on the courts: good servers go mostly for the corners. But what about the returner? Well, once the server has "balanced" his options, it doesn't really matter what he does, as each choice has an expected payoff \\(\lambda\\). If the server has not, for instance setting \\(q_0 + q_2 = \lambda - \epsilon\\), then setting \\(p_1 = 0\\) would maximize the returner's payoff, raising it to \\(\lambda + \epsilon/2\\). Not a good deal for the server.

Is this model too draconian? It can't possibly be that servers always hit the spots they want, or that returners nail a return that comes straight to their forehand. That's a valid criticism, but we can generalize the model introducing two parameters, \\(\alpha\\) and \\(\beta\\), each representing the probability of the returner winning the point when returning a shot at comfortable distance for the former, and out of his reach or at the body for the latter, with \\(\alpha > \beta\\). Our fist cut at this model corresponds to \\(\alpha =1\\) and \\(\beta = 0\\) The expected payoff for the returner gets a bit more complicated:

$$

S = \sum_i p_i \sum_j q_j \delta(i,j)

$$

where \\(\delta(i,j) = \alpha\\) if \\(i = j \pm 1\\), \\(\beta\\) otherwise.

The Lagrange multiplier equations now read, for each \\(i\\):

$$
\sum_j q_j \delta(i,j) = \lambda
$$

Again, we sum all these equations together, this time factoring out \\(\alpha\\) and \\(\beta\\) and using the fact that \\(\sum_j q_j = 1\\). We obtain:

$$
q_2  = \frac{3S - \alpha - 2\beta}{\alpha - \beta}
$$

Again, \\(q_2\\) is proportional to the returner payoff. No choice for the server but to give up on the body serve. In hindsight the intuition is simple: of the serve directions available, the body serve is the only one defused by two of the three available return positions. That's the reason why \\(q_2\\) plays a special role in the above equations. But this intuition is not readily accessible to somebody as knowledgeable about tennis as Boris Becker, despite the available strong evidence from multiple players. He's not alone. A google search for "is the body serve effective" returns pearls of wisdom such as "The Most Underused and Underrated Serve"; "The most effective and underutilized serve in doubles"; "Serve to the body for safety". While the model discussed here is highly simplified, it isn't immediately clear how additional realism would restore the body serve reputation. For instance, on slower surfaces a fourth option for the returner is to drop backwards, which also defuses the body serve by allowing more time for the returner to coordinate. It seems more likely that the best players are abandoning the body serve in response to a moving target, the returner. Much is said about the role of technology and physical preparation in driving changes in the game, but technical changes are highly underestimated in my opinion. Players are using grips my teachers would have banned as conducive to fractures, and switch them with great variety during play, hitting the ball at any height. Squash shots (extreme defensive slice hit with reverse-side grips), open stance backends, sliding on all surfaces, jump shots, 360s (returning to the central position after a corner shot by turning one's back to the net), playing way behind the baseline and dynamic return positions, the list could go on and on. Players are not leaving any stone unturned in their quest for tennis glory. May mathematical models and data guide their efforts.

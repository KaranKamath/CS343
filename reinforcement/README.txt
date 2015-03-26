Karan Kamath
kpk462

Question 1
Implemented as per Value Iteration Bellman Update.

For each iteration, states that are terminal are considered to have 0 future rewards.

Methods are implemented generically, with extensibility in mind.

Question 2
Noise is reduced from 20% to 1.5%. This works because the agent needs to reach the reward at
the far end at least a few times for its utility to propagate, and with a noise of 20%, it is
unlikely to do so, as along the bridge, it has a 50% chance of falling into the pit for a random
move, and it needs to cut across 5 steps without this random move.

Question 3
a> The agent can be trained to prefer the close exit risking the cliff by setting the living reward to -2, keeping discount at 0.9 and noise at 0.2.
This works as a high regret for living leads to a rush to the nearest exit, overcoming the fear of the cliff.

b> This can be done by setting the discount to 0.5 as now the expected future rewards are not as significant as earlier, leading to a strategy that prioritizes
reaching the exit avoiding the danger that is closer (as this has a greater effect on the current utility than the exit which is further away. Regret at -2
ensures the closer exit is chosen.

c> Setting the regret to 1.0 helps choose the further exit as it is not as high as to prefer the closer exit, and still ensures the cliff is risked.

d> Setting no regret helps ensure the distant exit is chosen on a safer route, as there is no hurry to end the episode.

e> Setting the reward to 1.0 helps avoid both exits as the agent benefits continuosly from every step it takes and has no motivation to exit.

Question 4
Implemented as per QLearning Bellman Update equation.

Question 5a
Implemented as per specification in question.

Question 6
This is not possible as the tradeoff between exploration and learning (used in exploitation) always favours the nearer exit given noise. This is because along the
bridge, the agent is at least half as likely to fall into the pit for any move (given noise / epsilon), and random moves cannot move it consistently to the right
for consecutive steps for it to discover the larger payoff enough.

Moreover, it learns to move to the closer exit (as it is much easier), and lowering the learning rate also makes it less risk averse.

Question 7
Generic implementation caused it to work with no change.

Question 8
Qlearning logic extended to support weight learning. Conceptually just applying the update to a vector rather than a single function.

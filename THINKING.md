# Episode 12
## 10 Aug, 2026
I see that the car does make some progress and it does reach somewhere. But it is not able to reach the destination. Only sometimes.

The jitterriness has reduced - I trained it on 4 action repeats - maybe that's why. But for watching it drive, I set action_repeat=1 and that smooths things out.

My hypothesis for it not reaching destination is this - it probably is just not getting enough rewards for completing the circuit. If it strolls around it gets 100s of points. If it reaches destination, the reward is 500. Maybe I should increase it to 10000 or somethingl like that.

====

The agent came back with a very good response - if I score it like 10k, near the success point it will lead to unstability of gradients and policy being noisy. We should not pull out a number from feeling, instead we should calculate what a good number looks like from gamma.

# Episode 14
## Aug 21, 2026

The car is still acting jittery. It is better than before but still there. We are doing both of the following which are supposed to address the jitteriness
1. Action repeat - with repeat = 4
2. Action low pass - with steer low pass of 0.3

I wonder if we should increase the repeat to 8 or 16. Where I am coming from is this - the wall clock frequency is 20 MhZ. A human's
reaction time is of the order of fraction of a second (say 0.5 seconds). So we should perhaps repeate the action at least 10 times
to get the "human level" effect.

=================
# Episode 12
## 10 Aug, 2026
I see that the car does make some progress and it does reach somewhere. But it is not able to reach the destination. Only sometimes.

The jitterriness has reduced - I trained it on 4 action repeats - maybe that's why. But for watching it drive, I set action_repeat=1 and that smooths things out.

My hypothesis for it not reaching destination is this - it probably is just not getting enough rewards for completing the circuit. If it strolls around it gets 100s of points. If it reaches destination, the reward is 500. Maybe I should increase it to 10000 or somethingl like that.

====

The agent came back with a very good response - if I score it like 10k, near the success point it will lead to unstability of gradients and policy being noisy. We should not pull out a number from feeling, instead we should calculate what a good number looks like from gamma. I also want the driver to drive a bit more so taht it sees enough successes.


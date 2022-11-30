# Coding stuff
+ Fix my bad env decisions:
	- Back to terminal nodes as actual nodes (for symmetry)
	- New: Swap rule implemented in env and in NN
	- New: Optionally keep maker+breaker graph. 
		* Disadvantage: move time * 1.5
		* Advantage: Able to always use maker graph as NN input (for both sides, simplifies task, even more symmetry).
		* Advantage: Alternatively able to combine information from maker and breaker graph for more rich representation.
+ GNN now with sum aggregation. Staying similar to GraphSage, but with sum aggregation. Let's hope this works.
+ Talked to Johannes. Playing multiple games in parallel during RL data generation is smart and something he wanted to implement, but hadn't had time for yet. Johannes used batch size 8 in his experiments.
	- I think efficiency gains could be huge ~ 15x
		* Multithreading ~12 threads could speed up cpu part up to 15x
		* Batch size 128 is much more efficient than batch size 8 + better gpu util because other thread provides batch before previous batch has been processes by GPU. 20x.
	- I'll try to implement this myself soon.

# Writing the thesis
+ As promised, I'll start writing the introduction in December.
+ I got a lot of ideas in mind, especially about making the motivation clear. (Should there be a *Motivation* subsection?)
+ I usually tend to run out of steem writing the introduction after a few pages/paragraphs.
	- Then I guess I'll need to do some more literature research.

# If we have time: How do I get a PHD in AI?
+ How did you get your position / when to worry about applying?
+ How is the time after finishing the Master?
	- Where you jobless for a while?
	- Is it bad to be jobless for a while?
	- Does it make sense to start another Master, if I don't find a PHD position immediately?

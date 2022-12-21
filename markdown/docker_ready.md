# 7x7 experiment
+ started before last week on my home computer and trained for 4 days.
+ End result: with MCTS plays close to perfect
	- Raw net is also not bad, but fails unexpectedly some simple positions that are fully solved by MCTS. Not sure why.
	- Not super happy that it took so long and still isn't that great raw on this fairly simple env 7x7, might need to work on training schedule.
		* Training schedule was generate 4000 games, then do 40 epochs training on last 12000 games + 3000 replay memory games.
		* Training took about as long as data generation here ~1h each.
		* Winrates against prev model where consistently high (~66%) 40 times
		* Maybe 40 data generation model updates are to few and I should more often run arena and update the model.
		* Maybe it overfits to beating the prev model and I should include more replay memory / reduce number of training epochs per iteration.
+ After reading the AlphaGo paper and the Alpha Go Zero paper, I just read the AlphaZero paper and they do not use arena evaluation at all and just continuously update their network. I think I should aim to be more online as well and not 1 nn data generator update every 2 hours.


# DGX
+ I managed to log into the dgx machines and uploaded a working docker image of the current version
+ Now contemplating what experiment to start now.
	- I think my GNN architecture / GNN inputs could be improved.
		* I've added layer norm for next experiment
		* I've thought about combining maker and breaker graph into a more rich input (currently only maker graph)
			+ This would mean two edge types
			+ I would not like to use edge attributes as this seems like overkill.
			+ Each step, do one propagation with breaker graph edges and one with maker graph edges and then compute mean (using different parameters for maker and breaker propagation)?
			+ Or propagate with maker graph first and then use output features of maker graph to propagate with breaker graph?
	- How big to go? 11x11 is the goal but training it could take pretty long and I might profit from figuring out my params with smaller sizes first.
	- How much compute should I reasonably use?
		+ On my bachelorthesis I did fine with only 1 GPU.
		+ However, training data generation should be easily parallelizable and 2 GPU / 20 cpu means half training time, 4 GPU / 40 cpu means a fourth etc.

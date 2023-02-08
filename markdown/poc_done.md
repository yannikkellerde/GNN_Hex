# Proof of concept for HexAra
![rr_table](/images/roundrobin_table.png)
+ Model trained to reproduce Mohex-1k beats Mohex-1k itself if plugged it into HexAra.
+ I can continue training this model with AlphaGo method by generating training data from HexAra.
	- This results in a significantly stronger model that, when used with HexAra, beats Mohex-1k convincingly.
	- I need a lot of training data to not overfit to the training set.
		* even with 10k games, training loss sinks while validation loss doesn't
		* Used 100k games of training data now. However, just generating them takes 1.5 days.
	- My dreams of beating top HexAI won't happen, but at least I have a working proof of concept.
+ Got a rock paper scissors outcome from roundrobin. This makes sense though, as the models trained from Mohex-1k are obviously more fit to positions that occur in play with Mohex.

# Misc
+ What has your experience with using dropout/weight decay to prevent overfitting been like?
	- I won't have time to do experiments on that during the thesis probably
	- However, asking because in my experience the only thing that ever worked was collecting more training data and dropout/weight decay did not really change anything.

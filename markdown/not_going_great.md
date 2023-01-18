# Q: How are the experiments going?
+ A: Not great.
+ Day: Friday, I spent the last 3 days trying to figure out why my model is behaving so weirdly.
	- Observation 1: My model is not learning what's in the training data.
	- Observation 2: When I transfer the model and current training data to my local machine and try to learn it in supervised fashion (not generating any new training data), it works as expected.
	- Observation 3: The **exact same model** gives vastly different outputs for the same inputs on my docker at the dgx and my local installation.
	- Action: I push my 30GB docker image from the DGX to dockerhub and pull it at my local machine.
	- Observation 4: Vastly different outputs on DGX vs my local machine for the exact same docker image.
+ Day: Monday, I did some more tests
	- This only happens if I load a torch\_script model in libtorch c++.
	- So either libtorch, torch\_scatter/torch\_sparse c++ implementation or CUDA itself is betraying me.
	- This is bad, but I don't have time to spend the whole next weeks trying to figure this out.
	- I'll switch focus to preparing the RainbowDQN results for the thesis for now

## RainbowDQN status
+ So I have this fairly good model I trained at the start of the thesis for 6 days and I just revived it
+ I do still have the logs of the run, but the elo evaluation during the training process is somewhat wrong
+ I created a new elo evaluation scheme that should make more sense:
	-  When a new agent is added, a roundrobin tournament is started including the new agent, a random agent with elo fixed a 0, my old model (from the run in the beginning of the semester) and up to 8 randomly chosen older checkpoints from the same run.
+ I modified (hopefully improved the model): Layer Norm and multiple aggregation types before an MLP to compute the value. (Duelling scheme, one value for position and then advantages per move)
+ I started new

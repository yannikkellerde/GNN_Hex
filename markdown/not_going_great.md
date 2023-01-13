# Q: How are the experiments going?
+ A: Not great.
+ Day: Friday, I spent the last 3 days trying to figure out why my model is behaving so weirdly.
	- Observation 1: My model is not learning what's in the training data.
	- Observation 2: When I transfer the model and current training data to my local machine and try to learn it in supervised fashion (not generating any new training data), it works.
	- Observation 3: The **exact same model** gives vastly different outputs for the same inputs on my docker at the dgx and my local installation.
	- Action: I push my 30GB docker image from the DGX to dockerhub and pull it at my local machine.
	- Observation 4: Vastly different outputs on DGX vs my local machine for the exact same docker image.
		+ Switching the GPU on the DGX does not change anything.
+ At this point, I give up for now.
	- I think I eliminated all possible factors except the type of GPU.
	- Getting different results depending on NVIDIA GeForce 1070 Ti vs Tesla V100 should not happen I think. That is what we have CUDA for.
	- I'll definitely want to come back to this soon, but I think I need a break from this crazy bug fixing
+ Next I'll focus on writing the thesis and I'll try to reproduce my old RainbowDQN results, so I have some results to write about in the paper already.



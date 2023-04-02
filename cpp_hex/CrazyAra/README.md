# HexAra

The HexAra implementation is too big to describe each file in detail

Instead I will highlight some differences to CrazyAra:

+ [nn\_api.cpp](nn_api.cpp) replaces the CrazyAra neural network api that does not work well with GNNs.
+ [rl/traindataexporter.cpp](rl/traindataexporter.cpp) was modified to export using torch instead of other formats.
+ [rl/selfplay.cpp](rl/selfplay.cpp), [agents/agent.cpp](agents/agent.cpp), [agents/mctsagent.cpp](agents/mctsagent.cpp) , [searchthread.cpp](searchthread.cpp) among other files where modified to allow running parallel games with multiprocessing to create larger input batches as neural network inputs during training data generation. This significantly decreases runtime per generated training sample
+ [main/playmode.cpp](main/playmode.cpp) was created to make it possible to play against agents in the terminal. Also used together with python api.
+ [main/convert\_moves\_to\_training\_data.cpp](main/convert_moves_to_training_data.cpp) can convert a sgf game record to imitation learining policy training data. Used for mohex imitation (Figure 5.7)


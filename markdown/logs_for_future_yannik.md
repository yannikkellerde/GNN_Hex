I want to do multi-gpu training using 3 gpu for data generation and 1 for training. I'll run them in parallel, (4 processes). AlphaZero does not check winrate against prev model before updating, so I also won't do that anymore. However, I still need a reliable way to estimate the model performance.

I thought about using information from maker and breaker graph together in the gnn, but I am not sure how. Maybe using edge attribs? However, no edge attrib architecture seems like what I need.

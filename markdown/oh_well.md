# Oh well, things are not looking so good
* Turns out that the main reason that my GNNs did better board size transfer was that my CNN design was bad.
    + In comparison to my FCN-S, their architecture is actually not that different. They go a little deeper and use skip connections. But that is basically it.
    + However, the *Gao* baseline pads the boards with red and blue stones indicating who's border it is.
    + Somewhat surprisingly, that makes a huge difference. Not sure why I didn't try that myself during my thesis. I remember thinking about it, but the CNN is able to differenciate between the borders even without the padding so I was to lazy to try.
    + Otherwise, using Unet for a board game, was probably a stupid idea.
* The long-range dependency advantage of GNN does seem to hold up even against the Gao architecture. However, a fair comparison controlling for training time and number of parameters is still TODO.
* Current evaluation was done for Q-learning. Imitation learning is still TODO.
* Board size transfer comparison up to very large board sizes (20) seems to suggest that there is still some GNN advantage for transfer to much larger boards. However, I was kind of shocked how quick and easy it is to train a strong CNN Hex agent with the Gao architecture.


I found [this](https://ieeexplore.ieee.org/abstract/document/10108022) very fresh (25 April 2023) publication that tried using GNN for a Risk-like board game. Their results seem very non-convincing. Just using existing GNN architectures for board games seems to be inherently problematic.

** I am worried that the current results are not enough for a publication at a major conference :(. Additonally, the [Risk like](https://ieeexplore.ieee.org/abstract/document/10108022) paper seems to suggest that typical GNN architectures for board games might be a dead end.**

## Where do we go from here?
* More extensive experiments with Gao architecture, supervised (imitation) learning, fair long range dependency test etc. Does Gao still overfit more that GNN on supervised learning?
* Still go for ICLR publication?
* Go back to the drawing board and look for better GNN architectures?
* Still finish the paper with comparison to gao and very fair (highlighting limitation of GNN) assesment of GNN performance and just go for arxiv publication?

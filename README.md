# GN0
## Literature
Graph nets: https://arxiv.org/pdf/1806.01261.pdf  
Go zero: https://www.nature.com/articles/nature24270  
Pytorch Geometric: https://arxiv.org/pdf/1903.02428.pdf  
GNs suck: https://arxiv.org/abs/2010.13993  
GCN: https://arxiv.org/pdf/1609.02907.pdf  
Hex Graphs: https://webdocs.cs.ualberta.ca/~hayward/theses/ph.pdf
Hex Variants, hex information: https://www.taylorfrancis.com/books/mono/10.1201/9780429031960/hex-inside-ryan-hayward-bjarne-toft  
Computer Hex: https://webdocs.cs.ualberta.ca/~hayward/hex/#wolve  
Some MCTS engine: https://github.com/FutaAlice/FutaHex2  
HEX Monte Carlo Value Iteration: https://github.com/harbecke/HexHex  
General Alpha-Zero player playing HEX: https://github.com/richemslie/galvanise_zero  
Latex hex package: https://ctan.org/tex-archive/macros/latex/contrib/hexgame/  
BayesElo: https://github.com/ddugovic/BayesianElo  
Expert Iteration: https://arxiv.org/pdf/1705.08439.pdf

## A path through github repos
A very inefficent, but simple to understand implementation of Alpha-Zero: https://github.com/suragnair/alpha-zero-general/  
A more efficent implementation by some python multiprocessing wizard. This makes it much more difficult to modify for GNNs however: https://github.com/bhansconnect/fast-alphazero-general  
A C++ plus Python implementation by the same guy: https://github.com/bhansconnect/alphazero-pybind11  
But probably it is better to go with https://github.com/richemslie/galvanise_zero if I wan't to go for a C++ based implementation.  

## Some things to note
Hex: There is an easy, fairly accurate position evaluation function using "voltage flow".

## Benchmark
https://ogb.stanford.edu/docs/leader_nodeprop/

## Tips
Use build pytorch with gpu support without attached gpu: TORCH_CUDA_ARCH_LIST=Turing

## Name of thesis
Maybe: "Improving Generalization of Self-Play Reinforcement Learning Algorithms with Graph Neural Networks"

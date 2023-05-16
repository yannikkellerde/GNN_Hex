# How to continue GNN-Hex research?

## Making existing results more rigorous
+ Implement CNN for HexAra and compare to GNN
+ Do fair comparison between curriculum and not curriculum

## Proceeding where I left off
+ Figure out why Alpha-Go style training improvement stopped after some time (e.g. Agent after one day Alpha-Go style training stronger than only supervised. However, after 5 days, just as strong as after 1 day.)
+ Use other, quicker non Alpha-Go methods to train GNN that then get's put into HexAra. (CrazyAra was not trained with full Alpha-Go method?)

## A little different direction (Back to basics)
+ Before the start of the thesis I implemented thread-space search for winpattern-game Graphs (tic-tac-toe, Qango)
    - What the algorithm does, feels a lot like message passing.
    - For Hex, there is H-search
+ Can GNNs learn to reproduce these search-algorithms?
+ Maybe this question can guide the choice of GNN architecture. Up to now it was pretty arbitrary.
    - H-search is basically a repeated application of first order logic.
    - Maybe we can create a GNN architecture that can implement fuzzy logic operations?
    - To implement H-search OR operation, we can use the max of incoming messages.
    - To implement H-search AND operation, we need a fuzzy operation corresponding to at least two neighbors are true. Not sure how to do this the best way.
![](/images/h-search.svg)

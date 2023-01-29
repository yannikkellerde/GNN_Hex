# GNN, smartly implementing value and swap probability computation
+ After the first move in Hex, the second player has the opportunity to swap
+ Currently, I compute the swap probability taking the node embeddings from my first GNN, then aggregating via sum, min, max, mean and concatenating and then putting them into a 1 hidden layer MLP. The output is then 

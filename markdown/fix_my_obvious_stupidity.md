# Turns out I did stupid things with the loss
+ I pretty much kept the CrazyAra loss functions
+ However, batching in GNN works differently. E.g. just one large dimension.
	- Thus I accidentally computed the mean of the squared error over all batches. (While the Cross-entropy works normally as it has sum reduction)
	- Fixed now, explains a lot of my previous confusion.

#include "util.h"

using namespace std;
using namespace torch::indexing;


int repeatable_random_choice(vector<int>& vec) {
	return vec[rand()%vec.size()]; // This is biased, but who cares
}

// Create batched input for GNN
tuple<vector<torch::jit::IValue>,vector<int>> collate_batch(std::vector<torch::Tensor> & node_features, std::vector<torch::Tensor> & edge_index){
	int starting_ei, next_ei, starting_vi, next_vi;
	starting_ei = 0; starting_vi = 0;
	torch::Tensor big_features = torch::cat(node_features,0);
	torch::Tensor big_ei = torch::cat(edge_index,1);
	torch::Tensor graph_indices = big_ei.new_empty(big_features.size(0));
	vector<int> batch_ptr({0});
	for (int i=0;i<edge_index.size();++i){
		next_ei = starting_ei+edge_index[i].size(1);
		next_vi = starting_vi+node_features[i].size(0);

		graph_indices.index_put_({Slice(starting_vi,next_vi)},i);
		big_ei.index_put_({Ellipsis,Slice(starting_ei,next_ei)},big_ei.index({Ellipsis,Slice(starting_ei,next_ei)})+starting_vi);
		starting_ei = next_ei;
		starting_vi = next_vi;
		batch_ptr.push_back(next_vi);
	}
	return {vector<c10::IValue>({big_features,big_ei,graph_indices}),batch_ptr};
}


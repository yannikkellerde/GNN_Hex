#include "util.h"
#include "shannon_node_switching_game.h"
#include "nn_api.h"

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

void starting_eval_img(int hex_size, NN_api* net){
	stringstream ss;
	std::vector<torch::Tensor> converted, outputs, node_features, edge_indices;
	std::vector<torch::jit::IValue> inputs;
	vector<int> batch_ptr;
	Node_switching_game game(hex_size);
	torch::TensorOptions options_long = torch::TensorOptions().dtype(torch::kLong).device(net->device);
	converted = game.convert_graph(net->device);
	node_features.push_back(converted[0]);
	edge_indices.push_back(converted[1]);
	tie(inputs,batch_ptr) = collate_batch(node_features,edge_indices);
	outputs = net->predict(inputs);
	vector<string> nodetext(game.graph.num_vertices);
	for (int i=0;i<game.graph.num_vertices;++i){
		ss.str(string());
		ss << i << "(" << setprecision(3) << outputs[0][i].item<double>() << ")";
		nodetext[i] = ss.str();
	}
	game.graphviz_me(nodetext,"starting_eval.dot",game.graph);
	system("neato -Tpng starting_eval.dot -o starting_eval.png");
}

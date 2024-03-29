/* Utilities for the graph game environment
 * Key functions:
 * 		collate_batch: create batched GNN input from vectors of node_features and edge_indices
 * 		gen_swap_map: create a swapmap using a neural net and saves to txt (e.g. should a player swap after the first player played at x)
 * 		gen_starting_eval: create a map of evaluations for each possible starting move using a neural net.
 */

#include "util.h"
#include "main/customuci.h"
#include "shannon_node_switching_game.h"
#include "nn_api.h"

using namespace std;
using namespace torch::indexing;


int repeatable_random_choice(vector<int>& vec) {
	return vec[rand()%vec.size()]; // This is biased, but who cares
}

// Create batched input for GNN
vector<torch::jit::IValue> collate_batch(std::vector<torch::Tensor> & node_features, std::vector<torch::Tensor> & edge_index){
	int starting_ei, next_ei, starting_vi, next_vi;
	starting_ei = 0; starting_vi = 0;
	torch::Tensor big_features = torch::cat(node_features,0);
	torch::Tensor big_ei = torch::cat(edge_index,1);
	torch::Tensor graph_indices = big_ei.new_empty(big_features.size(0));
	torch::Tensor batch_ptr = big_ei.new_empty(node_features.size()+1);
	batch_ptr[0] = 0;
	for (int i=0;i<edge_index.size();++i){
		next_ei = starting_ei+edge_index[i].size(1);
		next_vi = starting_vi+node_features[i].size(0);

		graph_indices.index_put_({Slice(starting_vi,next_vi)},i);
		big_ei.index_put_({Ellipsis,Slice(starting_ei,next_ei)},big_ei.index({Ellipsis,Slice(starting_ei,next_ei)})+starting_vi);
		starting_ei = next_ei;
		starting_vi = next_vi;
		batch_ptr[i+1] = next_vi;
	}
	return vector<c10::IValue>({big_features,big_ei,graph_indices,batch_ptr});
}

void gen_swap_map(int hex_size, NN_api* net, bool net_allows_swap){
	std::vector<torch::Tensor> converted, outputs, node_features, edge_indices;
	std::vector<torch::jit::IValue> inputs;
	torch::Tensor output_graph_indices, policy, value;
	ofstream swap_file;
	swap_file.open("swap_map.txt",std::ofstream::out | std::ofstream::trunc);
	Node_switching_game game(hex_size);
	/* for (int i:game.get_actions()){ */
	/* 	game.make_move(i,false,NOPLAYER,true,false); */
	/* 	converted = game.convert_graph(net->device); */
	/* 	node_features.clear(); */
	/* 	edge_indices.clear(); */
	/* 	node_features.push_back(converted[0]); */
	/* 	edge_indices.push_back(converted[1]); */
	/* 	inputs = collate_batch(node_features,edge_indices); */
	/* 	outputs = net->predict(inputs); */
	/* 	swap_file << torch::exp(outputs[0])[game.graph.num_vertices-2].item<double>() << " "; */
	/* 	game.reset(); */
	/* } */
	for (int i:game.get_actions()){
		game.make_move(i,false,NOPLAYER,true,false);
		if (Options["CNN_Mode"]){
			converted = game.convert_planes(net->device);
		}
		else{
			converted = game.convert_graph(net->device);
			edge_indices.push_back(converted[1]);
		}
		node_features.push_back(converted[0]);
		game.reset();
	}
	inputs = collate_batch(node_features,edge_indices);
	outputs = net->predict(inputs);
	if (net_allows_swap){
		policy = torch::exp(outputs[0]);
		output_graph_indices = outputs[3];
		for (int i=1;i<output_graph_indices.size(0);++i){
			swap_file << policy[output_graph_indices[i]-1].item<double>() << " ";
		}
	}
	else{
		value = outputs[1];
		for (int i=0;i<value.size(0);++i){
			swap_file << (0.5-0.5*value[i].item<double>()) << " ";
		}
	}
	/* swap_file << endl << policy << endl << output_graph_indices; */
	/* swap_file << endl << policy.sizes() << endl << output_graph_indices.sizes(); */
	swap_file.close();
}

void gen_starting_eval_file(int hex_size, NN_api* net){
	stringstream ss;
	std::vector<torch::Tensor> converted, outputs, node_features, edge_indices;
	std::vector<torch::jit::IValue> inputs;
	Node_switching_game game(hex_size);
	torch::TensorOptions options_long = torch::TensorOptions().dtype(torch::kLong).device(net->device);
	if (Options["CNN_Mode"]){
		converted = game.convert_planes(net->device);
	}
	else{
		converted = game.convert_graph(net->device);
		edge_indices.push_back(converted[1]);
	}
	node_features.push_back(converted[0]);
	inputs = collate_batch(node_features,edge_indices);
	outputs = net->predict(inputs);
	ofstream ev_file;
	ev_file.open("starting_eval.txt",std::ofstream::out | std::ofstream::trunc);
	for (int i=0;i<game.graph.num_vertices-2;++i){
		ev_file << torch::exp(outputs[0])[i].item<double>() << " ";
	}
	ev_file.close();

	/* vector<string> nodetext(game.graph.num_vertices); */
	/* for (int i=0;i<game.graph.num_vertices-2;++i){ */
	/* 	ss.str(string()); */
	/* 	ss << i << "(" << setprecision(3) << outputs[0][i].item<double>() << ")"; */
	/* 	nodetext[i] = ss.str(); */
	/* } */
	/* game.graphviz_me(nodetext,"starting_eval.dot",game.graph); */
	/* system("neato -Tpng starting_eval.dot -o starting_eval.png"); */
}

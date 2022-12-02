#include "../util.h"
#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include "../shannon_node_switching_game.h"
#include "../nn_api.h"
#include <blaze/Math.h>

#include <iostream>
#include <memory>
#include "test_torch_script.h"

torch::Device device(torch::kCUDA,0);

/* TORCH_LIBRARY(my_ops, m) { */
/* 	m.def("torch_scatter::segment_sum_csr", segment_sum_csr); */
/* } */

void test_torch_script(string fname) {
	/* torch::Tensor a = torch::ones({10}); */
	/* torch::Tensor indptr = torch::empty({3}).to(torch::kLong); */
	/* indptr[0] = 0; */
	/* indptr[1] = 3; */
	/* indptr[2] = 10; */
	/* torch::Tensor res = segment_sum_csr(a,indptr,at::nullopt); */
	/* std::cout << res[0] << "  " << res[1]; */

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(fname);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return;
  }

	module.to(device);
  std::cout << "ok\n";
	torch::TensorOptions options_long = torch::TensorOptions().dtype(torch::kLong).device(device);
	torch::TensorOptions options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
	torch::Tensor node_features,edge_index,graph_indices;
	std::vector<torch::Tensor> inputs;
	/* node_features = torch::ones({5,3},options_float); */
	/* edge_index = torch::empty({2,2},options_long); */
	/* graph_indices = torch::zeros({5},options_long); */
	/* edge_index[0][0] = 0;edge_index[0][1] = 3;edge_index[1][0] = 3;edge_index[1][1]=4; */
	/* inputs.push_back(node_features); */
	/* inputs.push_back(edge_index); */
	/* inputs.push_back(graph_indices); */
	/* std::cout << module.forward(inputs); */
	
	Node_switching_game game(5);
	graph_indices = torch::zeros(game.graph.num_vertices,options_long);
	inputs = game.convert_graph(device);
	inputs.push_back(graph_indices);
	vector<c10::IValue> vi(inputs.begin(),inputs.end());
	c10::IValue res = module.forward(vi);

	std::cout << res << endl;

	NN_api net(fname,device);
	inputs = game.convert_graph(device);
	vector<torch::Tensor> ei, node_feat;
	ei.push_back(inputs[1]);
	node_feat.push_back(inputs[0]);
	game.make_move(8);
	game.make_move(4);
	inputs = game.convert_graph(device);
	ei.push_back(inputs[1]);
	node_feat.push_back(inputs[0]);
	torch::Tensor batch_ptr;
	vi = collate_batch(node_feat,ei);
	std::map<string,vector<torch::Tensor>> stuff;
	/* stuff["node_features"] = node_feat; */
	/* stuff["ei"] = ei; */
	/* torch::save(stuff,"test.pt"); */

	cout << vi[0].toTensor().sizes() << endl;
	cout << vi[1].toTensor().sizes() << endl;
	cout << vi[2].toTensor().sizes() << endl;
	cout << vi[0].toTensor().dtype() << endl;
	cout << vi[1].toTensor().dtype() << endl;
	cout << vi[2].toTensor().dtype() << endl;
	cout << vi[0].toTensor().max() << endl;
	cout << vi[1].toTensor().max() << endl;
	cout << vi[2].toTensor().max() << endl;
	cout << vi[1] << endl;

	torch::jit::Module load_stuff = torch::jit::load("test.pt");
	cout << *(++load_stuff.parameters().begin()) << endl;

	/* torch::Tensor tens; */
	vector<torch::Tensor> tens;
	torch::load(tens,"test.pt");
	cout << tens << endl;
	/* vector<at::Tensor> out = net.predict(vi); */

	/* blaze::DynamicVector<double> probvec = torch_to_blaze<double>(out[0]); */
	/* std::cout << res << endl; */
	/* std::cout << probvec << endl << probvec.size() << endl; */
}

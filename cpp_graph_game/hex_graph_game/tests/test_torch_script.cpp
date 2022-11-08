/* #include <torch/script.h> // One-stop header. */
#include <torch/script.h>
/* #include <torchscatter/scatter.h> */
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include "../shannon_node_switching_game.cpp"

#include <iostream>
#include <memory>

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
	std::vector<torch::jit::IValue> inputs;
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
	c10::IValue res = module.forward(inputs);
	auto out = res.toTupleRef();
	auto he = out.elements()[0].toTensor();

	std::cout << he;
}

#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <string.h>
#include <vector>

#if !defined(NNAPI_H)
#define NNAPI_H

class NN_api{
	public:
		torch::Tensor probOutputs;
		torch::Tensor valueOutputs;
		std::vector<torch::Tensor> edge_indices;
		std::vector<torch::Tensor> node_features;
		torch::Tensor batch_ptr;

		torch::jit::script::Module model;
		torch::Device device;
		std::string model_name;
		NN_api(std::string fname, torch::Device device);
		std::vector<at::Tensor> predict(std::vector<torch::jit::IValue> inputs);
		std::vector<at::Tensor> predict(std::vector<torch::Tensor> inputs);
		void predict_stored();
};
#endif

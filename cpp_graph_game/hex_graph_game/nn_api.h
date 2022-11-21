#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <string.h>

#if !defined(NNAPI_H)
#define NNAPI_H

class NN_api{
	public:
		torch::jit::script::Module model;
		torch::Device device;
		std::string model_name;
		NN_api(std::string fname, torch::Device device);
		std::vector<at::Tensor> predict(std::vector<torch::jit::IValue> inputs);
		std::vector<at::Tensor> predict(std::vector<torch::Tensor> inputs);
};
#endif

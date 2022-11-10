#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <string.h>

using namespace std;

#if !defined(NNAPI_H)
#define NNAPI_H

class NN_api{
	public:
		torch::jit::script::Module model;
		torch::Device device;
		string model_name;
		NN_api(string fname, torch::Device device);
		vector<at::Tensor> predict(vector<torch::jit::IValue> inputs);
};
#endif

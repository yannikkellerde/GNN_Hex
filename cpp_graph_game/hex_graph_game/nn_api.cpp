#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <string.h>

using namespace std;

#if !defined(NNAPI)
#define NNAPI

class NN_api{
	public:
		torch::jit::script::Module model;
		torch::Device device;
		string model_name;
		NN_api(string fname, torch::Device device):
			model_name(fname.substr(fname.find_last_of("/\\") + 1)),
			model(torch::jit::load(fname)),device(device){
				model.to(device);
		}
		vector<at::Tensor> predict(vector<torch::jit::IValue> inputs){
			// This is clearly not the most efficient way to do this, but idk, that's the only
			// thing that seems to work.
			c10::ivalue::Tuple res = model.forward(inputs).toTupleRef();
			return vector<at::Tensor>({res.elements()[0].toTensor(),res.elements()[1].toTensor()});
		}
};
#endif

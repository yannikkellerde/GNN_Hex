#include "nn_api.h"

using namespace std;

NN_api::NN_api(string fname, torch::Device device):
	model_name(fname.substr(fname.find_last_of("/\\") + 1)),
	model(torch::jit::load(fname)),device(device){
		model.to(device);
}

vector<at::Tensor> NN_api::predict(vector<torch::jit::IValue> inputs){
	c10::ivalue::Tuple res = model.forward(inputs).toTupleRef();
	return vector<at::Tensor>({res.elements()[0].toTensor(),res.elements()[1].toTensor(),res.elements()[2].toTensor(),res.elements()[3].toTensor()});
}

vector<at::Tensor> NN_api::predict(vector<torch::Tensor> inputs){
	vector<c10::IValue> vi(inputs.begin(),inputs.end());
	return predict(vi);
}

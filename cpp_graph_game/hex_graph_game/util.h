#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <blaze/Math.h>
#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include "nn_api.h"


#if !defined(UTIL_H)
#define UTIL_H

int repeatable_random_choice(std::vector<int>& vec);

void starting_eval_img(int hex_size, NN_api* net);

std::tuple<std::vector<torch::jit::IValue>,std::vector<int>> collate_batch(std::vector<torch::Tensor> & node_features, std::vector<torch::Tensor> & edge_index);

template <class T>
blaze::DynamicVector<T> torch_to_blaze(const at::Tensor& t){
	c10::ScalarType torch_type;
	torch::Tensor m;
	if (std::is_same<T,float>()){
		torch_type = torch::kFloat32;
	}
	else if (std::is_same<T,double>()){
		torch_type = torch::kDouble;
	}
	else if (std::is_same<T,long>()){
		torch_type = torch::kLong;
	}
	else{
		throw std::logic_error("Not Implemented");
	}
	m = t.cpu().to(torch_type).contiguous();
	blaze::DynamicVector<T> v(m.numel(),m.data_ptr<T>());
	return v;
}

template<typename T>
void info_string(const T &message ){
	std::cout << "INFO: " << message << std::endl;
}
template<typename T, typename U>
void info_string(const T &messageA, const U &messageB) {
    std::cout << "INFO: " << messageA << ' ' << messageB << std::endl;
}
template<typename T, typename U, typename V>
void info_string(const T &messageA, const U &messageB, const V &messageC) {
    std::cout << "INFO: " << messageA << ' ' << messageB << ' ' << messageC << std::endl;
}

template<typename T>
void print_info(const int line_number, const std::string filename, const T &message ){
	std::string base_filename = filename.substr(filename.find_last_of("/\\") + 1);
	std::cout << "File " << base_filename << " Line " << line_number << ": " << message << std::endl;
}
template<typename T, typename U>
void print_info(const int line_number, const std::string filename, const T &messageA, const U &messageB ){
	std::string base_filename = filename.substr(filename.find_last_of("/\\") + 1);
	std::cout << "File " << base_filename << " Line " << line_number << ": " << messageA << ' ' << messageB << std::endl;
}

template<typename T, typename U, typename V>
void print_info(const int line_number, const std::string filename, const T &messageA, const U &messageB, const V &messageC){
	std::string base_filename = filename.substr(filename.find_last_of("/\\") + 1);
	std::cout << "File " << base_filename << " Line " << line_number << ": " << messageA << ' ' << messageB << ' ' << messageC << std::endl;
}

template<typename T, typename U, typename V, typename K>
void print_info(const int line_number, const std::string filename, const T &messageA, const U &messageB, const V &messageC, const K &messageD){
	std::string base_filename = filename.substr(filename.find_last_of("/\\") + 1);
	std::cout << "File " << base_filename << " Line " << line_number << ": " << messageA << ' ' << messageB << ' ' << messageC << ' ' << messageD << std::endl;
}

#endif

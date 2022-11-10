#include "util.h"

using namespace std;
using namespace torch::indexing;
using blaze::DynamicVector;

#if !defined(UTIL_H)
#define UTIL_H

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}

int repeatable_random_choice(vector<int>& vec) {
	return vec[rand()%vec.size()]; // This is biased, but who cares
}

template <class T>
DynamicVector<T> torch_to_blaze(const torch::Tensor& t){
	c10::ScalarType torch_type;
	torch::Tensor m;
	if (is_same<T,float>()){
		torch_type = torch::kFloat32;
	}
	else if (is_same<T,double>()){
		torch_type = torch::kDouble;
	}
	else if (is_same<T,long>()){
		torch_type = torch::kLong;
	}
	else{
		throw std::logic_error("Not Implemented");
	}
	m = t.cpu().to(torch_type).contiguous();
	DynamicVector<T> v(m.numel(),m.data_ptr<double>());
	return v;
}


// Create batched input for GNN
tuple<vector<torch::jit::IValue>,vector<int>> collate_batch(std::vector<torch::Tensor> & node_features, std::vector<torch::Tensor> & edge_index){
	int starting_ei, next_ei;
	starting_ei = 0;
	torch::Tensor big_features = torch::cat(node_features,0);
	torch::Tensor big_ei = torch::cat(edge_index,1);
	torch::Tensor graph_indices = big_ei.new_empty(big_features.size(0));
	vector<int> batch_ptr({0});
	for (int i=0;i<edge_index.size();++i){
		next_ei = starting_ei+node_features[i].size(0);
		graph_indices.index_put_({Slice(starting_ei,next_ei)},i);
		big_ei.index_put_({Ellipsis,Slice(starting_ei,next_ei)},big_ei.index({Ellipsis,Slice(starting_ei,next_ei)})+starting_ei);
		starting_ei = next_ei;
		batch_ptr.push_back(next_ei);
	}
	return {vector<c10::IValue>({big_features,big_ei,graph_indices}),batch_ptr};
}

template<typename T>
void info_string(const T &message ){
    cout << "INFO: " << message << endl;
}
template<typename T, typename U>
void info_string(const T &messageA, const U &messageB) {
    cout << "INFO: " << messageA << ' ' << messageB << endl;
}
template<typename T, typename U, typename V>
void info_string(const T &messageA, const U &messageB, const V &messageC) {
    cout << "INFO: " << messageA << ' ' << messageB << ' ' << messageC << endl;
}

#endif

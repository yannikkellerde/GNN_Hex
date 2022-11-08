#include <iterator>
#include <utility>
#include <vector>
#include <map>
#include <random>
#include <blaze/Math.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>


using namespace std;
using blaze::DynamicVector;

#if !defined(UTIL)
#define UTIL

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

DynamicVector<double> torch_to_blaze(torch::Tensor& t){
	t = t.cpu().to(torch::kDouble).contiguous();
	DynamicVector<double> v(t.numel(),t.data_ptr<double>());
	return v;
}


#endif

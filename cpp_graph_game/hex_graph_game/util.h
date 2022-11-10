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
using namespace torch::indexing;
using blaze::DynamicVector;

#if !defined(UTIL_H)
#define UTIL_H

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g);

template<typename Iter>
Iter select_randomly(Iter start, Iter end);

int repeatable_random_choice(vector<int>& vec);

template <class T>
DynamicVector<T> torch_to_blaze(const torch::Tensor& t);

// Create batched input for GNN
tuple<vector<torch::jit::IValue>,vector<int>> collate_batch(std::vector<torch::Tensor> & node_features, std::vector<torch::Tensor> & edge_index);

template<typename T>
void info_string(const T &message );

template<typename T, typename U>
void info_string(const T &messageA, const U &messageB);


template<typename T, typename U, typename V>
void info_string(const T &messageA, const U &messageB, const V &messageC);

#endif

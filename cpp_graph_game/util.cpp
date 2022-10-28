#include <iterator>
#include <utility>
#include <vector>
#include <map>
#include <random>

using namespace std;

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

int repeatable_random_choice(vector<int> vec) {
	return vec[rand()%vec.size()]; // This is biased, but who cares
}

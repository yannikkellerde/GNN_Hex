#include <iostream>
#include <iterator>
#include <utility>
#include <algorithm>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
using namespace std;
using namespace boost;

int main() {
	typedef adjacency_list<vecS, vecS, bidirectionalS> Graph;

	enum {A,B,C,D,E,N};
	const int num_vertices = N;
	const char* name = "ABCDE";

	typedef std::pair<int, int> Edge;

  return 0;
} 

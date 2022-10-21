#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <vector>
using namespace std;

template<int N>
class Graph{
	private:
		vector<int> adj[N];
		int num_vertices = N;
	public:
		Graph(){}
		void add_edge(int vertex1, int vertex2){
			adj[vertex1].push_back(vertex2);
			adj[vertex2].push_back(vertex1);
		}
		void remove_vertex(int vertex){
			
		}

};

#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include <algorithm>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include "hex_board_game.cpp"
using namespace std;
using namespace boost;



struct PropertyStruct{
	bool filter;
	int index;
};
typedef adjacency_list<vecS, vecS, undirectedS, PropertyStruct, no_property> Graph;
typedef pair<int, int> Edge;
typedef property_map<Graph, vertex_index_t>::type IndexMap;
typedef typename Graph::vertex_descriptor Vertex;
typedef set<string> labels ;

struct Predicate { // both edge and vertex
	bool operator()(Graph::edge_descriptor) const      { return true; } // all
	bool operator()(Vertex vd) const { return (*g)[vd].filter; }
	Graph* g;
};

typedef filtered_graph<Graph, Predicate, Predicate> Filtered;

int main() {
  Graph G;
	write_graphviz(cout,G);
	Vertex v,vend;
  Graph G;
	Predicate predicate{&G};
	int i;
	for (i=0;i<8;++i){
		add_vertex(PropertyStruct{true,i},G);
	}
	for (tie(v,vend)=vertices(G);v!=vend;v++){
		G[v].filter = true;
	}
	G[*vertices(G).first].filter = false;
	Filtered fg(G, predicate, predicate);
	write_graphviz(cout, G, make_label_writer(get(&PropertyStruct::filter, G)));
	cout << endl << endl;
	write_graphviz(cout, fg, make_label_writer(get(&PropertyStruct::filter, fg)));
	typedef adjacency_list<vecS, vecS, undirectedS> Graph;

	enum {A,B,C,D,E,N};
	const int num_vertices = N;
	const char* name = "ABCDE";

	typedef pair<int, int> Edge;
	Edge edge_array[] = {Edge(A,B),Edge(A,D),Edge(C,A),Edge(D,C),Edge(C,E)};
	const int num_edges = sizeof(edge_array)/sizeof(edge_array[0]);

	Graph g(num_vertices);

	for (int i=0;i<num_edges;++i)
		add_edge(edge_array[i].first,edge_array[i].second,g);
	

	typedef property_map<Graph,vertex_index_t>::type IndexMap;
	IndexMap index = get(vertex_index,g);

	cout << vertex_index;
	cout << "\n";
	cout << N;
	cout << "\nvertices(g) = ";

	typedef graph_traits<Graph>::vertex_iterator vertex_iter;
	pair<vertex_iter, vertex_iter> vp;
	for (vp = vertices(g);vp.first!=vp.second;++vp.first)
		cout << index[*vp.first] << " ";
	cout << endl;

	cout << "edges(g) = ";
	graph_traits<Graph>::edge_iterator ei,ei_end;
	for (tie(ei, ei_end) = edges(g); ei!=ei_end; ++ei)
		cout << "(" << index[source(*ei, g)]
				 << "," << index[target(*ei, g)] << ") ";
	cout << endl;

	for_each(vertices(g).first,vertices(g).second,exercise_vertex<Graph>(g));

	ofstream graph_file("graph_file.txt");
	write_graphviz(graph_file,g);
	return 0;
} 

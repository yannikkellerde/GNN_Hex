#include <string>
#include <vector>

#if !defined(GRAPH_H)
#define GRAPH_H

typedef std::pair<std::vector<int>::iterator,std::vector<int>::iterator> Neighbors;

class Graph{
	public:
		std::vector<int> sources;
		std::vector<int> targets;
		int num_vertices;
		std::vector<std::vector<int>> lprops;
		std::vector<std::vector<float>> fprops;
		std::vector<int> edge_starts;

		Graph();

		Graph(int num_verts);

		bool add_edge_onside(int s, int t);

		bool add_edge(int v1, int v2);

		Neighbors adjacent_vertices(int vertex);

		int num_neighbors(int vertex);

		bool delete_edge(int v1, int v2);

		bool delete_edge_onesided(int source, int target);

		void delete_many_onesided(std::vector<int> s, int t);

		void add_lprop(int init);

		void add_fprop(float init);

		bool edge_exists(int source, int target);

		void remove_vertex(int vertex);

		void clear_vertex(int vertex);

		void do_complete_dump(std::string fname="graph_dump.txt");

		void graphviz_me(std::vector<std::pair<std::string,std::vector<std::string>>> props, std::string fname="my_graph.dot", bool undirected=true);

		void graphviz_me(std::string fname="my_graph.dot",bool undirected=true);
};

#endif

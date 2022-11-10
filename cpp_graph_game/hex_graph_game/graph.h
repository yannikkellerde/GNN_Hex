#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

#if !defined(GRAPH_H)
#define GRAPH_H

typedef pair<vector<int>::iterator,vector<int>::iterator> Neighbors;

class Graph{
	public:
		vector<int> sources;
		vector<int> targets;
		int num_vertices;
		vector<vector<int>> lprops;
		vector<vector<float>> fprops;
		vector<int> edge_starts;

		Graph();

		Graph(int num_verts);

		bool add_edge_onside(int s, int t);

		bool add_edge(int v1, int v2);

		Neighbors adjacent_vertices(int vertex);

		int num_neighbors(int vertex);

		bool delete_edge(int v1, int v2);

		bool delete_edge_onesided(int source, int target);

		void delete_many_onesided(vector<int> s, int t);

		void add_lprop(int init);

		void add_fprop(float init);

		bool edge_exists(int source, int target);

		void remove_vertex(int vertex);

		void clear_vertex(int vertex);

		void do_complete_dump(string fname="graph_dump.txt");

		void graphviz_me(vector<pair<string,vector<string>>> props, string fname="my_graph.dot", bool undirected=true);

		void graphviz_me(string fname="my_graph.dot",bool undirected=true);
};

#endif

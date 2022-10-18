#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include <algorithm>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include "hex_board_game.cpp"
#include <math.h>
using namespace std;
using namespace boost;



struct PropertyStruct{
	int board_location;
};
typedef adjacency_list<vecS, vecS, undirectedS, PropertyStruct, no_property> Graph;
typedef pair<int, int> Edge;
typedef property_map<Graph, vertex_index_t>::type IndexMap;
typedef typename Graph::vertex_descriptor Vertex;
typedef set<string> labels ;
enum Onturn {maker,breaker};
enum Teminals {terminal1, terminal2};

class Node_switching_game {
	public:
		Graph graph;
		IndexMap vi_map;
		Onturn onturn=maker;
		int board_size;

		Node_switching_game (){
			vi_map = get(vertex_index, graph);
		};
		Node_switching_game (Graph g){
			graph = g;
		};
		template<int S>
		Node_switching_game (Hex_board<S> board){
			board_size = S;
			graph = Graph(board.num_squares+2);
			for (int i=0;i<board.num_squares;i++){
				graph[i+2].board_location = i;
				if (i<board.size){
					add_edge(i+2,terminal1,graph);
				}
				if (floor(i/board.size)==board.size-1){
					add_edge(i+2,terminal2,graph);
				}
				if (i%board.size>0 && board.size<=i<=board.num_squares-1){
					add_edge(i+2,i+1,graph);
				}
				if (i>=board.size){
					add_edge(i+2,i+2-board.size,graph);
					if (i%board.size!=board.size-1){
						add_edge(i+2,i+3-board.size,graph);
					}
				}
			}
			vi_map=get(vertex_index, graph);
		};
		void switch_onturn(){
			onturn = onturn==maker?breaker:maker;
		}

		void make_move(int vertex, bool do_force_color=false, Onturn force_color=maker){
			Onturn player = do_force_color?force_color:onturn;
			if (!do_force_color){
				switch_onturn();
			}
			if (player==maker){
				int have_to_fix = -1;
				auto neighbors = adjacent_vertices(vertex,graph);
				for (auto neigh = adjacent_vertices(vertex,graph);neigh.first!=neigh.second;++neigh.first){
					for (auto it2 = neigh.first+1;it2!=neigh.second;++it2){
						Vertex v1 = *neigh.first;
						Vertex v2 = *it2;
						if (vi_map[v1] == terminal1 || vi_map[v1] == terminal2){
							have_to_fix = vi_map[v1];
						}
						if (!((edge(v1,terminal1,graph).second&&edge(v2,terminal1,graph).second)||
									(edge(v1,terminal2,graph).second&&edge(v2,terminal2,graph).second)||
									edge(v1,v2,graph).second)){
							add_edge(v1,v2,graph);
						}
					}
				}
			}
			clear_vertex(vertex,graph);
			remove_vertex(vertex,graph);
		}

		vector<double>* get_grid_layout(){
			double scale;
			vector<double>* position_array;
			position_array = new vector<double>[num_vertices(graph)];
			scale = 400./board_size;
			const double xstart = 0;
			const double ystart = 0;
			const double xend = xstart+1.5*(board_size-1)*scale;
			const double yend = ystart+sqrt(3./4.)*(board_size-1)*scale;
			position_array[terminal1][0] = xstart;
			position_array[terminal1][1] = yend/2;
			position_array[terminal2][0] = xend;
			position_array[terminal2][1] = yend/2;
			for (int i=2;i<num_vertices(graph);i++){
				int bi = graph[i].board_location;
				int row = floor(bi/board_size);
				int col = bi%board_size;
				position_array[i][0] = xstart+(0.5*col+row)*scale;
				position_array[i][1] = yend - (sqrt(3./4.)*col)*scale;
			}
			return position_array;
		}

		void graphviz_me (ostream &out){
			vector<double> *pos_array = get_grid_layout();
			dynamic_properties dp;
			dp.property("pos",pos_array);
			write_graphviz_dp(out,graph,dp);
		};
};



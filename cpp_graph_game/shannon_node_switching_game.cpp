#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include <algorithm>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_iterator.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/depth_first_search.hpp>
#include "hex_board_game.cpp"
#include <math.h>
#include <boost/config.hpp>
using namespace std;
using namespace boost;


struct PropertyStruct{
	int board_location;
	bool removed;
};

typedef adjacency_list<vecS, vecS, undirectedS, PropertyStruct, no_property> Graph;
typedef pair<int, int> Edge;
typedef property_map<Graph, vertex_index_t>::type IndexMap;
typedef iterator_property_map<string*,IndexMap> StringPropMap;
typedef typename Graph::vertex_descriptor Vertex;
typedef set<string> labels ;
typedef pair<typename Graph::adjacency_iterator,typename Graph::adjacency_iterator> Neighbors; 
enum Onturn {noplayer,maker,breaker};
enum Teminals {terminal1, terminal2};

bool path_exists(Graph g, int src, int dest) {
  bool visited[num_vertices(g)];
	fill(visited,visited+num_vertices(g),false);
  visited[src] = true;
  std::stack<int> next;
  next.push(src);

  while(!next.empty()) {
    int cv = next.top();
    next.pop();
		for (Neighbors neigh = adjacent_vertices(cv,g);neigh.first!=neigh.second;++neigh.first){
			int nv = *neigh.first;
			if (!visited[nv]){
				if (nv==dest){
					return true;
				}
				visited[nv] = true;
				next.push(nv);
			}
		}
	}
  return false;
}

bool check_if_same(Graph g, Neighbors neigh1, Neighbors neigh2, int ignore1, int ignore2){
	set<int> store;
	for (;neigh1.first!=neigh1.second;++neigh1.first){
		int v = *neigh1.first;
		if (v != ignore1){
			store.insert(v);
		}
	}
	for (;neigh2.first!=neigh2.second;++neigh2.first){
		int v = *neigh2.first;
		if (v != ignore2){
			if (store.find(v)!=store.end()){
				store.erase(v);
			}
			else{
				return false;
			}
		}
	}
	return store.empty();
}

bool is_fully_connected(Graph g,Neighbors neigh, int ignore){
	for (;neigh.first!=neigh.second;++neigh.first){
		int v1 = *neigh.first;
		if (v1!=ignore){
			for (Graph::adjacency_iterator it2 = neigh.first+1;it2!=neigh.second;++it2){
				int v2 = *it2;
				if (v2!=ignore && !edge(v1,v2,g).second){
					return false;
				}
			}
		}
	}
	return true;
}

bool is_fully_connected(Graph g,Neighbors neigh){
	for (;neigh.first!=neigh.second;++neigh.first){
		for (Graph::adjacency_iterator it2 = neigh.first+1;it2!=neigh.second;++it2){
			if (!edge(*neigh.first,*it2,g).second){
				return false;
			}
		}
	}
	return true;
}

template <int S>
class Node_switching_game {
	public:
		Graph graph;
		IndexMap vi_map;
		Onturn onturn=maker;
		int board_size = S;
		Hex_board<S> board;
		int speed_cut = 10;

		Node_switching_game (){
			vi_map = get(vertex_index, graph);
		};
		Node_switching_game (Graph g){
			graph = g;
		};
		Node_switching_game (Hex_board<S> from_board){
			board = from_board;
			graph = Graph(board.num_squares+2);
			graph[terminal1].removed = false;
			graph[terminal2].removed = false;
			for (int i=0;i<board.num_squares;i++){
				graph[i+2].board_location = i;
				/* graph[i+2].removed = false; */
				if (i<board.size){
					add_edge(i+2,terminal1,graph);
				}
				if (floor(i/board.size)==board.size-1){
					add_edge(i+2,terminal2,graph);
				}
				if (i%board.size>0 && board.size<=i && i<=board.num_squares-board.size){
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

		set<int> fix_terminal_connections(int terminal){
			set<int> change_set;
			for (Neighbors neigh = adjacent_vertices(terminal,graph);neigh.first!=neigh.second;++neigh.first){
				for (Graph::adjacency_iterator it2 = neigh.first+1;it2!=neigh.second;++it2){
					int v1 = *neigh.first;
					int v2 = *it2;
					if (edge(v1,v2,graph).second){
						remove_edge(v1,v2,graph);
						change_set.insert(v1);
						change_set.insert(v2);
					}
				}
			}
			return change_set;
		}

		void remove_marked_nodes(){
			for (int i=num_vertices(graph)-1;i>=0;--i){
				if (graph[i].removed){
					remove_vertex(i,graph);
					cout << "removing " << i << endl;
				}
			}
		}

		void remove_dead_and_captured(set<int> consider_set){
			bool can_continue,is_dead;
			set<int> big_set, change_set;
			Neighbors neigh_neigh,tmp_neigh;
			for (int vertex : consider_set){
				/* cout << vertex << endl; */
				is_dead = true;
				can_continue = false;
				if (graph[vertex].removed || vertex<2){
					continue;
				}
				Neighbors neighbors = adjacent_vertices(vertex,graph);
				
				if (neighbors.second-neighbors.first<speed_cut){
					neigh_neigh = adjacent_vertices(*neighbors.first,graph);
					for (;neigh_neigh.first!=neigh_neigh.second;++neigh_neigh.first){
						int v1 = *neigh_neigh.first;
						if (v1<2 || v1==vertex){
							continue;
						}
						Neighbors neigh_neigh_neigh = adjacent_vertices(v1,graph);
						neighbors = adjacent_vertices(vertex,graph);
						if (neigh_neigh_neigh.second-neigh_neigh_neigh.first < speed_cut){
							if (check_if_same(graph,neighbors,neigh_neigh_neigh,v1,vertex)){
									/* tmp_neigh = adjacent_vertices(vertex,graph);       // These are not */
									/* big_set.insert(tmp_neigh.first,tmp_neigh.second);  // in the python original */
								cout << "maker captured " << v1 << vertex << endl;
								neigh_neigh_neigh = adjacent_vertices(v1,graph);
								big_set.insert(neigh_neigh_neigh.first,neigh_neigh_neigh.second);
								graph[v1].removed = true;
								clear_vertex(v1,graph);
								change_set = make_move(vertex,true,maker,false,true);
								big_set.insert(change_set.begin(),change_set.end());
								can_continue = true;
								break;
							}
						}
					}
				}
				if (can_continue){
					continue;
				}
				for (neighbors = adjacent_vertices(vertex,graph);neighbors.first!=neighbors.second;++neighbors.first){
					int v1 = *neighbors.first;
					for (Graph::adjacency_iterator it2 = neighbors.first+1;it2!=neighbors.second;++it2){
						if (!edge(*neighbors.first,*it2,graph).second){
							is_dead = false;
						}
					}
					if (v1 < 2){
						continue;
					}
					tmp_neigh = adjacent_vertices(vertex,graph);
					neigh_neigh = adjacent_vertices(v1,graph);
					if (tmp_neigh.second-tmp_neigh.first < speed_cut || neigh_neigh.second-neigh_neigh.first < speed_cut){
						if (is_fully_connected(graph,tmp_neigh,v1) && is_fully_connected(graph,neigh_neigh,vertex)){
							cout << "breaker captured " << v1 << " " << vertex << endl;
							tmp_neigh = adjacent_vertices(vertex,graph);
							neigh_neigh = adjacent_vertices(v1,graph);
							big_set.insert(tmp_neigh.first,tmp_neigh.second);
							big_set.insert(neigh_neigh.first,neigh_neigh.second);
							clear_vertex(v1,graph);
							clear_vertex(vertex,graph);
							graph[v1].removed = true;
							graph[vertex].removed = true;
							can_continue = true;
							break;
						}
					}
				}
				if (is_dead && !can_continue){
					cout << "node is dead" << endl;
					neighbors = adjacent_vertices(vertex,graph);
					big_set.insert(neighbors.first,neighbors.second);
					graph[vertex].removed = true; // node is dead
					clear_vertex(vertex,graph);
					continue;
				}
			}
			if (!big_set.empty()){
				remove_dead_and_captured(big_set);
			}
		}

		set<int> make_move(int vertex, bool do_force_color=false, Onturn force_color=maker,bool do_remove_dead_and_captured=false,bool only_mark_removed=false){
			set<int> change_set;
			Onturn player = do_force_color?force_color:onturn;
			if (!do_force_color){
				switch_onturn();
			}
			if (player==maker){
				int have_to_fix = -1;
				for (Neighbors neigh = adjacent_vertices(vertex,graph);neigh.first!=neigh.second;++neigh.first){
					int v1 = *neigh.first;
					if (v1 == terminal1 || v1 == terminal2){
						have_to_fix = v1;
					}
					for (Graph::adjacency_iterator it2 = neigh.first+1;it2!=neigh.second;++it2){
						Vertex v2 = *it2;
						if (!((edge(v1,terminal1,graph).second&&edge(v2,terminal1,graph).second)||
									(edge(v1,terminal2,graph).second&&edge(v2,terminal2,graph).second)||
									edge(v1,v2,graph).second)){
							add_edge(v1,v2,graph);
						}
					}
				}
				if (have_to_fix!=-1){
					change_set = fix_terminal_connections(have_to_fix);
				}
			}
			if (do_remove_dead_and_captured){
				Neighbors neigh = adjacent_vertices(vertex,graph);
				change_set.insert(neigh.first,neigh.second);
			}
			clear_vertex(vertex,graph);
			if (do_remove_dead_and_captured || only_mark_removed){
				graph[vertex].removed = true;
			}
			else{
				remove_vertex(vertex,graph);
			}
			if (do_remove_dead_and_captured){
				remove_dead_and_captured(change_set);
				remove_marked_nodes();
			}
			return change_set;
		}

		Onturn who_won(){
			if (edge(terminal1,terminal2,graph).second){
				return maker;
			}
			if (!path_exists(graph,terminal1,terminal2)){
				return breaker;
			}
			return noplayer;
		}

		StringPropMap get_grid_layout(){
			double scale;
			string* position_array;
			position_array = new string[num_vertices(graph)];
			scale = 5./board_size;
			const double xstart = 0;
			const double ystart = 0;
			const double xend = xstart+1.5*(board_size-1)*scale;
			const double yend = ystart+sqrt(3./4.)*(board_size-1)*scale;
			position_array[terminal1] = to_string(xstart)+","+to_string((yend/2))+"!";
			position_array[terminal2] = to_string(xend)+","+to_string((yend/2))+"!";
			for (int i=2;i<num_vertices(graph);i++){
				int bi = graph[i].board_location;
				int row = floor(bi/board_size);
				int col = bi%board_size;
				position_array[i] = to_string((xstart+(0.5*col+row)*scale))+","+to_string((ystart + (sqrt(3./4.)*col)*scale))+"!";
			}
			StringPropMap pos_map = make_iterator_property_map(position_array,vi_map);
			return pos_map;
		}

		StringPropMap get_colors(){
			string* color_array;
			color_array = new string[num_vertices(graph)];
			for (int i=2;i<num_vertices(graph);i++){
				color_array[i] = "black";
			}
			color_array[terminal1] = "red";
			color_array[terminal2] = "red";
			StringPropMap pos_map = make_iterator_property_map(color_array,vi_map);
			return pos_map;
		}

		void graphviz_me (ostream &out){
			get_grid_layout();
			StringPropMap pos_map = get_grid_layout();
			StringPropMap color_map = get_colors();
			cout << "here " << num_vertices(graph) << endl;
			dynamic_properties dp;
			dp.property("color", color_map);
			dp.property("pos",pos_map);
			dp.property("node_id", get(vertex_index, graph));
			write_graphviz_dp(out,graph,dp);
		};
};



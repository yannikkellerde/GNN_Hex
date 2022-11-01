#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <ATen/ATen.h>
#include "hex_board_game.cpp"
#include "i_dont_need_boost.cpp"
#include "util.cpp"
#include <math.h>
using namespace std;

int init_time = 0;
int feat_time = 0;
int ei_time = 0;
int different_time = 0;


struct PropertyStruct{
#ifdef FOR_INFERENCE
	int board_location;
#endif
	bool removed;
};

enum Onturn {noplayer,maker,breaker};
enum VProps {t1connect,t2connect,removed,board_location};


bool check_if_same(Graph& graph, int v1, int v2){
	if (graph.vprops[t1connect][v1]!=graph.vprops[t1connect][v2] || graph.vprops[t2connect][v1]!=graph.vprops[t2connect][v2]){
		return false;
	}
	Neighbors neigh1 = graph.adjacent_vertices(v1);
	Neighbors neigh2 = graph.adjacent_vertices(v2);
	if (neigh1.second-neigh1.first!=neigh2.second-neigh2.first){
		return false;
	}
	for (;neigh1.first!=neigh1.second;++neigh1.first){
		int n1 = *neigh1.first;
		if (n1 != v2){
			vector<int>::iterator pi = neigh2.first;
			bool found = false;
			for (;pi!=neigh2.second;++pi){
				if (*pi==n1){
					found = true;
					break;
				}
			}
			if (!found){
				return false;
			}
		}
	}
	return true;
}

bool is_fully_connected(Graph& g,int vert, int ignore){
	Neighbors neigh = g.adjacent_vertices(vert);
	if (g.vprops[t1connect][vert]&&g.vprops[t2connect][vert]){
		return false;
	}
	if (g.vprops[t1connect][vert]||g.vprops[t2connect][vert]){
		if (neigh.second - neigh.first > 1){
			return false;
		}
		else if (neigh.second - neigh.first==1){
			return *neigh.first==ignore;
		}
		else{
			return true;
		}
	}
	for (;neigh.first!=neigh.second;++neigh.first){
		int v1 = *neigh.first;
		if (v1!=ignore){
			for (vector<int>::iterator it2 = neigh.first+1;it2!=neigh.second;++it2){
				int v2 = *it2;
				if (v2!=ignore && !g.edge_exists(v1,v2)){
					return false;
				}
			}
		}
	}
	return true;
}

bool is_fully_connected(Graph& g,Neighbors& neigh, int ignore){
	for (;neigh.first!=neigh.second;++neigh.first){
		int v1 = *neigh.first;
		if (v1!=ignore){
			for (vector<int>::iterator it2 = neigh.first+1;it2!=neigh.second;++it2){
				int v2 = *it2;
				if (v2!=ignore && !g.edge_exists(v1,v2)){
					return false;
				}
			}
		}
	}
	return true;
}

bool is_fully_connected(Graph& g,Neighbors& neigh){
	for (;neigh.first!=neigh.second;++neigh.first){
		for (vector<int>::iterator it2 = neigh.first+1;it2!=neigh.second;++it2){
			if (!g.edge_exists(*neigh.first,*it2)){
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
		Onturn onturn=maker;
		int board_size = S;
		Hex_board<S> board;
		bool maker_won=false;
#ifdef FOR_INFERENCE
		map<int,int> response_set_maker;
		map<int,int> response_set_breaker;
#endif

		Node_switching_game (){
			reset();
		};
		Node_switching_game (Graph& g){
			graph = g;
		};
		Node_switching_game (std::vector<torch::jit::IValue> &data){
			// Convert model repr back to graph
			torch::Tensor node_features = data[0].toTensor().cpu();
			torch::Tensor edge_index = data[1].toTensor().cpu();
			graph = Graph(node_features.size(0));
			graph.add_vprop(0); // t1connect
			graph.add_vprop(0); // t2connect
			graph.add_vprop(0); // removed
			graph.add_vprop(0); // board_location
			for (int i=0;i<graph.num_vertices;i++){
				graph.vprops[board_location][i] = i;
				if (node_features[i][0].item<float>()==1.){
					graph.vprops[t1connect][i] = 1;
				}
				if (node_features[i][1].item<float>()==1.){
					graph.vprops[t2connect][i] = 1;
				}
			}
			for (int i=0;i<edge_index.size(1);i++){
				long source = edge_index[0][i].item<long>();
				long target = edge_index[1][i].item<long>();
				graph.add_edge(source,target);
			}
		}
		Node_switching_game (Hex_board<S>& from_board):board(from_board){
			reset();
		};

		void reset(){
#ifdef FOR_INFERENCE
			response_set_maker = map<int,int>();
			response_set_breaker = map<int,int>();
#endif
			onturn = maker;
			graph = Graph(board.num_squares);
			graph.add_vprop(0); // t1connect
			graph.add_vprop(0); // t2connect
			graph.add_vprop(0); // removed
			graph.add_vprop(0); // board_location
			for (int i=0;i<board.num_squares;i++){
				graph.vprops[board_location][i] = i;
				if (i<board.size){
					graph.vprops[t1connect][i] = 1;
				}
				if (floor(i/board.size)==board.size-1){
					graph.vprops[t2connect][i] = 1;
				}
				if (i%board.size>0 && board.size<=i && i<=board.num_squares-board.size){
					graph.add_edge(i,i-1);
				}
				if (i>=board.size){
					graph.add_edge(i,i-board.size);
					if (i%board.size!=board.size-1){
						graph.add_edge(i,i+1-board.size);
					}
				}
			}
		}

		int vertex_from_board_location(int bl){
				int i=0;
				for (vector<int>::iterator vs = graph.vprops[board_location].begin();vs!=graph.vprops[board_location].end();++vs,++i){
					int v = *vs;
					if (v == board_location){
						return i;
					}
				}
				return -1;
		}

#ifdef FOR_INFERENCE
		int get_response(int bloc,bool for_maker){
			if (response_set_maker.find(bloc)!=response_set_maker.end()){
				if (for_maker){
					return response_set_maker[bloc];
				}
				else{
					int otherside = response_set_maker[bloc];
					response_set_maker.erase(otherside);
					response_set_maker.erase(bloc);
					return -1;
				}
			}
			else if (response_set_breaker.find(bloc)!=response_set_breaker.end()){
				if (!for_maker){
					return response_set_breaker[bloc];
				}
				else{
					int otherside = response_set_breaker[bloc];
					response_set_breaker.erase(otherside);
					response_set_breaker.erase(bloc);
					return -1;
				}
			}
			return -1;
		}
#endif

		void switch_onturn(){
			onturn = onturn==maker?breaker:maker;
		}

		set<int> fix_terminal_connections(int vertex, VProps conn_prop){
			set<int> change_set;
			for (Neighbors neigh = graph.adjacent_vertices(vertex);neigh.first!=neigh.second;++neigh.first){
				int v1 = *neigh.first;
				for (Neighbors neigh2 = graph.adjacent_vertices(v1);neigh2.first!=neigh2.second;++neigh2.first){
					int v2 = *neigh2.first;
					if (v2!=vertex && graph.vprops[conn_prop][v2]){
						graph.delete_edge(v1,v2); //TODO: Watch out, this may destroy iterators.
						change_set.insert(v1);
						change_set.insert(v2);
					}
				}
			}
			return change_set;
		}

		void remove_marked_nodes(){
			vector<int>::iterator rem = graph.vprops[removed].end()-1;
			for (int i=graph.num_vertices-1;i>=0;--i,--rem){
				if (*rem){
					graph.remove_vertex(i);
				}
			}
		}

		int get_random_action(){
			vector<int> actions = get_actions();
			/* return *select_randomly(actions.begin(),actions.end()); // unbiased, non-repeatable*/
			return repeatable_random_choice(actions);
		}

		vector<int> get_actions(){
			vector <int> res(graph.num_vertices);
			iota(res.begin(),res.end(),0);
			return res;
		}


		set<int> make_move(int vertex, bool do_force_color=false, Onturn force_color=noplayer,bool do_remove_dead_and_captured=false,bool only_mark_removed=false){
			assert(vertex<graph.num_vertices);
			set<int> change_set;
			Onturn player = do_force_color?force_color:onturn;
			if (!do_force_color){
				switch_onturn();
			}
			if (player==maker){
				bool t1_infect = graph.vprops[t1connect][vertex];
				bool t2_infect = graph.vprops[t2connect][vertex];
				if (t1_infect && t2_infect){
					maker_won = true;
					return change_set; // No need to continue, if game is already won.
				}
				for (Neighbors neigh = graph.adjacent_vertices(vertex);neigh.first!=neigh.second;++neigh.first){
					int v1 = *neigh.first;
					if (t1_infect){
						graph.vprops[t1connect][v1] = true;
					}
					else if (t2_infect){
						graph.vprops[t2connect][v1] = true;
					}
					else{
						for (vector<int>::iterator it2 = neigh.first+1;it2!=neigh.second;++it2){
							int v2 = *it2;
							if (!((graph.vprops[t1connect][v1])&&(graph.vprops[t1connect][v2])||
										(graph.vprops[t2connect][v1])&&(graph.vprops[t2connect][v2]))){
								graph.add_edge(v1,v2); // Problem: This destroys vertex iterator
							}
						}
					}
				}
				if (t1_infect||t2_infect){
					change_set = fix_terminal_connections(vertex,t1_infect?t1connect:t2connect);
				}
			}
			if (do_remove_dead_and_captured){
				Neighbors neigh = graph.adjacent_vertices(vertex);
				change_set.insert(neigh.first,neigh.second);
			}
			graph.clear_vertex(vertex);
			if (do_remove_dead_and_captured || only_mark_removed){
				graph.vprops[removed][vertex] = 1;
			}
			else{
				graph.remove_vertex(vertex);
			}
			if (do_remove_dead_and_captured){
				remove_dead_and_captured(change_set);
				remove_marked_nodes();
			}
			return change_set;
		}

		void remove_dead_and_captured(set<int> &consider_set){
			bool can_continue,is_dead;
			int v1;
			set<int> big_set, change_set;
			Neighbors neigh_neigh,tmp_neigh,neighbors,neigh_neigh_neigh;
			for (int vertex : consider_set){
				/* cout << vertex << endl; */
				is_dead = true;
				can_continue = false;
				if (graph.vprops[removed][vertex]){
					continue;
				}
				neighbors = graph.adjacent_vertices(vertex);
				
				if (neighbors.first!=neighbors.second){
					neigh_neigh = graph.adjacent_vertices(*neighbors.first);
					for (;neigh_neigh.first<=neigh_neigh.second;++neigh_neigh.first){
						if (neigh_neigh.first==neigh_neigh.second){
							v1 = *neighbors.first;
						}
						else{
							v1 = *neigh_neigh.first;
						}
						if (v1==vertex){
							continue;
						}
						if (check_if_same(graph,v1,vertex)){
#ifdef FOR_INFERENCE
							response_set_maker[graph.vprops[board_location][v1]]=graph.vprops[board_location][vertex];
							response_set_maker[graph.vprops[board_location][vertex]]=graph.vprops[board_location][v1];
#endif
								/* tmp_neigh = adjacent_vertices(vertex,graph);       // These are not */
								/* big_set.insert(tmp_neigh.first,tmp_neigh.second);  // in the python original */
							/* cout << "maker captured " << v1 << vertex << endl; */
							neigh_neigh_neigh = graph.adjacent_vertices(v1);
							big_set.insert(neigh_neigh_neigh.first,neigh_neigh_neigh.second);
							graph.vprops[removed][v1] = 1;
							graph.clear_vertex(v1);
							change_set = make_move(vertex,true,maker,false,true);
							big_set.insert(change_set.begin(),change_set.end());
							can_continue = true;
							break;
						}
					}
				}
				if (can_continue){
					continue;
				}
				if (graph.vprops[t1connect][vertex]&&graph.vprops[t2connect][vertex]){
					is_dead = false;
				}
				else if ((graph.vprops[t1connect][vertex]||graph.vprops[t2connect][vertex])&&graph.num_neighbors(vertex)>0){
					is_dead = false;
				}

				for (neighbors = graph.adjacent_vertices(vertex);neighbors.first!=neighbors.second;++neighbors.first){
					v1 = *neighbors.first;
					for (vector<int>::iterator it2 = neighbors.first+1;it2!=neighbors.second;++it2){
						if (is_dead&&graph.edge_exists(*neighbors.first,*it2)){
							is_dead = false;
						}
					}
					if (is_fully_connected(graph,vertex,v1) && is_fully_connected(graph,v1,vertex)){
#ifdef FOR_INFERENCE
						response_set_breaker[graph[v1].board_location]=graph[vertex].board_location;
						response_set_breaker[graph[vertex].board_location]=graph[v1].board_location;
#endif
						/* cout << "breaker captured " << v1 << " " << vertex << endl; */
						tmp_neigh = graph.adjacent_vertices(vertex);
						neigh_neigh = graph.adjacent_vertices(v1);
						big_set.insert(tmp_neigh.first,tmp_neigh.second);
						big_set.insert(neigh_neigh.first,neigh_neigh.second);
						graph.clear_vertex(v1);
						graph.clear_vertex(vertex);
						graph.vprops[removed][v1] = 1;
						graph.vprops[removed][vertex] = 1;
						can_continue = true;
						break;
					}
				}
				if (is_dead && !can_continue){
					/* cout << "node is dead" << endl; */
					neighbors = graph.adjacent_vertices(vertex);
					big_set.insert(neighbors.first,neighbors.second);
					graph.vprops[removed][vertex] = 1; // node is dead
					graph.clear_vertex(vertex);
					continue;
				}
			}
			if (!big_set.empty()){
				remove_dead_and_captured(big_set);
			}
		}
		Onturn who_won(){
			int src=0;
			bool found1,found2,not_again;
			if (maker_won){
				return maker;
			}
			bool visited[graph.num_vertices];
			bool ever_visited[graph.num_vertices];
			while (true){
				fill(visited,visited+graph.num_vertices,false);
				visited[src] = true;
				std::stack<int> next;
				found1 = graph.vprops[t1connect][src];
				found2 = graph.vprops[t2connect][src];
				next.push(src);

				while(!next.empty()) {
					int cv = next.top();
					next.pop();
					for (Neighbors neigh = graph.adjacent_vertices(cv);neigh.first!=neigh.second;++neigh.first){
						int nv = *neigh.first;
						if (!visited[nv]){
							if (graph.vprops[t1connect][nv]){
								found1 = true;
							}
							if (graph.vprops[t2connect][nv]){
								found2 = true;
							}
							if (found1&&found2){
								return noplayer;
							}
							visited[nv] = true;
							ever_visited[nv] = true;
							next.push(nv);
						}
					}
				}
				if (found1&&found2){
					return noplayer;
				}
				not_again = true;
				for (int i=src+1;i<graph.num_vertices;i++){
					if (!ever_visited[i]){
						src = i;
						not_again=false;
						break;
					}
				}
				if (not_again){
					return breaker;
				}
			}
		}

		vector<string> get_grid_layout(){
			double scale;
			vector<string> position_array(graph.num_vertices);
			scale = 1.;
			const double xstart = 0;
			const double ystart = 0;
			const double xend = xstart+1.5*(board_size-1)*scale;
			const double yend = ystart+sqrt(3./4.)*(board_size-1)*scale;
			for (int i=0;i<graph.num_vertices;i++){
				int bi = graph.vprops[board_location][i];
				int row = floor(bi/board_size);
				int col = bi%board_size;
				position_array[i] = to_string((xstart+(0.5*col+row)*scale))+","+to_string((ystart + (sqrt(3./4.)*col)*scale))+"!";
			}
			return position_array;
		}

		vector<string> get_colors(){
			vector<string> color_array(graph.num_vertices);
			for (int i=0;i<graph.num_vertices;i++){
				if (graph.vprops[t1connect][i]){
					if (graph.vprops[t2connect][i]){
						color_array[i] = "brown";
					}
					else{
						color_array[i] = "green";
					}
				}
				else if (graph.vprops[t2connect][i]){
					color_array[i] = "red";
				}
				else{
					color_array[i] = "black";
				}
			}
			return color_array;
		}

		void graphviz_me (string fname){
			vector<string> color_map = get_colors();
			vector<string> pos_map = get_grid_layout();
			vector<pair<string,vector<string>>> props;
			props.push_back(pair<string,vector<string>>("color",color_map));
			props.push_back(pair<string,vector<string>>("pos",pos_map));

			graph.graphviz_me(props,fname,true);
		};

	/* std::vector<torch::jit::IValue> convert_graph(torch::Device &device){ */

	/* 	auto start = chrono::high_resolution_clock::now(); */
	/* 	int n = graph.num_vertices; */
	/* 	torch::TensorOptions options_long = torch::TensorOptions().dtype(torch::kLong).device(device); */
	/* 	torch::TensorOptions options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device); */
	/* 	torch::Tensor node_features = torch::zeros({n-2,3},options_float); */
	/* 	auto stop = chrono::high_resolution_clock::now(); */
	/* 	auto duration = chrono::duration_cast<chrono::microseconds>(stop - start); */
	/* 	init_time+=duration.count(); */
	/* 	start = chrono::high_resolution_clock::now(); */
	/* 	Neighbors neigh = adjacent_vertices(terminal1,graph); //mark neighbors of terminals in node features */
	/* 	int num_t1_neigh = neigh.second-neigh.first; */
	/* 	for (;neigh.first!=neigh.second;++neigh.first){ */
	/* 		node_features[(*neigh.first)-2][0] = 1.; */
	/* 	} */
	/* 	neigh = adjacent_vertices(terminal2,graph); */
	/* 	int num_t2_neigh = neigh.second-neigh.first; */
	/* 	for (;neigh.first!=neigh.second;++neigh.first){ */
	/* 		node_features[(*neigh.first)-2][1] = 1.; */
	/* 	} */
	/* 	if (onturn==maker){ */
	/* 		for (int i=0;i<n-2;++i){ */
	/* 			node_features[i][2] = 1; */
	/* 		} */
	/* 	} */
	/* 	torch::Tensor edge_index = torch::empty({2,((int)num_edges(graph)-(int)num_t1_neigh-(int)num_t2_neigh)*2},options_long); */
	/* 	stop = chrono::high_resolution_clock::now(); */
	/* 	duration = chrono::duration_cast<chrono::microseconds>(stop - start); */
	/* 	feat_time+=duration.count(); */
	/* 	start = chrono::high_resolution_clock::now(); */

	/* 	vector<int> source_vec; */
	/* 	vector<int> targ_vec; */

	/* 	graph_traits<Graph>::edge_iterator ei, ei_end; */
	/* 	int ind = 0; */
	/* 	for (boost::tie(ei, ei_end) = edges(graph); ei != ei_end; ++ei){ */
	/* 		int s = source(*ei,graph); */
	/* 		int t = target(*ei,graph); */
	/* 		if (s>1 && t>1){ */
	/* 			edge_index[0][ind] = s-2;  // double, because undirected graph */
	/* 			edge_index[1][ind] = t-2;  // -2, because we throw out terminals */
	/* 			ind++; */
	/* 			edge_index[1][ind] = s-2; */
	/* 			edge_index[0][ind] = t-2; */
	/* 			ind++; */
	/* 			source_vec.push_back(s-2); */
	/* 			source_vec.push_back(t-2); */
	/* 			targ_vec.push_back(t-2); */
	/* 			targ_vec.push_back(s-2); */
	/* 		} */
	/* 	} */
	/* 	stop = chrono::high_resolution_clock::now(); */
	/* 	duration = chrono::duration_cast<chrono::microseconds>(stop - start); */
	/* 	ei_time+=duration.count(); */

	/* 	start = chrono::high_resolution_clock::now(); */
	/* 	torch::Tensor source_tensor = torch::tensor(source_vec,options_long); */
	/* 	torch::Tensor target_tensor = torch::tensor(targ_vec,options_long); */
	/* 	stop = chrono::high_resolution_clock::now(); */
	/* 	duration = chrono::duration_cast<chrono::microseconds>(stop - start); */
	/* 	different_time+=duration.count(); */
	/* 	assert(source_tensor.size(0)==edge_index.size(1)); */

	/* 	std::vector<torch::jit::IValue> parts; */
	/* 	parts.push_back(node_features); */
	/* 	parts.push_back(edge_index); */
	/* 	return parts; */
	/* } */
};



#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_iterator.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <ATen/ATen.h>
#include "hex_board_game.cpp"
#include "util.cpp"
#include <math.h>
#include <boost/config.hpp>
using namespace std;
using namespace boost;
using namespace torch::indexing;

int init_time = 0;
int feat_time = 0;
int ei_time = 0;
int different_time = 0;


struct PropertyStruct{
	int board_location;
	bool t1connect;
	bool t2connect;
	bool removed;
};

typedef adjacency_list<vecS, vecS, undirectedS, PropertyStruct, no_property> Graph;
typedef pair<int, int> Edge;
typedef property_map<Graph, vertex_index_t>::type IndexMap;
typedef iterator_property_map<string*,IndexMap> StringPropMap;
typedef typename Graph::vertex_descriptor Vertex;
typedef set<string> labels ;
typedef pair<typename Graph::adjacency_iterator,typename Graph::adjacency_iterator> Neighbors; 
typedef pair<graph_traits<Graph>::vertex_iterator, graph_traits<Graph>::vertex_iterator> Viters;
enum Onturn {noplayer,maker,breaker};


bool check_if_same(Graph& g, int v1, int v2){
	if (g[v1].t1connect!=g[v2].t1connect || g[v1].t2connect!=g[v2].t2connect){
		return false;
	}
	Neighbors neigh1 = adjacent_vertices(v1,g);
	Neighbors neigh2 = adjacent_vertices(v2,g);
	if (neigh1.second-neigh1.first!=neigh2.second-neigh2.first){
		return false;
	}
	for (;neigh1.first!=neigh1.second;++neigh1.first){
		int n1 = *neigh1.first;
		if (n1 != v2){
			auto pi = neigh2.first;
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

bool is_fully_connected(Graph& g,Neighbors& neigh, int ignore){
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

bool is_fully_connected(Graph& g,int vert,int ignore){
	if (g[vert].t1connect&&g[vert].t2connect){
		return false;
	}
	Neighbors neigh = adjacent_vertices(vert,g);
	if (g[vert].t1connect||g[vert].t2connect){
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
	return is_fully_connected(g,neigh,ignore);
}


bool is_fully_connected(Graph& g,Neighbors& neigh){
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
			vi_map=get(vertex_index, graph);
		};
		Node_switching_game (std::vector<torch::jit::IValue> &data){
			// Convert model repr back to graph
			torch::Tensor node_features= data[0].toTensor().cpu();
			torch::Tensor edge_index = data[1].toTensor().cpu();
			graph = Graph(node_features.size(0));
			for (int i=0;i<num_vertices(graph);i++){
				graph[i].t1connect = false;
				graph[i].t2connect = false;
				graph[i].board_location = i;  // This only works if in starting position
				if (node_features[i][0].item<float>()==1.){
					graph[i].t1connect = true;
				}
				if (node_features[i][1].item<float>()==1.){
					graph[i].t2connect = true;
				}
				graph[i].removed = false;
			}
			for (int i=0;i<edge_index.size(1);i++){
				long source = edge_index[0][i].item<long>();
				long target = edge_index[1][i].item<long>();
				if (!edge(source,target,graph).second){
					add_edge(source,target,graph);
				}
			}
			vi_map=get(vertex_index, graph);
		}
		Node_switching_game (Hex_board<S>& from_board):board(from_board){
			reset();
		};

		void reset(){
#ifdef FOR_INFERENCE
			response_set_maker = map<int,int>();
			response_set_breaker = map<int,int>();
#endif
			maker_won = false;
			onturn = maker;
			graph = Graph(board.num_squares);
			for (int i=0;i<board.num_squares;i++){
				graph[i].board_location = i;
				graph[i].t1connect = false;
				graph[i].t2connect = false;
				graph[i].removed = false;
				if (i<board.size){
					graph[i].t1connect = true;
				}
				if (floor(i/board.size)==board.size-1){
					graph[i].t2connect = true;
				}
				if (i%board.size>0 && board.size<=i && i<=board.num_squares-board.size){
					add_edge(i,i-1,graph);
				}
				if (i>=board.size){
					add_edge(i,i-board.size,graph);
					if (i%board.size!=board.size-1){
						add_edge(i,i+1-board.size,graph);
					}
				}
			}
			vi_map=get(vertex_index, graph);
		}

		int vertex_from_board_location(int board_location){
				for (Viters vs = vertices(graph);vs.first!=vs.second;++vs.first){
					int v = *vs.first;
					if (graph[v].board_location == board_location){
						return v;
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

		Node_switching_game copy(){
			Graph new_graph;
			copy_graph(graph,new_graph);
			return Node_switching_game<S>(new_graph);
		}

		void switch_onturn(){
			onturn = onturn==maker?breaker:maker;
		}

		set<int> fix_terminal_connections(int vertex, bool is_t1){
			set<int> change_set;
			vector<pair<int,int>> to_remove;
			for (Neighbors neigh = adjacent_vertices(vertex,graph);neigh.first!=neigh.second;++neigh.first){
				int v1 = *neigh.first;
				for (Neighbors neigh2 = adjacent_vertices(v1,graph);neigh2.first!=neigh2.second;++neigh2.first){
					int v2 = *neigh2.first;
					if (v2!=vertex && (is_t1?graph[v2].t1connect:graph[v2].t2connect)){
						to_remove.push_back(pair<int,int>(v1,v2)); //otherwise iterators get messed up
						change_set.insert(v1);
						change_set.insert(v2);
					}
				}
			}
			for (auto p=to_remove.begin();p!=to_remove.end();++p){
				remove_edge(p->first,p->second,graph);
			}
			return change_set;
		}

		void remove_marked_nodes(){
			for (int i=num_vertices(graph)-1;i>=0;--i){
				if (graph[i].removed){
					remove_vertex(i,graph);
				}
			}
		}

		int get_random_action(){
			vector<int> actions = get_actions();
			/* return *select_randomly(actions.begin(),actions.end()); // unbiased, non-repeatable*/
			return repeatable_random_choice(actions);
		}

		vector<int> get_actions(){
			vector <int> res(num_vertices(graph));
			iota(res.begin(),res.end(),0);
			return res;
		}

		set<int> make_move(int vertex, bool do_force_color=false, Onturn force_color=noplayer,bool do_remove_dead_and_captured=false,bool only_mark_removed=false){
			assert(vertex<num_vertices(graph));
			set<int> change_set;
			Onturn player = do_force_color?force_color:onturn;
			if (!do_force_color){
				switch_onturn();
			}
			if (player==maker){
				if (graph[vertex].t1connect&&graph[vertex].t2connect){
					maker_won=true;
					return change_set;
				}
				for (Neighbors neigh = adjacent_vertices(vertex,graph);neigh.first!=neigh.second;++neigh.first){
					int v1 = *neigh.first;
					if (graph[vertex].t1connect){
						graph[v1].t1connect = true;
					}
					else if (graph[vertex].t2connect){
						graph[v1].t2connect = true;
					}
					else{
						for (Graph::adjacency_iterator it2 = neigh.first+1;it2!=neigh.second;++it2){
							Vertex v2 = *it2;
							if (!((graph[v1].t1connect&&graph[v2].t1connect)||
										(graph[v1].t2connect&&graph[v2].t2connect)||
										edge(v1,v2,graph).second)){
								add_edge(v1,v2,graph);
							}
						}
					}
				}
				if (graph[vertex].t1connect||graph[vertex].t2connect){
					change_set = fix_terminal_connections(vertex,graph[vertex].t1connect);
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

		void remove_dead_and_captured(set<int> &consider_set){
			bool can_continue,is_dead;
			int v1;
			set<int> big_set, change_set;
			Neighbors neigh_neigh,tmp_neigh,neighbors,neigh_neigh_neigh;
			for (int vertex : consider_set){
				/* cout << vertex << endl; */
				is_dead = true;
				can_continue = false;
				if (graph[vertex].removed){
					continue;
				}
				neighbors = adjacent_vertices(vertex,graph);

				
				if (neighbors.first!=neighbors.second){
					neigh_neigh = adjacent_vertices(*neighbors.first,graph);
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
							response_set_maker[graph[v1].board_location]=graph[vertex].board_location;
							response_set_maker[graph[vertex].board_location]=graph[v1].board_location;
#endif
								/* tmp_neigh = adjacent_vertices(vertex,graph);       // These are not */
								/* big_set.insert(tmp_neigh.first,tmp_neigh.second);  // in the python original */
							/* cout << "maker captured " << v1 << vertex << endl; */
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
				if (can_continue){
					continue;
				}
				neighbors = adjacent_vertices(vertex,graph);
				if (graph[vertex].t1connect&&graph[vertex].t2connect){
					is_dead = false;
				}
				else if ((graph[vertex].t1connect||graph[vertex].t2connect)&&(neighbors.second-neighbors.first)>0){
					is_dead = false;
				}
				for (;neighbors.first!=neighbors.second;++neighbors.first){
					v1 = *neighbors.first;
					for (Graph::adjacency_iterator it2 = neighbors.first+1;it2!=neighbors.second;++it2){
						if (is_dead&&!edge(*neighbors.first,*it2,graph).second){
							is_dead = false;
						}
					}
					if (is_fully_connected(graph,vertex,v1) && is_fully_connected(graph,v1,vertex)){
#ifdef FOR_INFERENCE
						response_set_breaker[graph[v1].board_location]=graph[vertex].board_location;
						response_set_breaker[graph[vertex].board_location]=graph[v1].board_location;
#endif
						/* cout << "breaker captured " << v1 << " " << vertex << endl; */
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
				if (is_dead && !can_continue){
					/* cout << "node is dead" << endl; */
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


		Onturn who_won(){
			int src=0;
			bool found1,found2,not_again;
			bool visited[num_vertices(graph)];
			bool ever_visited[num_vertices(graph)];
			if (maker_won){
				return maker;
			}
			if (num_vertices(graph)==0){
				return breaker;
			}
			while (true){
				fill(visited,visited+num_vertices(graph),false);
				visited[src] = true;
				std::stack<int> next;
				found1 = graph[src].t1connect;
				found2 = graph[src].t2connect;
				next.push(src);

				while(!next.empty()) {
					int cv = next.top();
					next.pop();
					for (Neighbors neigh = adjacent_vertices(cv,graph);neigh.first!=neigh.second;++neigh.first){
						int nv = *neigh.first;
						if (!visited[nv]){
							if (graph[nv].t1connect){
								found1 = true;
							}
							if (graph[nv].t2connect){
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
				for (int i=src+1;i<num_vertices(graph);i++){
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

		StringPropMap get_grid_layout(){
			double scale;
			string* position_array;
			position_array = new string[num_vertices(graph)];
			scale = 1.;
			const double xstart = 0;
			const double ystart = 0;
			const double xend = xstart+1.5*(board_size-1)*scale;
			const double yend = ystart+sqrt(3./4.)*(board_size-1)*scale;
			for (int i=0;i<num_vertices(graph);i++){
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
			for (int i=0;i<num_vertices(graph);i++){
				if (graph[i].t1connect){
					if (graph[i].t2connect){
						color_array[i] = "brown";
					}
					else{
						color_array[i] = "green";
					}
				}
				else if (graph[i].t2connect){
					color_array[i] = "red";
				}
				else{
					color_array[i] = "black";
				}
			}
			StringPropMap pos_map = make_iterator_property_map(color_array,vi_map);
			return pos_map;
		}

		void graphviz_me (string out){
			ofstream my_file;
			my_file.open(out);
			graphviz_me(my_file);
			my_file.close();
		}

		void graphviz_me (ostream &out){
			StringPropMap color_map = get_colors();
			/* cout << "here " << num_vertices(graph) << endl; */
			dynamic_properties dp;
			dp.property("color", color_map);
			StringPropMap pos_map = get_grid_layout();
			dp.property("pos",pos_map);
			dp.property("node_id", get(vertex_index, graph));
			write_graphviz_dp(out,graph,dp);
		};

	std::vector<torch::jit::IValue> convert_graph(torch::Device &device){
		Neighbors neigh;
		auto start = chrono::high_resolution_clock::now();
		int n = num_vertices(graph);
		torch::TensorOptions options_long = torch::TensorOptions().dtype(torch::kLong).device(device);
		torch::TensorOptions options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
		torch::Tensor node_features = torch::zeros({n,3},options_float);
		auto stop = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
		init_time+=duration.count();
		start = chrono::high_resolution_clock::now();
		for (int vi=0;vi<num_vertices(graph);++vi){
			node_features[vi][0] = graph[vi].t1connect;
			node_features[vi][1] = graph[vi].t2connect;
		}
		torch::Tensor edge_index = torch::empty({2,(int)num_edges(graph)*2},options_long);
		stop = chrono::high_resolution_clock::now();
		duration = chrono::duration_cast<chrono::microseconds>(stop - start);
		feat_time+=duration.count();
		start = chrono::high_resolution_clock::now();

		int ind = 0;
		for (int vi=0;vi<num_vertices(graph);++vi){
			neigh = adjacent_vertices(vi,graph);
			edge_index.index_put_({0,Slice(ind,ind+(neigh.second-neigh.first))},vi);
			/* cout << ind << " " << ind+(neigh.second-neigh.first) << endl; */
			edge_index.index_put_({1,Slice(ind,ind+(neigh.second-neigh.first))},torch::tensor(vector<int>(neigh.first,neigh.second)));
			ind+=neigh.second-neigh.first;
		}
		stop = chrono::high_resolution_clock::now();
		duration = chrono::duration_cast<chrono::microseconds>(stop - start);
		ei_time+=duration.count();

		std::vector<torch::jit::IValue> parts;
		parts.push_back(node_features);
		parts.push_back(edge_index);
		return parts;
	}
};



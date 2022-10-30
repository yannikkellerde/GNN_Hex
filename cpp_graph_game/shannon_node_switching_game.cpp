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

typedef adjacency_list<vecS, vecS, undirectedS, PropertyStruct, no_property> Graph;
typedef pair<int, int> Edge;
typedef property_map<Graph, vertex_index_t>::type IndexMap;
typedef iterator_property_map<string*,IndexMap> StringPropMap;
typedef typename Graph::vertex_descriptor Vertex;
typedef set<string> labels ;
typedef pair<typename Graph::adjacency_iterator,typename Graph::adjacency_iterator> Neighbors; 
typedef pair<graph_traits<Graph>::vertex_iterator, graph_traits<Graph>::vertex_iterator> Viters;
enum Onturn {noplayer,maker,breaker};
enum Teminals {terminal1, terminal2};

bool path_exists(Graph& g, Vertex src, Vertex dest) {
  bool visited[num_vertices(g)];
	fill(visited,visited+num_vertices(g),false);
  visited[src] = true;
  std::stack<Vertex> next;
  next.push(src);

  while(!next.empty()) {
    Vertex cv = next.top();
    next.pop();
		for (Neighbors neigh = adjacent_vertices(cv,g);neigh.first!=neigh.second;++neigh.first){
			Vertex nv = *neigh.first;
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

bool check_if_same(Graph& g, Neighbors& neigh1, Neighbors& neigh2, int ignore1, int ignore2){
	if (neigh1.second-neigh1.first!=neigh2.second-neigh2.first){
		return false;
	}
	for (;neigh1.first!=neigh1.second;++neigh1.first){
		int v1 = *neigh1.first;
		if (v1 != ignore1){
			auto pi = neigh2.first;
			bool found = false;
			for (;pi!=neigh2.second;++pi){
				if (*pi==v1){
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
			graph = Graph(node_features.size(0)+2);
			for (int i=0;i<num_vertices(graph);i++){
				if (i>1){
#ifdef FOR_INFERENCE
					graph[i].board_location = i-2;  // This only works if in starting position
#endif
					if (node_features[i-2][0].item<float>()==1.){
						add_edge(i,terminal1,graph);
					}
					if (node_features[i-2][1].item<float>()==1.){
						add_edge(i,terminal2,graph);
					}
				}
				graph[i].removed = false;
			}
			for (int i=0;i<edge_index.size(1);i++){
				long source = edge_index[0][i].item<long>()+2;
				long target = edge_index[1][i].item<long>()+2;
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
			onturn = maker;
			graph = Graph(board.num_squares+2);
			graph[terminal1].removed = false;
			graph[terminal2].removed = false;
#ifdef FOR_INFERENCE
			graph[terminal1].board_location = -1;
			graph[terminal2].board_location = -1;
#endif
			for (int i=0;i<board.num_squares;i++){
#ifdef FOR_INFERENCE
				graph[i+2].board_location = i;
#endif
				graph[i+2].removed = false;
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
		}

#ifdef FOR_INFERENCE
		int vertex_from_board_location(int board_location){
				for (Viters vs = vertices(graph);vs.first!=vs.second;++vs.first){
					int v = *vs.first;
					if (graph[v].board_location == board_location){
						return v;
					}
				}
				return -1;
		}

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
				}
			}
		}

		int get_random_action(){
			vector<int> actions = get_actions();
			/* return *select_randomly(actions.begin(),actions.end()); // unbiased, non-repeatable*/
			return repeatable_random_choice(actions);
		}

		vector<int> get_actions(){
			vector <int> res(num_vertices(graph)-2);
			iota(res.begin(),res.end(),2);
			return res;
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
				if (graph[vertex].removed || vertex<2){
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
						if (v1<2 || v1==vertex){
							continue;
						}
						neigh_neigh_neigh = adjacent_vertices(v1,graph);
						neighbors = adjacent_vertices(vertex,graph);
						if (check_if_same(graph,neighbors,neigh_neigh_neigh,v1,vertex)){
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
				for (neighbors = adjacent_vertices(vertex,graph);neighbors.first!=neighbors.second;++neighbors.first){
					v1 = *neighbors.first;
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
					if (is_fully_connected(graph,tmp_neigh,v1) && is_fully_connected(graph,neigh_neigh,vertex)){
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

		set<int> make_move(int vertex, bool do_force_color=false, Onturn force_color=noplayer,bool do_remove_dead_and_captured=false,bool only_mark_removed=false){
			assert(vertex<num_vertices(graph));
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

#ifdef FOR_INFERENCE

		StringPropMap get_grid_layout(){
			double scale;
			string* position_array;
			position_array = new string[num_vertices(graph)];
			scale = 1.;
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
#endif

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
			StringPropMap color_map = get_colors();
			/* cout << "here " << num_vertices(graph) << endl; */
			dynamic_properties dp;
			dp.property("color", color_map);
#ifdef FOR_INFERENCE
			get_grid_layout();
			StringPropMap pos_map = get_grid_layout();
			dp.property("pos",pos_map);
#endif
			dp.property("node_id", get(vertex_index, graph));
			write_graphviz_dp(out,graph,dp);
		};

	std::vector<torch::jit::IValue> convert_graph(torch::Device &device){

		auto start = chrono::high_resolution_clock::now();
		int n = num_vertices(graph);
		torch::TensorOptions options_long = torch::TensorOptions().dtype(torch::kLong).device(device);
		torch::TensorOptions options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
		torch::Tensor node_features = torch::zeros({n-2,3},options_float);
		auto stop = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
		init_time+=duration.count();
		start = chrono::high_resolution_clock::now();
		Neighbors neigh = adjacent_vertices(terminal1,graph); //mark neighbors of terminals in node features
		int num_t1_neigh = neigh.second-neigh.first;
		for (;neigh.first!=neigh.second;++neigh.first){
			node_features[(*neigh.first)-2][0] = 1.;
		}
		neigh = adjacent_vertices(terminal2,graph);
		int num_t2_neigh = neigh.second-neigh.first;
		for (;neigh.first!=neigh.second;++neigh.first){
			node_features[(*neigh.first)-2][1] = 1.;
		}
		if (onturn==maker){
			for (int i=0;i<n-2;++i){
				node_features[i][2] = 1;
			}
		}
		torch::Tensor edge_index = torch::empty({2,((int)num_edges(graph)-(int)num_t1_neigh-(int)num_t2_neigh)*2},options_long);
		stop = chrono::high_resolution_clock::now();
		duration = chrono::duration_cast<chrono::microseconds>(stop - start);
		feat_time+=duration.count();
		start = chrono::high_resolution_clock::now();

		vector<int> source_vec;
		vector<int> targ_vec;

		graph_traits<Graph>::edge_iterator ei, ei_end;
		int ind = 0;
		for (boost::tie(ei, ei_end) = edges(graph); ei != ei_end; ++ei){
			int s = source(*ei,graph);
			int t = target(*ei,graph);
			if (s>1 && t>1){
				edge_index[0][ind] = s-2;  // double, because undirected graph
				edge_index[1][ind] = t-2;  // -2, because we throw out terminals
				ind++;
				edge_index[1][ind] = s-2;
				edge_index[0][ind] = t-2;
				ind++;
				source_vec.push_back(s-2);
				source_vec.push_back(t-2);
				targ_vec.push_back(t-2);
				targ_vec.push_back(s-2);
			}
		}
		stop = chrono::high_resolution_clock::now();
		duration = chrono::duration_cast<chrono::microseconds>(stop - start);
		ei_time+=duration.count();

		start = chrono::high_resolution_clock::now();
		torch::Tensor source_tensor = torch::tensor(source_vec,options_long);
		torch::Tensor target_tensor = torch::tensor(targ_vec,options_long);
		stop = chrono::high_resolution_clock::now();
		duration = chrono::duration_cast<chrono::microseconds>(stop - start);
		different_time+=duration.count();
		assert(source_tensor.size(0)==edge_index.size(1));

		std::vector<torch::jit::IValue> parts;
		parts.push_back(node_features);
		parts.push_back(edge_index);
		return parts;
	}
};



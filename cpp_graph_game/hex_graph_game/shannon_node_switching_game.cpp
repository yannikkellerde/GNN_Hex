#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <math.h>
#include "shannon_node_switching_game.h"
#include "util.h"


bool check_if_same(Graph& graph, int v1, int v2){
	if (graph.fprops[t1connect][v1]!=graph.fprops[t1connect][v2] || graph.fprops[t2connect][v1]!=graph.fprops[t2connect][v2]){
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
	if (g.fprops[t1connect][vert]&&g.fprops[t2connect][vert]){
		return false;
	}
	if (g.fprops[t1connect][vert]||g.fprops[t2connect][vert]){
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

Node_switching_game::Node_switching_game (int board_size):board_size(board_size),board(board_size){
	/* info_string("Hello"); */
	reset();
};
Node_switching_game::Node_switching_game (Graph& g){
	graph = g;
};
Node_switching_game::Node_switching_game (std::vector<torch::Tensor> &data){
	// Convert model repr back to graph
	torch::Tensor node_features = data[0].cpu();
	torch::Tensor edge_index = data[1].cpu();
	graph = Graph(node_features.size(0));
	graph.add_fprop(0.); // t1connect
	graph.add_fprop(0.); // t2connect
	graph.add_lprop(0); // removed
	graph.add_lprop(0); // board_location
	for (int i=0;i<graph.num_vertices;i++){
		graph.lprops[board_location][i] = i;
		if (node_features[i][0].item<float>()==1.){
			graph.fprops[t1connect][i] = 1.;
		}
		if (node_features[i][1].item<float>()==1.){
			graph.fprops[t2connect][i] = 1.;
		}
	}
	for (int i=0;i<edge_index.size(1);i++){
		long source = edge_index[0][i].item<long>();
		long target = edge_index[1][i].item<long>();
		graph.add_edge(source,target);
	}
}
Node_switching_game::Node_switching_game (Hex_board& from_board):board(from_board){
	reset();
};

Node_switching_game::Node_switching_game (Node_switching_game& ref){
	graph = ref.graph;
	maker_won = ref.maker_won;
	response_set_maker = ref.response_set_maker;
	response_set_breaker = ref.response_set_breaker;
	onturn = ref.onturn;
	board_size = ref.board_size;
	board = ref.board;
	move_num = ref.move_num;
}

Node_switching_game * Node_switching_game::clone(){
	return new Node_switching_game(*this);
}

void Node_switching_game::reset(){
	move_num = 0;
	board_size = board.size;
	maker_won=false;
	response_set_maker = map<int,int>();
	response_set_breaker = map<int,int>();
	onturn = maker;
	graph = Graph(board.num_squares);
	graph.add_fprop(0.); // t1connect
	graph.add_fprop(0.); // t2connect
	graph.add_lprop(0); // removed
	graph.add_lprop(0); // board_location
	for (int i=0;i<board.num_squares;i++){
		graph.lprops[board_location][i] = i;
		if (i<board.size){
			graph.fprops[t1connect][i] = 1.;
		}
		if (floor(i/board.size)==board.size-1){
			graph.fprops[t2connect][i] = 1.;
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
uint16_t Node_switching_game::hash_key() { // https://stackoverflow.com/a/27216842
	std::size_t seed = graph.sources.size();
	for(int& i : graph.sources) {
		seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}
	for(int& i : graph.targets) {
		seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}
	return seed;
}

int Node_switching_game::vertex_from_board_location(int bl){
	int i=0;
	for (vector<int>::iterator vs = graph.lprops[board_location].begin();vs!=graph.lprops[board_location].end();++vs,++i){
		int v = *vs;
		if (v == bl){
			return i;
		}
	}
	return -1;
}

int Node_switching_game::get_response(int bloc,bool for_maker){
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

void Node_switching_game::switch_onturn(){
	onturn = onturn==maker?breaker:maker;
}

set<int> Node_switching_game::fix_terminal_connections(int vertex, Fprops conn_prop){
	set<int> change_set;
	vector<pair<int,int>> to_del;
	for (Neighbors neigh = graph.adjacent_vertices(vertex);neigh.first!=neigh.second;++neigh.first){
		int v1 = *neigh.first;
		for (Neighbors neigh2 = graph.adjacent_vertices(v1);neigh2.first!=neigh2.second;++neigh2.first){
			int v2 = *neigh2.first;
			if (v2!=vertex && graph.fprops[conn_prop][v2]){
				to_del.push_back(pair<int,int>(v1,v2));
				change_set.insert(v1);
				change_set.insert(v2);
			}
		}
	}
	for (pair<int,int>del_me:to_del){
		graph.delete_edge(del_me.first,del_me.second);
	}
	return change_set;
}

void Node_switching_game::remove_marked_nodes(){
	vector<int>::iterator rem = graph.lprops[removed].end()-1;
	for (int i=graph.num_vertices-1;i>=0;--i,--rem){
		if (*rem){
			graph.remove_vertex(i);
		}
	}
}

int Node_switching_game::get_random_action(){
	vector<int> actions = get_actions();
	/* return *select_randomly(actions.begin(),actions.end()); // unbiased, non-repeatable*/
	return repeatable_random_choice(actions);
}

vector<int> Node_switching_game::get_actions(){
	vector <int> res(graph.num_vertices);
	iota(res.begin(),res.end(),0);
	return res;
}


set<int> Node_switching_game::make_move(int vertex, bool do_force_color, Onturn force_color,bool do_remove_dead_and_captured,bool only_mark_removed){
	assert(vertex<graph.num_vertices);
	set<int> change_set;
	Onturn player = do_force_color?force_color:onturn;
	if (!do_force_color){
		++move_num;
		switch_onturn();
	}
	if (player==maker){
		bool t1_infect = graph.fprops[t1connect][vertex];
		bool t2_infect = graph.fprops[t2connect][vertex];
		if (t1_infect && t2_infect){
			maker_won = true;
			return change_set; // No need to continue, if game is already won.
		}
		vector<pair<int,int>> to_add;
		for (Neighbors neigh = graph.adjacent_vertices(vertex);neigh.first!=neigh.second;++neigh.first){
			int v1 = *neigh.first;
			if (t1_infect){
				graph.fprops[t1connect][v1] = 1.;
			}
			else if (t2_infect){
				graph.fprops[t2connect][v1] = 1.;
			}
			else{
				for (vector<int>::iterator it2 = neigh.first+1;it2!=neigh.second;++it2){
					int v2 = *it2;
					if (!((graph.fprops[t1connect][v1])&&(graph.fprops[t1connect][v2])||
								(graph.fprops[t2connect][v1])&&(graph.fprops[t2connect][v2]))){
						to_add.push_back(pair<int,int>(v1,v2));
					}
				}
			}
		}
		for (pair<int,int> add_me : to_add){
			graph.add_edge(add_me.first,add_me.second);
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
		graph.lprops[removed][vertex] = 1;
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

void Node_switching_game::remove_dead_and_captured(set<int> &consider_set){
	bool can_continue,is_dead;
	int v1;
	set<int> big_set, change_set;
	Neighbors neigh_neigh,tmp_neigh,neighbors,neigh_neigh_neigh;
	for (int vertex : consider_set){
		is_dead = true;
		can_continue = false;
		if (graph.lprops[removed][vertex]){
			continue;
		}
		/* cout << "considering " << vertex << endl; */
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
					response_set_maker[graph.lprops[board_location][v1]]=graph.lprops[board_location][vertex];
					response_set_maker[graph.lprops[board_location][vertex]]=graph.lprops[board_location][v1];
					/* tmp_neigh = adjacent_vertices(vertex,graph);       // These are not */
					/* big_set.insert(tmp_neigh.first,tmp_neigh.second);  // in the python original */
					/* cout << "maker captured " << v1 << vertex << endl; */
					neigh_neigh_neigh = graph.adjacent_vertices(v1);
					big_set.insert(neigh_neigh_neigh.first,neigh_neigh_neigh.second);
					graph.lprops[removed][v1] = 1;
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
		if (graph.fprops[t1connect][vertex]&&graph.fprops[t2connect][vertex]){
			is_dead = false;
		}
		else if ((graph.fprops[t1connect][vertex]||graph.fprops[t2connect][vertex])&&graph.num_neighbors(vertex)>0){
			is_dead = false;
		}

		for (neighbors = graph.adjacent_vertices(vertex);neighbors.first!=neighbors.second;++neighbors.first){
			v1 = *neighbors.first;
			for (vector<int>::iterator it2 = neighbors.first+1;it2!=neighbors.second;++it2){
				if (is_dead&&!graph.edge_exists(*neighbors.first,*it2)){
					is_dead = false;
				}
			}
			if (is_fully_connected(graph,vertex,v1) && is_fully_connected(graph,v1,vertex)){
				response_set_breaker[graph.lprops[board_location][v1]]=graph.lprops[board_location][vertex];
				response_set_breaker[graph.lprops[board_location][vertex]]=graph.lprops[board_location][v1];
				/* cout << "breaker captured " << v1 << " " << vertex << endl; */
				tmp_neigh = graph.adjacent_vertices(vertex);
				neigh_neigh = graph.adjacent_vertices(v1);
				big_set.insert(tmp_neigh.first,tmp_neigh.second);
				big_set.insert(neigh_neigh.first,neigh_neigh.second);
				graph.clear_vertex(v1);
				graph.clear_vertex(vertex);
				graph.lprops[removed][v1] = 1;
				graph.lprops[removed][vertex] = 1;
				can_continue = true;
				break;
			}
		}
		if (is_dead && !can_continue){
			/* cout << "node is dead " << vertex << endl; */
			neighbors = graph.adjacent_vertices(vertex);
			big_set.insert(neighbors.first,neighbors.second);
			graph.lprops[removed][vertex] = 1; // node is dead
			graph.clear_vertex(vertex);
			continue;
		}
	}
	if (!big_set.empty()){
		remove_dead_and_captured(big_set);
	}
}

TerminalType Node_switching_game::get_winner(){
	Onturn res = who_won();
	if (res==onturn){ // This is from perspective of current onturn player
		return TERMINAL_WIN;
	}
	else if (res==noplayer){
		return TERMINAL_NONE;
	}
	else{
		return TERMINAL_LOSS;
	}

}

Onturn Node_switching_game::who_won(){
	int src=0;
	bool found1,found2,not_again;
	if (maker_won){
		return maker;
	}
	if (graph.num_vertices==0){
		return breaker;
	}
	bool visited[graph.num_vertices];
	bool ever_visited[graph.num_vertices];
	while (true){
		fill(visited,visited+graph.num_vertices,false);
		visited[src] = true;
		std::stack<int> next;
		found1 = graph.fprops[t1connect][src];
		found2 = graph.fprops[t2connect][src];
		next.push(src);

		while(!next.empty()) {
			int cv = next.top();
			next.pop();
			for (Neighbors neigh = graph.adjacent_vertices(cv);neigh.first!=neigh.second;++neigh.first){
				int nv = *neigh.first;
				if (!visited[nv]){
					if (graph.fprops[t1connect][nv]){
						found1 = true;
					}
					if (graph.fprops[t2connect][nv]){
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

vector<string> Node_switching_game::get_grid_layout(){
	double scale;
	vector<string> position_array(graph.num_vertices);
	scale = 1.;
	const double xstart = 0;
	const double ystart = 0;
	const double xend = xstart+1.5*(board_size-1)*scale;
	const double yend = ystart+sqrt(3./4.)*(board_size-1)*scale;
	for (int i=0;i<graph.num_vertices;i++){
		int bi = graph.lprops[board_location][i];
		int row = floor(bi/board_size);
		int col = bi%board_size;
		position_array[i] = to_string((xstart+(0.5*col+row)*scale))+","+to_string((ystart + (sqrt(3./4.)*col)*scale))+"!";
	}
	return position_array;
}

vector<string> Node_switching_game::get_colors(){
	vector<string> color_array(graph.num_vertices);
	for (int i=0;i<graph.num_vertices;i++){
		if (graph.fprops[t1connect][i]){
			if (graph.fprops[t2connect][i]){
				color_array[i] = "brown";
			}
			else{
				color_array[i] = "green";
			}
		}
		else if (graph.fprops[t2connect][i]){
			color_array[i] = "red";
		}
		else{
			color_array[i] = "black";
		}
	}
	return color_array;
}

void Node_switching_game::graphviz_me (string fname){
	vector<string> color_map = get_colors();
	vector<string> pos_map = get_grid_layout();
	vector<pair<string,vector<string>>> props;
	props.push_back(pair<string,vector<string>>("color",color_map));
	props.push_back(pair<string,vector<string>>("pos",pos_map));

	graph.graphviz_me(props,fname,true);
};

std::vector<torch::Tensor> Node_switching_game::convert_graph(torch::Device &device){
	Neighbors neigh;
	int n = graph.num_vertices;
	torch::TensorOptions options_long = torch::TensorOptions().dtype(torch::kLong).device(device);
	torch::TensorOptions options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
	torch::Tensor node_features = torch::zeros({n,3},options_float);
	node_features.index_put_({Ellipsis,0},torch::tensor(graph.fprops[t1connect],options_float));
	node_features.index_put_({Ellipsis,1},torch::tensor(graph.fprops[t2connect],options_float));

	if (onturn==maker){
		node_features.index_put_({Ellipsis,2},1.);
	}

	torch::Tensor edge_index = torch::empty({2,(int)graph.sources.size()},options_long);
	edge_index.index_put_({0,Ellipsis},torch::tensor(graph.sources,options_long));
	edge_index.index_put_({1,Ellipsis},torch::tensor(graph.targets,options_long));

	std::vector<torch::Tensor> parts;
	parts.push_back(node_features);
	parts.push_back(edge_index);
	return parts;
};

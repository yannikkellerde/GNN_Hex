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
	Neighbors neigh1 = graph.adjacent_vertices(v1);
	Neighbors neigh2 = graph.adjacent_vertices(v2);
	if (neigh1.second-neigh1.first!=neigh2.second-neigh2.first){
		return false;
	}
	for (;neigh1.first!=neigh1.second;++neigh1.first){
		int n1 = *neigh1.first;
		if (n1 != v2){
			vector<int>::const_iterator pi = neigh2.first;
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
	for (;neigh.first!=neigh.second;++neigh.first){
		int v1 = *neigh.first;
		if (v1!=ignore){
			for (vector<int>::const_iterator it2 = neigh.first+1;it2!=neigh.second;++it2){
				int v2 = *it2;
				if (v2!=ignore && !g.edge_exists(v1,v2)){
					return false;
				}
			}
		}
	}
	return true;
}

Node_switching_game::Node_switching_game (int board_size):board_size(board_size),board(board_size){
	reset();
};

Node_switching_game::Node_switching_game (std::vector<torch::Tensor> &data){
	// Convert model repr back to graph
	torch::Tensor node_features = data[0].cpu();
	torch::Tensor edge_index = data[1].cpu();
	graph = Graph(node_features.size(0));
	graph.add_fprop(0.); // IS_TERMINAL
	graph.add_lprop(0); // REMOVED
	graph.add_lprop(0); // BOARD_LOCATION
	graph.fprops[IS_TERMINAL][TERMINAL_1] = 1;
	graph.fprops[IS_TERMINAL][TERMINAL_2] = 1;
	for (int i=0;i<graph.num_vertices;i++){
		graph.lprops[BOARD_LOCATION][i] = i;
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
#ifndef SINGLE_GRAPH
	graph2 = ref.graph2;
#endif
	graph = ref.graph;
#ifndef NO_PLAY
	response_set_red = ref.response_set_red;
	response_set_blue = ref.response_set_blue;
	board_moves_red = ref.board_moves_red;
	board_moves_blue = ref.board_moves_blue;
#endif
	onturn = ref.onturn;
	board_size = ref.board_size;
	board = ref.board;
	move_num = ref.move_num;
	maker_color = ref.maker_color;
	swap_allowed = ref.swap_allowed;
}

Node_switching_game * Node_switching_game::clone(){
	return new Node_switching_game(*this);
}

void Node_switching_game::reset(){
	maker_color = RED;
	move_num = 0;
	board_size = board.size;
	reset_graph();
}

void Node_switching_game::reset_graph(){
#ifndef NO_PLAY
	response_set_red = map<int,int>();
	response_set_blue = map<int,int>();
	board_moves_red = vector<int>();
	board_moves_blue = vector<int>();
#endif
	onturn = RED;
	auto f = [this](Graph& g) {
		g = Graph(board.num_squares+2);
		g.add_fprop(0.); // IS_TERMINAL
		g.add_lprop(0); // REMOVED
		g.add_lprop(0); // BOARD_LOCATION
		g.fprops[IS_TERMINAL][TERMINAL_1] = 1;
		g.fprops[IS_TERMINAL][TERMINAL_2] = 1;
		g.lprops[BOARD_LOCATION][TERMINAL_1] = -1;
		g.lprops[BOARD_LOCATION][TERMINAL_2] = -2;
	};
	f(graph);
	for (int i=0;i<board.num_squares;i++){
		int j = i+2;
		graph.lprops[BOARD_LOCATION][j] = i;
		if (i<board.size){
			graph.add_edge(j,TERMINAL_1);
		}
		if (floor(i/board.size)==board.size-1){
			graph.add_edge(j,TERMINAL_2);
		}
		if (i%board.size>0 && board.size<=i && i<=board.num_squares-board.size){
			graph.add_edge(j,j-1);
		}
		if (i>=board.size){
			graph.add_edge(j,j-board.size);
			if (i%board.size!=board.size-1){
				graph.add_edge(j,j+1-board.size);
			}
		}
	}
#ifndef SINGLE_GRAPH
	f(graph2);
	for (int i=0;i<board.num_squares;i++){
		int j = i+2;
		graph2.lprops[BOARD_LOCATION][j] = i;
		if (i%board.size==0){
			graph2.add_edge(j,TERMINAL_1);
		}
		if (i%board.size==board.size-1){
			graph2.add_edge(j,TERMINAL_2);
		}
		if (i%board.size>0){
			graph2.add_edge(j,j-1);
		}
		if (i>=board.size){
			if (i%board.size!=board_size-1&&i%board.size!=0){
				graph2.add_edge(j,j-board.size);
			}
			if (i%board.size!=board.size-1){
				graph2.add_edge(j,j+1-board.size);
			}
		}
	}
#endif
}

uint32_t Node_switching_game::hash_key() const { // https://stackoverflow.com/a/27216842
	std::size_t seed = graph.sources.size();
#ifdef SINGLE_GRAPH
	for (const Graph& g:{graph}){
#else
	for (const Graph& g:{graph,graph2}){
#endif
		for(const int& i : graph.sources) {
			seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		for(const int& i : graph.targets) {
			seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
	}
	return seed;
}

int Node_switching_game::vertex_from_board_location(int bl) const{
	int i=0;
	for (vector<int>::const_iterator vs = graph.lprops[BOARD_LOCATION].begin();vs!=graph.lprops[BOARD_LOCATION].end();++vs,++i){
		int v = *vs;
		if (v == bl){
			return i;
		}
	}
	return -1;
}

int Node_switching_game::action_from_board_location(int bl) const{
	int out = vertex_from_board_location(bl);
	return out==-1?out:out-2;
}

#ifndef NO_PLAY
int Node_switching_game::get_response(int bloc,bool for_red){
	if (response_set_red.find(bloc)!=response_set_red.end()){
		if (for_red){
			return response_set_red[bloc];
		}
		else{
			int otherside = response_set_red[bloc];
			response_set_red.erase(otherside);
			response_set_red.erase(bloc);
			return -1;
		}
	}
	else if (response_set_blue.find(bloc)!=response_set_blue.end()){
		if (!for_red){
			return response_set_blue[bloc];
		}
		else{
			int otherside = response_set_blue[bloc];
			response_set_blue.erase(otherside);
			response_set_blue.erase(bloc);
			return -1;
		}
	}
	return -1;
}
#endif

Onturn Node_switching_game::not_onturn(){
	return onturn==RED?BLUE:RED;
}

void Node_switching_game::switch_onturn(){
	onturn = onturn==RED?BLUE:RED;
}

set<int> Node_switching_game::fix_terminal_connections(int terminal){
	set<int> change_set;
	auto f = [this,terminal](Graph& g,set<int>& change_set) {
		vector<pair<int,int>> to_del;
		for (Neighbors neigh = g.adjacent_vertices(terminal);neigh.first!=neigh.second;++neigh.first){
			int v1 = *neigh.first;
			for (vector<int>::const_iterator neigh2 = neigh.first+1;neigh2!=neigh.second;++neigh2){
				int v2 = *neigh2;
				if (v2!=v1 && g.edge_exists(v1,v2)){
					to_del.push_back(pair<int,int>(v1,v2));
					change_set.insert(v1);
					change_set.insert(v2);
				}
			}
		}
		for (pair<int,int>del_me:to_del){
			g.delete_edge(del_me.first,del_me.second);
		}
	};
	f(graph,change_set);
#ifndef SINGLE_GRAPH
	f(graph2,change_set);
#endif

	return change_set;
}

void Node_switching_game::remove_marked_nodes(){
	auto f = [this](Graph& g) {
		vector<int>::iterator rem = g.lprops[REMOVED].end()-1;
		for (int i=g.num_vertices-1;i>=0;--i,--rem){
			if (*rem){
				g.remove_vertex(i);
			}
		}
	};
	f(graph);
#ifndef SINGLE_GRAPH
	f(graph2);
#endif
}

int Node_switching_game::get_random_action() const{
	vector<int> actions = get_actions();
	return repeatable_random_choice(actions);
}

int Node_switching_game::get_num_actions() const{
	return (swap_allowed&&move_num==1)?graph.num_vertices+1:graph.num_vertices;
}

vector<int> Node_switching_game::get_actions() const{
	vector<int> res;
	if (swap_allowed&&move_num==1){ // allow swap
		res.resize(graph.num_vertices-1);
	}
	else{
		res.resize(graph.num_vertices-2);
	}
	iota(res.begin(),res.end(),0);
	return res;
}


set<int> Node_switching_game::make_move(int action, bool do_force_color, Onturn force_color,bool do_remove_dead_and_captured,bool only_mark_removed){
	set<int> change_set;
	int vertex = action+2;
	Onturn player = do_force_color?force_color:onturn;
	if (!do_force_color){
#ifdef SINGLE_GRAPH
		if (move_num==0){
			first_move = action;
		}
#endif
		++move_num;
		switch_onturn();
		if (swap_allowed&&vertex == graph.num_vertices&&move_num==2){ // swap rule
			/* print_info(__LINE__,__FILE__,"swap is happening"); */
#ifdef SINGLE_GRAPH
			reset_graph();
			make_move(first_move,true,not_onturn(),true,false);
#else
			maker_color = maker_color==RED?BLUE:RED;
#ifndef NO_PLAY
			swap(response_set_blue,response_set_red);
			swap(board_moves_blue,board_moves_red);
#endif
			swap(graph,graph2);
#endif
			return change_set;
		}
	}
	assert(vertex<graph.num_vertices);
	auto f = [this,player,vertex,do_remove_dead_and_captured,only_mark_removed](Graph& g, Onturn maker_color) {
		set<int> change_set;
#ifndef NO_PLAY
		if (player == RED){
			board_moves_red.push_back(g.lprops[BOARD_LOCATION][vertex]);
		}
		else{
			board_moves_blue.push_back(g.lprops[BOARD_LOCATION][vertex]);
		}
#endif
		if (player==maker_color){
			vector<pair<int,int>> to_add;
			int have_to_fix=-1;
			for (Neighbors neigh = g.adjacent_vertices(vertex);neigh.first!=neigh.second;++neigh.first){
				int v1 = *neigh.first;
				if (v1<2){
					have_to_fix = v1;
				}
				for (vector<int>::const_iterator it2 = neigh.first+1;it2!=neigh.second;++it2){
					int v2 = *it2;
					if (v2<2){
						have_to_fix = v2;
					}
					if ((!((g.edge_exists(v1,TERMINAL_1))&&(g.edge_exists(v2,TERMINAL_1))))&&(!((g.edge_exists(v1,TERMINAL_2))&&(g.edge_exists(v2,TERMINAL_2)))))
						to_add.push_back(pair<int,int>(v1,v2));
					}
				}
			for (pair<int,int> add_me : to_add){
				g.add_edge(add_me.first,add_me.second);
			}
			if (have_to_fix!=-1){
				change_set = fix_terminal_connections(have_to_fix);
			}
		}
		if (do_remove_dead_and_captured){
			Neighbors neigh = g.adjacent_vertices(vertex);
			change_set.insert(neigh.first,neigh.second);
		}
		g.clear_vertex(vertex);
		if (do_remove_dead_and_captured || only_mark_removed){
			g.lprops[REMOVED][vertex] = 1;
		}
		else{
			g.remove_vertex(vertex);
		}
		return change_set;
	};
	change_set = f(graph,RED);
#ifndef SINGLE_GRAPH
	f(graph2,BLUE);
#endif
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
		if (graph.lprops[REMOVED][vertex]||vertex<2){
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
				if (v1==vertex||v1<2){
					continue;
				}
				if (check_if_same(graph,v1,vertex)){
#ifndef NO_PLAY
					response_set_red[graph.lprops[BOARD_LOCATION][v1]]=graph.lprops[BOARD_LOCATION][vertex];
					response_set_red[graph.lprops[BOARD_LOCATION][vertex]]=graph.lprops[BOARD_LOCATION][v1];
#endif
					/* tmp_neigh = graph.adjacent_vertices(vertex);       // These are not */
					/* big_set.insert(tmp_neigh.first,tmp_neigh.second);  // in the python original */
					/* cout << "maker captured " << v1 << " " << vertex << endl; */
					neigh_neigh_neigh = graph.adjacent_vertices(v1);
					big_set.insert(neigh_neigh_neigh.first,neigh_neigh_neigh.second);
					change_set = make_move(v1-2,true,BLUE,false,true);
					change_set = make_move(vertex-2,true,RED,false,true);
					big_set.insert(change_set.begin(),change_set.end());
					can_continue = true;
					break;
				}
			}
		}
		if (can_continue){
			continue;
		}

		for (neighbors = graph.adjacent_vertices(vertex);neighbors.first!=neighbors.second;++neighbors.first){
			v1 = *neighbors.first;
			for (vector<int>::const_iterator it2 = neighbors.first+1;it2!=neighbors.second;++it2){
				if (is_dead&&!graph.edge_exists(*neighbors.first,*it2)){
					is_dead = false;
				}
			}
			if (v1<2) continue;
			if (is_fully_connected(graph,vertex,v1) && is_fully_connected(graph,v1,vertex)){
#ifndef NO_PLAY
				response_set_blue[graph.lprops[BOARD_LOCATION][v1]]=graph.lprops[BOARD_LOCATION][vertex];
				response_set_blue[graph.lprops[BOARD_LOCATION][vertex]]=graph.lprops[BOARD_LOCATION][v1];
				board_moves_blue.push_back(graph.lprops[BOARD_LOCATION][v1]);
				board_moves_blue.push_back(graph.lprops[BOARD_LOCATION][vertex]);
#endif
				/* cout << "breaker captured " << v1 << " " << vertex << endl; */
				tmp_neigh = graph.adjacent_vertices(vertex);
				neigh_neigh = graph.adjacent_vertices(v1);
				big_set.insert(tmp_neigh.first,tmp_neigh.second);
				big_set.insert(neigh_neigh.first,neigh_neigh.second);
				change_set = make_move(v1-2,true,BLUE,false,true);
				change_set = make_move(vertex-2,true,BLUE,false,true);
				can_continue = true;
				break;
			}
		}
		if (is_dead && !can_continue){
#ifndef NO_PLAY
			board_moves_blue.push_back(graph.lprops[BOARD_LOCATION][vertex]);
#endif
			/* cout << "node is dead " << vertex << endl; */
			neighbors = graph.adjacent_vertices(vertex);
			big_set.insert(neighbors.first,neighbors.second);
			change_set = make_move(vertex-2,true,BLUE,false,true);
			continue;
		}
	}
	if (!big_set.empty()){
		remove_dead_and_captured(big_set);
	}
}

TerminalType Node_switching_game::get_winner() const{
	Onturn res = who_won();
	if (res==onturn){ // This is from perspective of current onturn player
		return TERMINAL_WIN;
	}
	else if (res==NOPLAYER){
		return TERMINAL_NONE;
	}
	else{
		return TERMINAL_LOSS;
	}

}

Onturn Node_switching_game::who_won() const{
	bool found1,found2,not_again;
	if (graph.edge_exists(TERMINAL_1,TERMINAL_2)){
		return RED;
	}
#ifdef SINGLE_GRAPH
	bool visited[graph.num_vertices];
	int src=TERMINAL_1;
	fill(visited,visited+graph.num_vertices,false);
	visited[src] = true;
	std::stack<int> next;
	next.push(src);

	while(!next.empty()) {
		int cv = next.top();
		next.pop();
		for (Neighbors neigh = graph.adjacent_vertices(cv);neigh.first!=neigh.second;++neigh.first){
			int nv = *neigh.first;
			if (!visited[nv]){
				if (nv == TERMINAL_2){
					return NOPLAYER;
				}
					visited[nv] = true;
					next.push(nv);
				}
			}
		}
	return BLUE;
#else
	if (graph2.edge_exists(TERMINAL_1,TERMINAL_2)){
		return BLUE;
	}
	return NOPLAYER;
#endif
}

vector<string> Node_switching_game::get_grid_layout(Onturn color) const{
	double scale;
	vector<string> position_array(graph.num_vertices);
	scale = 1+0.1*(11-board_size);
	const double xstart = 0;
	const double ystart = 0;
	const double xend = xstart+1.5*(board_size-1)*scale;
	const double yend = ystart+sqrt(3./4.)*(board_size-1)*scale;
	if (color==maker_color){
		position_array[0] = to_string(xstart)+","+to_string(yend/2)+"!";
		position_array[1] = to_string(xend)+","+to_string(yend/2)+"!";
	}
	else{
		position_array[0] = to_string(xend/3)+","+to_string(ystart-scale)+"!";
		position_array[1] = to_string((xend/3)*2)+","+to_string(yend+scale)+"!";
	}
	for (int i=2;i<graph.num_vertices;i++){
		int bi = graph.lprops[BOARD_LOCATION][i];
		int row = floor(bi/board_size);
		int col = bi%board_size;
		position_array[i] = to_string((xstart+(0.5*col+row)*scale))+","+to_string((ystart + (sqrt(3./4.)*col)*scale))+"!";
	}
	return position_array;
}

vector<string> Node_switching_game::get_colors() const{
	vector<string> color_array(graph.num_vertices);
	for (int i=0;i<graph.num_vertices;i++){
		if (graph.fprops[IS_TERMINAL][i]){
			color_array[i] = "red";
		}
		else{
			color_array[i] = "black";
		}
	}
	return color_array;
}

string Node_switching_game::format_action(int action) const{
	int vertex = action+2;
	assert (vertex<graph.num_vertices);
	return to_string(action)+"("+to_string(graph.lprops[BOARD_LOCATION][vertex])+")";
}

void Node_switching_game::graphviz_me (string fname) const{
	graphviz_me(fname,graph);
}

void Node_switching_game::graphviz_me (string fname, const Graph& g) const{
	vector<int> nodenum(g.num_vertices);
	vector<string> nodetext(g.num_vertices);
	std::iota(std::begin(nodenum), std::end(nodenum), 0);
	for (int i=0;i<g.num_vertices;++i){
		nodetext[i] = to_string(nodenum[i]);
	}
	graphviz_me(nodetext,fname,g);
}

void Node_switching_game::graphviz_me (vector<string> nodetext, string fname, const Graph& g) const{
	vector<string> color_map = get_colors();
#ifdef SINGLE_GRAPH
	vector<string> pos_map = get_grid_layout(RED);
#else
	vector<string> pos_map = get_grid_layout(&g==&graph2?BLUE:RED);
#endif
	vector<pair<string,vector<string>>> props;
	props.push_back(pair<string,vector<string>>("color",color_map));
	props.push_back(pair<string,vector<string>>("pos",pos_map));
	props.push_back(pair<string,vector<string>>("label",nodetext));

	g.graphviz_me(props,fname,true);
};

std::vector<torch::Tensor> Node_switching_game::convert_graph(torch::Device &device, const Graph& g) const{
	Neighbors neigh;
	int n = g.num_vertices;
	torch::TensorOptions options_long = torch::TensorOptions().dtype(torch::kLong).device(device);
	torch::TensorOptions options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
	torch::Tensor node_features = torch::zeros({n,3},options_float);
	node_features.index_put_({Ellipsis,0},torch::tensor(g.fprops[IS_TERMINAL],options_float));

/* #ifdef SINGLE_GRAPH */
/* 	if (onturn==RED){ */
/* #else */
/* 	if ((onturn==RED && &g==&graph)||(onturn==BLUE && &g==&graph2)){ */
/* #endif */
/* 		node_features.index_put_({Ellipsis,2},1.); */
/* 	} */
	if (swap_allowed){
		switch (move_num){
			case 0:
				node_features.index_put_({Ellipsis,1},1.);
				node_features.index_put_({Ellipsis,2},0.);
				break;
			case 1:
				node_features.index_put_({Ellipsis,1},0.);
				node_features.index_put_({Ellipsis,2},1.);
				break;
			case 2:
				node_features.index_put_({Ellipsis,1},0.);
				node_features.index_put_({Ellipsis,2},0.);
				break;
		}
	}

	torch::Tensor edge_index = torch::empty({2,(int)g.sources.size()},options_long);
	edge_index.index_put_({0,Ellipsis},torch::tensor(g.sources,options_long));
	edge_index.index_put_({1,Ellipsis},torch::tensor(g.targets,options_long));

	std::vector<torch::Tensor> parts;
	parts.push_back(node_features);
	parts.push_back(edge_index);
	return parts;
};

std::vector<torch::Tensor> Node_switching_game::convert_graph(torch::Device &device) const{
#ifdef SINGLE_GRAPH
	return convert_graph(device,graph);
#else
	return onturn==BLUE?convert_graph(device,graph2):convert_graph(device,graph);
#endif
};

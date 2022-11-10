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
#include "hex_board_game.h"
#include "graph.h"
#include "util.h"
#include <math.h>

using namespace std;
using namespace torch::indexing;

#if !defined(SHANNON_H)
#define SHANNON_H

enum TerminalType {
    TERMINAL_LOSS,
    TERMINAL_DRAW,
    TERMINAL_WIN,
    TERMINAL_CUSTOM,
    TERMINAL_NONE
};


enum Onturn {maker,breaker,noplayer};
enum Fprops {t1connect,t2connect};
enum Lprops {removed,board_location};


bool check_if_same(Graph& graph, int v1, int v2);

bool is_fully_connected(Graph& g,int vert, int ignore);

bool is_fully_connected(Graph& g,Neighbors& neigh, int ignore);

bool is_fully_connected(Graph& g,Neighbors& neigh);

class Node_switching_game {
	public:
		Graph graph;
		Onturn onturn=maker;
		int board_size;
		Hex_board board;
		bool maker_won=false;
		int move_num;
		map<int,int> response_set_maker;
		map<int,int> response_set_breaker;

		Node_switching_game (int board_size=11);

		Node_switching_game (Graph& g);

		Node_switching_game (std::vector<torch::Tensor> &data);

		Node_switching_game (Hex_board& from_board);

		Node_switching_game(Node_switching_game& ref);

		Node_switching_game * clone();

		void reset();

		uint16_t hash_key();

		int vertex_from_board_location(int bl);

		int get_response(int bloc,bool for_maker);

		set<int> fix_terminal_connections(int vertex, Fprops conn_prop);

		void remove_marked_nodes();

		void switch_onturn();

		int get_random_action();

		vector<int> get_actions();

		set<int> make_move(int vertex, bool do_force_color=false, Onturn force_color=noplayer,bool do_remove_dead_and_captured=false,bool only_mark_removed=false);

		void remove_dead_and_captured(set<int> &consider_set);

		TerminalType get_winner();

		Onturn who_won();

		vector<string> get_grid_layout();

		vector<string> get_colors();

		void graphviz_me (string fname);

	std::vector<torch::Tensor> convert_graph(torch::Device &device);
};
#endif

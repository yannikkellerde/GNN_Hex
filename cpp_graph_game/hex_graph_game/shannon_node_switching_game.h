#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <ATen/ATen.h>
#include "hex_board_game.h"
#include "graph.h"

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
#ifndef NO_PLAY
		map<int,int> response_set_maker;
		map<int,int> response_set_breaker;
		vector<int> board_moves_maker;
		vector<int> board_moves_breaker;
#endif

		Node_switching_game (int board_size=11);

		Node_switching_game (Graph& g);

		Node_switching_game (std::vector<torch::Tensor> &data);

		Node_switching_game (Hex_board& from_board);

		Node_switching_game(Node_switching_game& ref);

		Node_switching_game * clone();

		void reset();

		uint32_t hash_key() const;

		int vertex_from_board_location(int bl) const;

		int get_response(int bloc,bool for_maker); // Not const, deletes response

		set<int> fix_terminal_connections(int vertex, Fprops conn_prop);

		void remove_marked_nodes();

		void switch_onturn();

		Onturn not_onturn();

		int get_random_action() const;

		vector<int> get_actions() const;

		set<int> make_move(int vertex, bool do_force_color=false, Onturn force_color=noplayer,bool do_remove_dead_and_captured=false,bool only_mark_removed=false);

		void remove_dead_and_captured(set<int> &consider_set);

		TerminalType get_winner() const;

		Onturn who_won() const;

		string format_action(int action) const;

		vector<string> get_grid_layout() const;

		vector<string> get_colors() const;

		void graphviz_me (string fname) const;

		void graphviz_me (vector<string> nodetext,string fname) const;

	std::vector<torch::Tensor> convert_graph(torch::Device &device) const;
};
#endif

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


enum TerminalID {TERMINAL_1, TERMINAL_2};
enum Onturn {RED,BLUE,NOPLAYER};
enum Fprops {IS_TERMINAL};
enum Lprops {REMOVED,BOARD_LOCATION};

bool check_if_same(Graph& graph, int v1, int v2);

bool is_fully_connected(Graph& g,int vert, int ignore);

bool is_fully_connected(Graph& g,Neighbors& neigh, int ignore);

bool is_fully_connected(Graph& g,Neighbors& neigh);

class Node_switching_game {
	public:
		Graph graph;
#ifndef SINGLE_GRAPH
		Graph graph2;
#else
		int first_move;
#endif
		Onturn onturn=RED;
		Onturn maker_color=RED;
		int board_size;
		Hex_board board;
		int move_num;
		bool swap_allowed=false;
#ifndef NO_PLAY
		map<int,int> response_set_red;
		map<int,int> response_set_blue;
		vector<int> board_moves_red;
		vector<int> board_moves_blue;
#endif

		Node_switching_game (int board_size=11, bool swap_allowed=false);

		Node_switching_game (std::vector<torch::Tensor> &data);

		Node_switching_game (Hex_board& from_board);

		Node_switching_game(Node_switching_game& ref);

		Node_switching_game * clone();

		void reset();

		void reset_graph();

		uint32_t hash_key() const;

		int vertex_from_board_location(int bl) const;

		void load_sgf(string &sgf);

		int action_from_board_location(int bl) const;

		int get_response(int bloc,bool for_red); // Not const, deletes response

		std::string number_to_notation(int number);
		int notation_to_number(std::string &notation);

		set<int> fix_terminal_connections(int terminal);

		void remove_marked_nodes();

		void switch_onturn();

		Onturn not_onturn();

		int get_random_action() const;

		vector<int> get_actions() const;

		int get_num_actions() const;

		set<int> make_move(int action, bool do_force_color=false, Onturn force_color=NOPLAYER, bool do_remove_dead_and_captured=false, bool only_mark_removed=false);

		void remove_dead_and_captured(set<int> &consider_set);

		TerminalType get_winner() const;

		Onturn who_won() const;

		string format_action(int action) const;

		vector<string> get_grid_layout(Onturn color) const;

		vector<string> get_colors() const;

		void graphviz_me (string fname) const;

		void graphviz_me (string fname, const Graph& g) const;

		void graphviz_me (vector<string> nodetext,string fname,const Graph& g) const;

		std::vector<torch::Tensor> convert_graph(torch::Device &device) const;

		std::vector<torch::Tensor> convert_graph(torch::Device &device, const Graph& graph) const;
		std::vector<torch::Tensor> convert_planes(torch::Device &device) const;
		std::vector<torch::Tensor> convert_planes_gao(torch::Device &device) const;
};
#endif

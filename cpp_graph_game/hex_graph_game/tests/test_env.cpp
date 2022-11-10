#include "../shannon_node_switching_game.h"
#include<iostream>
#include <unistd.h>
#include "test_env.h"

void interactive_env(){
	torch::Device device(torch::kCPU,0);
	int move;
	Hex_board board(3);
	Node_switching_game game_old(board);
	std::vector<torch::Tensor> data = game_old.convert_graph(device);
	Node_switching_game game(data);
	game.board_size = 3;
	/* Node_switching_game<5> game; */
	while (true){
		/* game.graphviz_me(cout); */
		game.graphviz_me("my_graph.dot");
		game.graph.do_complete_dump("graph_dump.txt");
    system("pkill -f 'mupdf my_graph.pdf'");
		system("neato -Tpdf my_graph.dot -o my_graph.pdf");
		system("mupdf my_graph.pdf &");
		usleep(100000U);
		system("bspc node -f west");
		cin >> move;
		game.make_move(move,false,maker,true);
		Onturn winner = game.who_won();
		if (winner==maker){
			cout << "maker won" << endl;
		}
		if (winner==breaker){
			cout << "breaker won" << endl;
		}
	}
}

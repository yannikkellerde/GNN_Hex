#include "../shannon_node_switching_game.h"
#include<iostream>
#include <unistd.h>
#include "test_env.h"

void interactive_env(){
	torch::Device device(torch::kCPU,0);
	int move;
	Hex_board board(5);
	Node_switching_game game(board);
	Graph* cur_graph = &game.graph;
	/* std::vector<torch::Tensor> data = game_old.convert_graph(device); */
	/* Node_switching_game game(data); */
	/* game.board_size = 5; */
	/* Node_switching_game<5> game; */
	while (true){
		/* game.graphviz_me(cout); */
		cout << game.graph.num_vertices <<endl;
		game.graphviz_me("my_graph.dot",*cur_graph);
		game.graph.do_complete_dump("graph_dump.txt");
    system("pkill -f 'mupdf my_graph.pdf'");
		system("neato -Tpdf my_graph.dot -o my_graph.pdf");
		system("mupdf my_graph.pdf &");
		usleep(100000U);
		system("bspc node -f west");
		cin >> move;
#ifndef SINGLE_GRAPH
		if (move==-1){
			cur_graph = cur_graph==&game.graph?&game.graph2:&game.graph;
		}
		else{
			game.make_move(move,false,NOPLAYER,true);
		}
#else
		game.make_move(move,false,NOPLAYER,true);
#endif
		Onturn winner = game.who_won();
		if (winner==RED){
			cout << "red won" << endl;
		}
		if (winner==BLUE){
			cout << "blue won" << endl;
		}
	}
}

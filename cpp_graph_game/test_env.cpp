#include "shannon_node_switching_game.cpp"
#include<iostream>
#include <unistd.h>

int main(){
	int move;
	Hex_board<5> board;
	Node_switching_game game(board);
	while (true){
		/* game.graphviz_me(cout); */
		ofstream my_file;
		my_file.open("my_graph.dot");
		game.graphviz_me(my_file);
		my_file.close();
    system("pkill -f 'mupdf my_graph.pdf'");
		system("neato -Tpdf my_graph.dot -o my_graph.pdf");
		system("mupdf my_graph.pdf &");
		usleep(100000U);
		system("bspc node -f west");
		cin >> move;
		game.make_move(move);
	}
	
	return 0;
}

#include "shannon_node_switching_game.cpp"

int main(){
	Hex_board<5> board;
	Node_switching_game game(board);
	for (int i=0;i<10;i++){
		game.make_move(3);
	}
	game.graphviz_me(cout);
	ofstream my_file;
	my_file.open("my_graph.txt");
	game.graphviz_me(my_file);
	my_file.close();

	ofstream next_file;
	my_file.open("prop_graph.txt");
	write_graphviz(my_file, game.graph, make_label_writer(get(&PropertyStruct::board_location, game.graph)));
	next_file.close();
	
	return 0;
}

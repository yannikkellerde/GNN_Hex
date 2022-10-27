#include "shannon_node_switching_game.cpp"
#include<iostream>
#include <unistd.h>

void interactive_env(){
	torch::Device device(torch::kCUDA,0);
	int move;
	Hex_board<5> board;
	Node_switching_game<5> game_old(board);
	std::vector<torch::jit::IValue> data = game_old.convert_graph(device);
	Node_switching_game<5> game(data);
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
		cout << game.graph[12].board_location;
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

void test_dead_and_captured_consistency(){
	Node_switching_game<5> simple_game;
	Node_switching_game<5> fancy_game;
	for (int i=0;i<500;i++){
		simple_game = Node_switching_game<5>();
		fancy_game = Node_switching_game<5>();
		while (simple_game.who_won()==noplayer){

		}
	}
}

int main(){
	
	return 0;
}

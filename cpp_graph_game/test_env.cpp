#include "shannon_node_switching_game.cpp"
#include<iostream>
#include <unistd.h>

void interactive_env(){
	torch::Device device(torch::kCUDA,0);
	int move;
	/* Hex_board<11> board; */
	/* Node_switching_game<11> game_old(board); */
	/* std::vector<torch::jit::IValue> data = game_old.convert_graph(device); */
	/* Node_switching_game<11> game(data); */
	Node_switching_game<11> game;
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

#ifdef FOR_INFERENCE
void test_dead_and_captured_consistency(){
	Node_switching_game<11> simple_game;
	Node_switching_game<11> fancy_game;
	int move, board_response, vertex_response, bmove, fancy_vmove;
	vector<int> moves;
	for (int i=0;i<500;i++){
		while (simple_game.who_won()==noplayer){
			assert(simple_game.onturn == fancy_game.onturn);
			move = simple_game.get_random_action();
			bmove = simple_game.graph[move].board_location;
			/* cout << "board index " << simple_game.graph[move].board_location << " selected. Move was " << move << endl; */
			simple_game.make_move(move,false,noplayer,false);
			fancy_vmove = fancy_game.vertex_from_board_location(bmove);
			if (fancy_vmove!=-1){
				if (fancy_game.who_won()==noplayer){
					/* cout << "fancy player makes board move " << bmove << " vmove:" << fancy_vmove << endl; */
					fancy_game.make_move(fancy_vmove,false,noplayer,true);
				}
				else{
					fancy_game.switch_onturn();
				}
			}
			else{
				board_response = fancy_game.get_response(bmove,fancy_game.onturn == breaker); // if breaker moved, maker responds
				if (board_response!=-1){
					vertex_response = simple_game.vertex_from_board_location(board_response);
					/* cout << "got response " << board_response << " " << vertex_response << " " << bmove << endl; */
					simple_game.make_move(vertex_response,false,noplayer,false);
				}
				else{
					fancy_game.switch_onturn();
				}
			}
		}
		cout << simple_game.who_won() << "  " << fancy_game.who_won() << endl;
		assert(simple_game.who_won()==fancy_game.who_won());
		simple_game.reset();
		fancy_game.reset();
		/* return; */
	}
}
#endif

int main(){
	srand(0);
#ifdef FOR_INFERENCE
	/* test_dead_and_captured_consistency(); */
	interactive_env();
#endif
	return 0;
}

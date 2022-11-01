#include "shannon_node_switching_game.cpp"

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
				board_response = fancy_game.get_response(bmove,simple_game.onturn == maker); // if breaker moved, maker responds
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
		Onturn sww = simple_game.who_won();
		Onturn fww = fancy_game.who_won();
		cout << sww << "  " << fww << endl;
		if (sww!=fww){
			simple_game.graphviz_me("simple.dot");
			fancy_game.graphviz_me("fancy.dot");
		}
		assert(sww==fww);
		simple_game.reset();
		fancy_game.reset();
		/* return; */
	}
}

/* #include "../shannon_node_switching_game.h" */
#include "../shannon_node_switching_game.h"
#include<iostream>
#include <unistd.h>
#include "consistency_test.h"
#include "../util.h"

void test_dead_and_captured_consistency(){
	Node_switching_game simple_game(11);
	Node_switching_game fancy_game(11);
	string f;
	int move, board_response, vertex_response, bmove, fancy_vmove, movenum, redwins, bluewins;
	redwins = 0; bluewins = 0;
	vector<int> moves;
	for (int i=0;i<500;i++){
		movenum = 0;
		while (simple_game.who_won()==NOPLAYER){
			fancy_game.graphviz_me("graphs/graph_"+to_string(movenum)+".dot",fancy_game.graph);
			/* fancy_game.graphviz_me("graphs/graph2_"+to_string(movenum)+".dot",fancy_game.graph2); */
			/* system(("neato -Tpdf -O graphs/graph_"+to_string(movenum)+".dot").c_str()); */
			/* system(("neato -Tpdf -O graphs/graph2_"+to_string(movenum)+".dot").c_str()); */
			assert(simple_game.onturn == fancy_game.onturn);
			move = simple_game.get_random_action();
			if (move+2==simple_game.graph.num_vertices){
				bmove = -1;
				fancy_vmove = fancy_game.graph.num_vertices-2;
			}
			else{
				bmove = simple_game.graph.lprops[BOARD_LOCATION][move+2];
				fancy_vmove = fancy_game.action_from_board_location(bmove);
			}
			simple_game.make_move(move,false,NOPLAYER,false);
			/* cout << "simple move " << move << endl; */
			/* cout << "fancy onturn " << fancy_game.onturn << endl; */
			/* cout << "fancy move " << fancy_vmove << endl; */
			/* cout << "fancy winner " << fancy_game.who_won() << endl; */

			if (fancy_vmove!=-1){
				if (fancy_game.who_won()==NOPLAYER){
					fancy_game.make_move(fancy_vmove,false,NOPLAYER,true);
				}
				else{
					fancy_game.switch_onturn();
				}
			}
			else{
				board_response = fancy_game.get_response(bmove,simple_game.onturn == RED); // if breaker moved, maker responds
				if (board_response!=-1){
					vertex_response = simple_game.action_from_board_location(board_response);
					/* cout << "responded to " << move << " with " << vertex_response << endl; */
					/* cout << "board responded to " << bmove << " with " << board_response << endl; */
					simple_game.make_move(vertex_response,false,NOPLAYER,false);
				}
				else{
					fancy_game.switch_onturn();
				}
			}
			movenum++;
		}
		Onturn sww = simple_game.who_won();
		Onturn fww = fancy_game.who_won();
		if (sww==RED) ++redwins;
		else ++bluewins;
		cout << sww << "  " << fww << endl;
		if (sww!=fww){
			cout << i << endl;
			simple_game.graphviz_me("simple.dot");
			fancy_game.graphviz_me("fancy.dot");
		}
		assert(sww==fww);
		simple_game.reset();
		fancy_game.reset();
		if (i%2==1){
			simple_game.switch_onturn();
			fancy_game.switch_onturn();
		}
		/* return; */
	}
	cout << "redwins " << redwins << endl; 
	cout << "bluewins " << bluewins << endl; 
}

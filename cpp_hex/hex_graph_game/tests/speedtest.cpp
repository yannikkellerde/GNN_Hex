#include "../shannon_node_switching_game.h"
#include <iostream>
#include <unistd.h>
#include <chrono>
#include "speedtest.h"

void speedtest(){
	torch::Device device(torch::kCPU,0);
	int move,move_num,move_time;
	Onturn winner;
	const int size=11;
	const int num_games = 500;
	move_time = 0;
	Hex_board board(size);
	Node_switching_game game(board);
	Node_switching_game game_list[num_games];
	for (int i=0;i<num_games;i++){
		/* game_list[i] = game.copy(); */
		/* game_list[i] = Node_switching_game<size>(game); // this copys, i think */
		/* game_list[i] = Node_switching_game<size>(); */
		if (i%2==0){
			game_list[i].switch_onturn();
		}
	}
	vector<Onturn> winstats;

	vector<vector<torch::jit::IValue>> converted;

	auto start_out = chrono::high_resolution_clock::now();
	for (int i=0;i<num_games;++i){
		move_num = 0;
		do{
			auto start = chrono::high_resolution_clock::now();
			move = game_list[i].get_random_action();

			game_list[i].make_move(move,false,NOPLAYER,true);
			auto stop = chrono::high_resolution_clock::now();
			auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
			move_time+=duration.count();
			winner = game_list[i].who_won();
			if (winner == NOPLAYER){
				game_list[i].convert_graph(device);
			}
			else{
				winstats.push_back(winner);
				break;
			}
		}
		while (true);
	}
	auto stop_out = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(stop_out - start_out);
	cout << endl;
	cout << "total time " << duration.count() << endl;
	/* cout << "initialization" << init_time << endl; */
	/* cout << "node features" << feat_time << endl; */
	/* cout << "edge indices" << ei_time << endl; */
	/* cout << different_time << endl; */
	cout << "red wins" << count(winstats.begin(),winstats.end(),RED) << endl;
	cout << "blue wins " << count(winstats.begin(),winstats.end(),BLUE) << endl;
	cout << "move time " << move_time << endl;
	
	/* return 0; */
}

#include "shannon_node_switching_game.cpp"
#include <iostream>
#include <unistd.h>
#include <chrono>

void run_experiment(){
	torch::Device device(torch::kCPU,0);
	int move,move_num,move_time;
	Onturn winner;
	const int size=11;
	const int num_games = 10;
	move_time = 0;
	Hex_board<size> board;
	Node_switching_game<size> game(board);
	Node_switching_game<size> game_list[num_games];
	for (int i=0;i<num_games;i++){
		game_list[i] = game.copy();
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
			assert (num_vertices(game_list[i].graph)!=0);
			move = (rand()/((RAND_MAX + 1u)/(num_vertices(game_list[i].graph))));
			game_list[i].make_move(move,false,noplayer,true);
			auto stop = chrono::high_resolution_clock::now();
			auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
			move_time+=duration.count();
			winner = game_list[i].who_won();
			if (winner == noplayer){
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
	cout << duration.count() << endl;
	cout << init_time << endl;
	cout << feat_time << endl;
	cout << ei_time << endl;
	/* cout << different_time << endl; */
	/* cout << count(winstats.begin(),winstats.end(),maker) << endl; */
	/* cout << count(winstats.begin(),winstats.end(),breaker) << endl; */
	/* cout << move_time << endl; */
	
	/* return 0; */
}

int main(){
	run_experiment();
}

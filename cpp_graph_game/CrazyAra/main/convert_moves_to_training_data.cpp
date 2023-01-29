#include "convert_moves_to_training_data.h"
#include "shannon_node_switching_game.h"
#include "rl/traindataexporter.h"
#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/api/include/torch/serialize.h>

void to_training_data(string &filename,int hex_size,string &output_folder,int max_games_per_file, bool with_swap){
	int move, best_move, ply, cur_idx, game_counter, file_idx, vertex_move;
	bool skip_this_game = false;
	double swap_prob,value;
	string line;
	vector<int> policy_vec;
	torch::Tensor policy;
  ifstream f(filename);
	file_idx = 0;
	unique_ptr<TrainDataExporter> exporter = std::make_unique<TrainDataExporter>(output_folder+"/mohex_data_"+to_string(file_idx));
	unique_ptr<Node_switching_game> game = std::make_unique<Node_switching_game>(hex_size,with_swap);
	game_counter = 0;
	cur_idx = 0;
	if (f.is_open())
		{
			while ( getline (f,line) )
			{
				if (line == "New game"){
					skip_this_game = false;
					exporter->gameStartPtr.push_back(cur_idx);
					for (int i=ply;i>0;--i){
						exporter->gamePlysToEnd.push_back(i);
					}
					game->reset();
					ply = 0;
					++game_counter;
					if (game_counter%max_games_per_file==0){
						cout << "Exporting ..." << endl;
						exporter->export_game_samples();
						cout << "Done" << endl;
						++file_idx;
						exporter.reset(new TrainDataExporter(output_folder+"/mohex_data_"+to_string(file_idx)));
					}
				}
				else{
					if (skip_this_game) continue;
					/* cout << line << endl; */
					std::stringstream sstr(line);
					std::string segment;
					std::vector<std::string> seglist;
					while(std::getline(sstr, segment, ','))
					{
						 seglist.push_back(segment);
						 /* cout << segment << endl; */
					}
					move = stoi(seglist[0]);
					best_move = stoi(seglist[1]);
					swap_prob = stod(seglist[2]);
					value = stod(seglist[3]);
					exporter->save_planes(game.get());
					policy = torch::zeros(game->get_actions().size());
					if (ply==1 && with_swap){
						if (swap_prob>0.5){
							policy[game->action_from_board_location(best_move)] = 0;
							policy[policy.size(0)-1] = 1;
						}
						else if (swap_prob==0.5){
							policy[game->action_from_board_location(best_move)] = 0.5;
							policy[policy.size(0)-1] = 0.5;
						}
						else{
							policy[game->action_from_board_location(best_move)] = 1;
							policy[policy.size(0)-1] = 0;
						}
					}
					else{
						policy[game->action_from_board_location(best_move)] = 1;
					}
					exporter->gamePolicy.push_back(policy);
					exporter->gameBestMoveQ.push_back(value);
					exporter->gameValue.push_back(value);
					exporter->moves.push_back(game->action_from_board_location(best_move));
					vertex_move = game->action_from_board_location(move);
					if (vertex_move==-1){
						cout << "Having to skip a game :(" << endl;
						skip_this_game = true;
						continue;
					}
					/* assert(vertex_move>=0); */
					game->make_move(vertex_move,false,NOPLAYER,true);
					ply+=1;
					cur_idx+=1;
				}
			}
			f.close();
  }
	exporter->curSampleIdx = 0;
	cout << "Exporting ..." << endl;
	exporter->export_game_samples();
	cout << "Done" << endl;
}

#include "convert_moves_to_training_data.h"
#include "shannon_node_switching_game.h"
#include "rl/traindataexporter.h"
#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/api/include/torch/serialize.h>

void to_training_data(string &filename,int hex_size,string &output_folder){
	int move, best_move, ply, cur_idx;
	double swap_prob,value;
	string line;
	vector<int> policy_vec;
	torch::Tensor policy;
  ifstream f(filename);
	unique_ptr<TrainDataExporter> exporter = std::make_unique<TrainDataExporter>(output_folder);
	unique_ptr<Node_switching_game> game = std::make_unique<Node_switching_game>(hex_size);
	if (f.is_open())
		{
			while ( getline (f,line) )
			{
				if (line == "New game"){
					exporter->gameStartPtr.push_back(cur_idx);
					for (int i=ply;i>0;--i){
						exporter->gamePlysToEnd.push_back(i);
					}
					game->reset();
					ply = 0;
				}
				else{
					cout << line << endl;
					std::stringstream sstr(line);
					std::string segment;
					std::vector<std::string> seglist;
					while(std::getline(sstr, segment, ','))
					{
						 seglist.push_back(segment);
						 cout << segment << endl;
					}
					move = stoi(seglist[0]);
					best_move = stoi(seglist[1]);
					swap_prob = stod(seglist[2]);
					value = stod(seglist[3]);
					exporter->save_planes(game.get());
					policy = torch::zeros(game->get_actions().size());
					if (ply==1){
						policy[game->action_from_board_location(best_move)] = 1-swap_prob;
						policy[policy.size(0)-1] = swap_prob;
					}
					else{
						policy[game->action_from_board_location(best_move)] = 1;
					}
					exporter->gamePolicy.push_back(policy);
					exporter->gameBestMoveQ.push_back(value);
					exporter->gameValue.push_back(value);
					exporter->moves.push_back(game->action_from_board_location(best_move));
					game->make_move(game->action_from_board_location(move),false,NOPLAYER,true);
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

#include "playmode.h"
#include "agents/mctsagent.h"
#include "agents/rawnetagent.h"
#include "util.h"
#include "nn_api.h"
#include "shannon_node_switching_game.h"
#include "options.h"
#include <unistd.h>

void playmode(MCTSAgent * mctsAgent, RawNetAgent * rawAgent, SearchLimits * searchLimits, EvalInfo * evalInfo){
	string action, command, subcommand;
	torch::Tensor policy;
	int move, response, hex_size;
	istringstream iss;
	stringstream ss;
	bool show = false;
	bool use_mcts = false;
	std::vector<torch::Tensor> inputs, outputs;
	hex_size = Options["Hex_Size"];
	unique_ptr<Node_switching_game> game = std::make_unique<Node_switching_game>(hex_size);
	while (true){
		if (game->who_won()==noplayer){
			if (use_mcts){
				mctsAgent->set_search_settings(game.get(),searchLimits,evalInfo);
				mctsAgent->evaluate_board_state();
			}
			else{
				rawAgent->set_search_settings(game.get(),searchLimits,evalInfo);
				rawAgent->evaluate_board_state();
			}
			assert(evalInfo->legalMoves.size() == evalInfo->policyProbSmall.size());
			policy = torch::empty(evalInfo->policyProbSmall.size());
			for (int i=0;i<evalInfo->policyProbSmall.size();++i){
				policy[evalInfo->legalMoves[i]] = evalInfo->policyProbSmall[i];
			}
			if (use_mcts){
				cout << "Value: " << evalInfo->bestMoveQ[0] << endl;
			}
			else{
				cout << "Value: " << rawAgent->valueOutputs.item<double>() << endl;
			}
			cout << "Evaluation: ";
			for (int i=0;i<game->graph.num_vertices;++i){
				cout << "(" << game->graph.lprops[board_location][i] << "," << policy[i].item<double>() << ") ";
			}
			cout << endl;
		}
#ifndef NO_PLAY
		cout << "board_moves_maker: ";
		for (int i=0;i<game->board_moves_maker.size();++i){
			cout << game->board_moves_maker[i] << " ";
		}
		cout << endl;
		cout << "board_moves_breaker: ";
		for (int i=0;i<game->board_moves_breaker.size();++i){
			cout << game->board_moves_breaker[i] << " ";
		}
		cout << endl;
#endif
		vector<string> nodetext(game->graph.num_vertices);
		for (int i=0;i<game->graph.num_vertices;++i){
			ss.str(string());
			ss << game->graph.lprops[board_location][i] << "(" << setprecision(3) << policy[i].item<double>() << ")";
			nodetext[i] = ss.str();
		}
		game->graphviz_me(nodetext,"my_graph.dot");
		system("neato -Tpdf my_graph.dot -o my_graph.pdf");
		if (show){
			system("pkill -f 'mupdf my_graph.pdf'");
			system("mupdf my_graph.pdf &");
			usleep(100000U);
			system("bspc node -f west");
		}
		move = -1;
		cout << "readyok" << endl;
		std::getline(cin,command);
		iss = istringstream(command);
		getline(iss,action,' ');
		if (action == "show") show=!show;
		else if (action == "mcts") use_mcts=true;
		else if (action == "raw") use_mcts=false;
		else if (action == "reset"){
			if (getline(iss,subcommand,' ')){
				print_info(__LINE__,__FILE__,"resetting to",subcommand);
				game = std::make_unique<Node_switching_game>(stoi(subcommand));
				hex_size = stoi(subcommand);
			}
			else{
				print_info(__LINE__,__FILE__,"resetting normally",hex_size);
				game = std::make_unique<Node_switching_game>(hex_size);
			}
		}
		else if (action == "engine_move"){
			if (game->who_won()==noplayer){
				move = policy.argmax().item<int>();
				cout << "Engine_move: " << move << endl;
			}
		}
		else{
			move = game->vertex_from_board_location(stoi(action));
			if (move==-1){
				response = game->get_response(stoi(action),game->onturn==breaker);
				if (response == -1){
					cout << "Dead_move" << endl;
				}
				else{
					cout << "Response: " << response << endl;
				}
			}
		}
		if (move!=-1){
			game->make_move(move,false,maker,true);
		}
		Onturn winner = game->who_won();
		if (winner==maker){
			cout << "Winner: maker" << endl;
		}
		else if (winner==breaker){
			cout << "Winner: breaker" << endl;
		}
	}
}

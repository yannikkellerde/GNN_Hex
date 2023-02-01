#include "playmode.h"
#include "agents/mctsagent.h"
#include "agents/rawnetagent.h"
#include "util.h"
#include "nn_api.h"
#include "shannon_node_switching_game.h"
#include "options.h"
#include <unistd.h>

void playmode(MCTSAgent * mctsAgent, RawNetAgent * rawAgent, SearchLimits * searchLimits, EvalInfo * evalInfo){
	string action, command, subcommand, sgf, argument, sgf_path;
	torch::Tensor policy;
	int move, response, hex_size;
	istringstream iss;
	stringstream ss;
	bool show = false;
	bool autoeval = true;
	bool nexteval = true;
	bool use_mcts = false;
	std::vector<torch::Tensor> inputs, outputs;
	hex_size = Options["Hex_Size"];
	unique_ptr<Node_switching_game> game = std::make_unique<Node_switching_game>(hex_size,Options["Swap_Allowed"]);
	while (true){
		if (game->who_won()==NOPLAYER && nexteval){
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
			for (int i=2;i<game->graph.num_vertices;++i){
				cout << "(" << game->graph.lprops[BOARD_LOCATION][i] << "," << policy[i-2].item<double>() << ") ";
			}
			if (game->swap_allowed&&game->move_num==1){
				cout << "(swap," << policy[game->graph.num_vertices-2].item<double>() << ") ";
			}
			cout << endl;
		}
#ifndef NO_PLAY
		cout << "onturn: " << (game->onturn==RED?"red":"blue") << endl;
		cout << "board_moves_red: ";
		for (int i=0;i<game->board_moves_red.size();++i){
			cout << game->board_moves_red[i] << " ";
		}
		cout << endl;
		cout << "board_moves_blue: ";
		for (int i=0;i<game->board_moves_blue.size();++i){
			cout << game->board_moves_blue[i] << " ";
		}
		cout << endl;
#endif
		if (nexteval){
			vector<string> nodetext(game->graph.num_vertices);
			for (int i=2;i<game->graph.num_vertices;++i){
				ss.str(string());
				ss << game->graph.lprops[BOARD_LOCATION][i] << "(" << setprecision(3) << policy[i-2].item<double>() << ")";
				nodetext[i] = ss.str();
			}
			game->graphviz_me(nodetext,"my_graph.dot",game->graph);
			system("neato -Tpdf my_graph.dot -o my_graph.pdf");
			if (show){
				system("pkill -f 'mupdf my_graph.pdf'");
				system("mupdf my_graph.pdf &");
				usleep(100000U);
				system("bspc node -f west");
			}
		}
		move = -1;
		cout << "readyok" << endl;
		std::getline(cin,command);
		iss = istringstream(command);
		getline(iss,action,' ');
		nexteval = autoeval;
		if (action == "show") show=!show;
		else if (action == "set_autoeval"){
			getline(iss,argument,' ');
			if (argument == "true"){
				autoeval = true;
			}
			else{
				autoeval = false;
			}
		}
		else if (action == "eval"){
			nexteval = true;
		}
		else if (action == "sgf"){
			cout << "Loading sgf" << endl;
			getline(iss,sgf_path,' ');
			ifstream sgffile(sgf_path);
			getline(sgffile,sgf);
			game->load_sgf(sgf);
			sgffile.close();
			hex_size = game->board_size;
		}
		else if (action == "mcts") use_mcts=true;
		else if (action == "raw") use_mcts=false;
		else if (action == "switch") game->switch_onturn();
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
			if (game->who_won()==NOPLAYER){
				move = policy.argmax().item<int>();
				cout << "Engine_move: " << move << endl;
			}
		}
		else if (action == "swap"){
			if (game->who_won()==NOPLAYER&&game->swap_allowed&&game->move_num==1){
				move = game->graph.num_vertices;
			}
		}
		else{
			move = game->action_from_board_location(stoi(action));
			if (move==-1){
				response = game->get_response(stoi(action),game->onturn==BLUE);
				if (response == -1){
					cout << "Dead_move" << endl;
				}
				else{
					cout << "Response: " << response << endl;
				}
			}
		}
		if (move!=-1){
			game->make_move(move,false,RED,true);
		}
		Onturn winner = game->who_won();
		if (winner==RED){
			cout << "Winner: maker" << endl;
		}
		else if (winner==BLUE){
			cout << "Winner: breaker" << endl;
		}
	}
}

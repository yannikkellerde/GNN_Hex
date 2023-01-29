/*
	 CrazyAra, a deep learning chess variant engine
	 Copyright (C) 2018       Johannes Czech, Moritz Willig, Alena Beyer
	 Copyright (C) 2019-2020  Johannes Czech

	 This program is free software: you can redistribute it and/or modify
	 it under the terms of the GNU General Public License as published by
	 the Free Software Foundation, either version 3 of the License, or
	 (at your option) any later version.

	 This program is distributed in the hope that it will be useful,
	 but WITHOUT ANY WARRANTY; without even the implied warranty of
	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	 GNU General Public License for more details.

	 You should have received a copy of the GNU General Public License
	 along with this program.  If not, see <https://www.gnu.org/licenses/>.
	 */

/*
 * @file: selfplay.cpp
 * Created on 16.09.2019
 * @author: queensgambit
 *
 */

#include "selfplay.h"
#include <filesystem>
#include "config/searchlimits.h"
#include <iostream>
#include <fstream>
#include "util/blazeutil.h"
#include "util/randomgen.h"
#include "rl/gamepgn.h"
#include "util.h"
#include "util/speedcheck.h"
#include "util/statlogger.h"


void play_move_and_update(const EvalInfo& evalInfo, Node_switching_game* state, GamePGN& gamePGN, Onturn& gameResult, bool make_random_move)
{
	if (evalInfo.bestMove>=state->get_num_actions()){
		cout << gamePGN << endl;
		print_info(__LINE__,__FILE__,"num actions",state->get_num_actions(),"best move",evalInfo.bestMove);
		print_info(__LINE__,__FILE__,"legal moves",evalInfo.legalMoves);

	}
	string sanMove = state->format_action(evalInfo.bestMove);
	/* print_info(__LINE__,__FILE__,"Playing move",evalInfo.bestMove); */
	speedcheck.track_next("make move");
	if (make_random_move){
		state->make_move(state->get_random_action(),false,NOPLAYER,true);
	}
	else{
		state->make_move(evalInfo.bestMove,false,NOPLAYER,true);
	}
	speedcheck.stop_track("make move");
	gameResult = state->who_won();

	/* print_info(__LINE__,__FILE__,"gameResult",gameResult); */

	if (gameResult!=NOPLAYER) {
		sanMove += "#";
	}
	gamePGN.gameMoves.emplace_back(sanMove);
}

SelfPlay::SelfPlay(RawNetAgent* rawAgent, MCTSAgent* mctsAgent, SearchLimits* searchLimits, PlaySettings* playSettings, SearchSettings * searchSettings,
		RLSettings* rlSettings, OptionsMap& options):
	rawAgent(rawAgent), mctsAgent(mctsAgent), searchLimits(searchLimits), playSettings(playSettings), searchSettings(searchSettings),
	rlSettings(rlSettings), gameIdx(0), options(options), exporter("torch")
{
	folder = "data/"+to_string((int)options["First_Device_ID"])+"/";
	gamePGN.variant = "hex";
	time_t     now = time(0);
	struct tm  tstruct;
	char       date[80];
	tstruct = *localtime(&now);
	strftime(date, sizeof(date), "%Y.%m.%d %X", &tstruct);
	gamePGN.date = date;

	gamePGN.event = "SelfPlay";
	gamePGN.site = "Darmstadt, GER";
	gamePGN.round = "?";
	filenamePGNSelfplay = string("games") + string(".pgn");
	filenamePGNArena = folder+string("arena_games") + string(".pgn");
	fileNameGameIdx = string("gameIdx") + string(".txt");

	backupNodes = searchLimits->nodes;
	backupQValueWeight = mctsAgent->get_q_value_weight();
	backupDirichletEpsilon = mctsAgent->get_dirichlet_noise();
}

SelfPlay::~SelfPlay()
{
}

void SelfPlay::adjust_node_count(SearchLimits* searchLimits, int randInt)
{
	size_t maxRandomNodes = size_t(searchLimits->nodes * rlSettings->nodeRandomFactor);
	if (maxRandomNodes != 0) {
		searchLimits->nodes += (size_t(randInt) % maxRandomNodes) - maxRandomNodes / 2;
	}
}

bool SelfPlay::is_quick_search() {
	if (rlSettings->quickSearchProbability < 0.01f) {
		return false;
	}
	return float(rand()) / RAND_MAX < rlSettings->quickSearchProbability;
}

bool SelfPlay::is_resignation_allowed() {
	if (rlSettings->resignProbability < 0.01f) {
		return false;
	}
	return float(rand()) / RAND_MAX < rlSettings->resignProbability;
}

void SelfPlay::check_for_resignation(const bool allowResingation, const EvalInfo &evalInfo, const Node_switching_game* state, Onturn &gameResult)
{
	if (!allowResingation) {
		return;
	}
	if (evalInfo.bestMoveQ[0] < rlSettings->resignThreshold) {
		if (state->onturn == RED) {
			/* print_info(__LINE__,__FILE__,"Blue resigned"); */
			gameResult = RED;
		}
		else {
			/* print_info(__LINE__,__FILE__,"Red resigned"); */
			gameResult = BLUE;
		}
	}
}

void SelfPlay::reset_search_params(bool isQuickSearch)
{
	searchLimits->nodes = backupNodes;
	if (isQuickSearch) {
		mctsAgent->update_q_value_weight(backupQValueWeight);
		mctsAgent->update_dirichlet_epsilon(backupDirichletEpsilon);
	}
}

void generate_parallel_games(int num_games, NN_api * net, vector<unique_ptr<TrainDataExporter>> * exporters, int total_games_to_generate, SelfPlay * sp, int idx){
	chrono::steady_clock::time_point thread_start = chrono::steady_clock::now();
	std::filesystem::create_directory(sp->folder);
	std::filesystem::create_directory(sp->folder+to_string(idx));
	speedcheck.track_next("initialization");
	assert(total_games_to_generate>=num_games);
	int game_restarts_left = total_games_to_generate-num_games;
	srand(unsigned(int(time(nullptr))));
	bool changed;
	size_t generatedSamples = 0;
	vector<unique_ptr<Node_switching_game>> states;
	vector<GamePGN> pgns(num_games);
	vector<unique_ptr<MCTSAgent>> agents;
	vector<EvalInfo> evalInfos(num_games);
	vector<Onturn> gameResults(num_games);
	fill(gameResults.begin(),gameResults.end(),NOPLAYER);
	SearchLimits searchLimits(*sp->searchLimits);
	states.reserve(num_games);
	agents.reserve(num_games);
	for (int i=0;i<num_games;++i){
		states.push_back(init_starting_state_from_random_moves(pgns[i],0,i%2==0));
		pgns[i].starting_color = i%2==0?"Blue":"Red";
		agents.push_back(make_unique<MCTSAgent>(net,sp->searchSettings,sp->playSettings));
	}
	speedcheck.stop_track("initialization");
	do{
		speedcheck.track_next("init_eval");
		for (int i=0;i<num_games;++i){
			if (gameResults[i] == NOPLAYER){
				agents[i]->set_search_settings(states[i].get(), sp->searchLimits, &evalInfos[i]);
				agents[i]->create_unexpanded_root_nodes();
			}
		}
		speedcheck.stop_track("init_eval");
		speedcheck.track_next("root_prediction");
		if (net->node_features.size()>0){
			net->predict_stored();
			for (int i=0;i<num_games;++i){
				if (agents[i]->root_node_to_fill){
					agents[i]->fill_root_nn_results();
				}
			}
		}
		speedcheck.stop_track("root_prediction");
		speedcheck.track_next("mcts_loop");
		int randInt = rand();
		sp->adjust_node_count(sp->searchLimits, randInt); //ensure each game explores slightly different amount of nodes. (For variance)
		do{
			changed = false;
			for (int i=0;i<num_games;++i){
				if (gameResults[i] == NOPLAYER && agents[i]->do_more_eval()){
					agents[i]->eval_step_start();
					changed = true;
				}
			}
			if (changed&&net->node_features.size()>0){
				net->predict_stored();
				for (int i=0;i<num_games;++i){
					agents[i]->eval_step_stop();
				}
			}
		}
		while(changed);
		speedcheck.stop_track("mcts_loop");
		/* print_info(__LINE__,__FILE__,"bout to make a move"); */
		speedcheck.track_next("actual_moving");
		sp->searchLimits->nodes = sp->backupNodes;
		for (int i=0;i<num_games;++i){
			if (gameResults[i] == NOPLAYER){
				agents[i]->eval_stop();
				statlogger.log_mean_statistic("Avg nodes",evalInfos[i].nodes);
				
				if (sp->rlSettings->lowPolicyClipThreshold > 0) {
					sharpen_distribution(evalInfos[i].policyProbSmall, sp->rlSettings->lowPolicyClipThreshold);
				}
				(*exporters)[i]->save_sample(states[i].get(), evalInfos[i]);
				++generatedSamples;
				play_move_and_update(evalInfos[i], states[i].get(), pgns[i], gameResults[i], false);

#ifdef LOG_DEGREE_HIST
				vector<int> degree_stat = states[i]->graph.get_degree_histogram();
				for (int i=0;i<degree_stat.size();++i){
					statlogger.log_sum_statistic("degree_"+to_string(i),degree_stat[i]);
				}
#endif

				if (gameResults[i]!=NOPLAYER){
					statlogger.log_mean_statistic("red_wins",(gameResults[i]==RED));
					statlogger.log_mean_statistic("first_player_winrate",((gameResults[i]==RED&&(pgns[i].starting_color=="Red"))||(gameResults[i]==BLUE&&(pgns[i].starting_color == "Blue"))));
					statlogger.log_mean_statistic("num_moves",states[i]->move_num);
					(*exporters)[i]->new_game(gameResults[i]);
					sp->set_game_result_to_pgn(gameResults[i],pgns[i].starting_color=="Blue",pgns[i]);
#ifdef DO_DEBUG
					sp->write_game_to_pgn(sp->folder+to_string(idx)+"/"+sp->filenamePGNSelfplay, true, pgns[i]);
#else
					sp->write_game_to_pgn(sp->folder+to_string(idx)+"/"+sp->filenamePGNSelfplay, false, pgns[i]);
#endif
					clean_up(pgns[i], agents[i].get());
					if (game_restarts_left>0){
						game_restarts_left-=1;
						gameResults[i] = NOPLAYER;
						states[i]->reset();
						if (game_restarts_left%2==0){
							states[i]->switch_onturn();
							pgns[i].starting_color = "Blue";
						}
						else{
							pgns[i].starting_color = "Red";
						}
					}
				}
			}
		}
		speedcheck.stop_track("actual_moving");
	}
	while(any_of(gameResults.begin(),gameResults.end(),[](Onturn res){return res==NOPLAYER;}));
	TrainDataExporter final_exporter = TrainDataExporter::merged_from_many(*exporters,sp->folder+to_string(idx)+"/torch");
	final_exporter.export_game_samples();
	statlogger.log_mean_statistic("samples thread/sec",final_exporter.node_features.size()/(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now()-thread_start).count()/1000.0f));
}


void generate_parallel_arena_games(int num_games, NN_api * net_player, NN_api * net_contender, int total_games_to_generate, SelfPlay * sp){
	assert(total_games_to_generate>=num_games);
	int game_restarts_left = total_games_to_generate-num_games;
	srand(unsigned(int(time(nullptr))));
	bool changed;
	size_t generatedSamples = 0;
	MCTSAgent * agent;
	vector<unique_ptr<Node_switching_game>> states;
	vector<GamePGN> pgns(num_games);
	vector<unique_ptr<MCTSAgent>> player_agents;
	vector<unique_ptr<MCTSAgent>> contender_agents;
	vector<EvalInfo> evalInfos(num_games);
	vector<Onturn> gameResults(num_games);
	vector<bool> contender_is_active(num_games);
	vector<bool> contender_is_red(num_games);
	fill(gameResults.begin(),gameResults.end(),NOPLAYER);
	SearchLimits searchLimits(*sp->searchLimits);
	states.reserve(num_games);
	player_agents.reserve(num_games);
	contender_agents.reserve(num_games);
	for (int i=0;i<num_games;++i){
		pgns[i].starting_color = i<total_games_to_generate/2?"Blue":"Red";
		states.push_back(init_starting_state_from_random_moves(pgns[i],0,pgns[i].starting_color=="Blue"));
		contender_agents.push_back(make_unique<MCTSAgent>(net_contender,sp->searchSettings,sp->playSettings));
		player_agents.push_back(make_unique<MCTSAgent>(net_player,sp->searchSettings,sp->playSettings));
		contender_is_active[i] = i%2==0;
		contender_is_red[i] = (contender_is_active[i]&&pgns[i].starting_color=="Red")||(!contender_is_active[i]&&pgns[i].starting_color=="Blue");
		pgns[i].red = contender_is_red[i]?"Contender":"Previous";
		pgns[i].blue = contender_is_red[i]?"Previous":"Contender";
	}
	do{
		for (int i=0;i<num_games;++i){
			if (gameResults[i] == NOPLAYER){
				if (contender_is_active[i])
					agent = contender_agents[i].get();
				else
					agent = player_agents[i].get();
				
				agent->set_search_settings(states[i].get(), sp->searchLimits, &evalInfos[i]);
				agent->create_unexpanded_root_nodes();
			}
		}
		if (net_player->node_features.size()>0){
			net_player->predict_stored();
		}
		if (net_contender->node_features.size()>0){
			net_contender->predict_stored();
		}
		for (int i=0;i<num_games;++i){
			if (contender_is_active[i])
				agent = contender_agents[i].get();
			else
				agent = player_agents[i].get();
			if (agent->root_node_to_fill){
				agent->fill_root_nn_results();
			}
		}
		do{
			changed = false;
			for (int i=0;i<num_games;++i){
				if (contender_is_active[i])
					agent = contender_agents[i].get();
				else
					agent = player_agents[i].get();
				if (gameResults[i] == NOPLAYER && agent->do_more_eval()){
					agent->eval_step_start();
					changed = true;
				}
			}
			if (changed&&net_player->node_features.size()>0){
				net_player->predict_stored();
			}
			if (changed&&net_contender->node_features.size()>0){
				net_contender->predict_stored();
			}
			for (int i=0;i<num_games;++i){
				if (contender_is_active[i])
					agent = contender_agents[i].get();
				else
					agent = player_agents[i].get();
				agent->eval_step_stop();
			}
		}
		while(changed);

		for (int i=0;i<num_games;++i){
			if (gameResults[i] == NOPLAYER){
				if (contender_is_active[i])
					agent = contender_agents[i].get();
				else
					agent = player_agents[i].get();
				agent->eval_stop();
				statlogger.log_mean_statistic("Avg nodes",evalInfos[i].nodes);
				
				if (sp->rlSettings->lowPolicyClipThreshold > 0) {
					sharpen_distribution(evalInfos[i].policyProbSmall, sp->rlSettings->lowPolicyClipThreshold);
				}
				play_move_and_update(evalInfos[i], states[i].get(), pgns[i], gameResults[i], false);
				contender_is_active[i] = !contender_is_active[i];

				if (gameResults[i]!=NOPLAYER){
					statlogger.log_mean_statistic("contender_wins",(gameResults[i]==RED&&contender_is_red[i])||(gameResults[i]==BLUE&&!contender_is_red[i]));
					statlogger.log_mean_statistic("red_wins",(gameResults[i]==RED));
					statlogger.log_mean_statistic("first_player_winrate",((gameResults[i]==RED&&pgns[i].starting_color=="Red"))||(gameResults[i]==BLUE&&pgns[i].starting_color=="Blue"));
					statlogger.log_mean_statistic("num_moves",states[i]->move_num);
					sp->set_game_result_to_pgn(gameResults[i],pgns[i].starting_color=="Blue",pgns[i]);
#ifdef DO_DEBUG
					sp->write_game_to_pgn(sp->filenamePGNArena, true, pgns[i]);
#else
					sp->write_game_to_pgn(sp->filenamePGNArena, false, pgns[i]);
#endif
					clean_up(pgns[i], agent);
					if (game_restarts_left>0){
						game_restarts_left-=1;
						gameResults[i] = NOPLAYER;
						states[i]->reset();
						contender_is_active[i] = (total_games_to_generate-game_restarts_left)%2==0;
						if (game_restarts_left>total_games_to_generate/2){
							states[i]->switch_onturn();
							pgns[i].starting_color = "Blue";
						}
						else pgns[i].starting_color = "Red";
						contender_is_red[i] = (contender_is_active[i]&&pgns[i].starting_color=="Red")||(!contender_is_active[i]&&pgns[i].starting_color=="Blue");
						pgns[i].red = contender_is_red[i]?"Contender":"Previous";
						pgns[i].blue = contender_is_red[i]?"Previous":"Contender";
					}
				}
			}
		}
	}
	while(any_of(gameResults.begin(),gameResults.end(),[](Onturn res){return res==NOPLAYER;}));
}


void clean_up(GamePGN& gamePGN, MCTSAgent* mctsAgent)
{
	gamePGN.new_game();
	mctsAgent->clear_game_history();
}

void SelfPlay::write_game_to_pgn(const std::string& pgnFileName, bool verbose, GamePGN & pgn)
{
	ofstream pgnFile;
	pgnFile.open(pgnFileName, std::ios_base::app);
	if (verbose) {
		cout << endl << pgn << endl;
	}
	pgnFile << pgn << endl;
	pgnFile.close();
}

void SelfPlay::set_game_result_to_pgn(Onturn res,bool bluestarts, GamePGN & pgn)
{
	if ((bluestarts && res==RED)||((!bluestarts) && res==BLUE))
		pgn.result = "0-1";
	else pgn.result = "1-0";
}

void SelfPlay::export_number_generated_games() const
{
	ofstream gameIdxFile;
	gameIdxFile.open(fileNameGameIdx);
	gameIdxFile << gameIdx;
	gameIdxFile.close();
}

void SelfPlay::go(size_t num_threads, size_t parallel_games_per_thread, size_t total_games_per_thread, vector<unique_ptr<NN_api>> & netBatches)
{
	chrono::steady_clock::time_point selfplay_start = chrono::steady_clock::now();
	gameIdx = 0;
	vector<vector<unique_ptr<TrainDataExporter>>> exporters(num_threads);
	thread** threads = new thread*[num_threads];

	for (int i = 0; i<num_threads;++i){
		for (int j = 0; j<parallel_games_per_thread;++j){
			exporters[i].push_back(make_unique<TrainDataExporter>(exporter.output_folder));
		}
		threads[i] = new thread(generate_parallel_games, parallel_games_per_thread, netBatches[i].get(), &exporters[i], total_games_per_thread, this, i);
	}

	for (size_t i = 0; i < num_threads; ++i) {
			threads[i]->join();
	}
	/* exporter = TrainDataExporter::merged_from_many(exporters,exporter.output_folder); */
	/* exporter.export_game_samples(); */
	delete[] threads;
	statlogger.log_mean_statistic("samples per sec",statlogger.mean_statistics["samples thread/sec"].first*num_threads);
	statlogger.log_mean_statistic("games per sec",(num_threads*total_games_per_thread)/(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now()-selfplay_start).count()/1000.0f));
	statlogger.print_statistics(cout);
}

void SelfPlay::go_arena(vector<unique_ptr<NN_api>> & net_player_batches,vector<unique_ptr<NN_api>> & net_contender_batches, size_t num_games, size_t total_games_per_thread, size_t num_threads)
{
	statlogger.reset_key("contender_wins");
	thread** threads = new thread*[num_threads];

	for (size_t i = 0; i<num_threads;++i){
		threads[i] = new thread(generate_parallel_arena_games,num_games, net_player_batches[i].get(), net_contender_batches[i].get(), total_games_per_thread, this);
	}
	for (size_t i = 0; i < num_threads; ++i) {
			threads[i]->join();
	}
	delete[] threads;
}

unique_ptr<Node_switching_game> init_starting_state_from_raw_policy(RawNetAgent &rawAgent, size_t plys, GamePGN &gamePGN, float rawPolicyProbTemp)
{
	unique_ptr<Node_switching_game> state= make_unique<Node_switching_game>(Options["Hex_Size"],Options["Swap_Allowed"]);

	for (size_t ply = 0; ply < plys; ++ply) {
		EvalInfo eval;
		rawAgent.set_search_settings(state.get(), nullptr, &eval);
		rawAgent.evaluate_board_state();
		apply_raw_policy_temp(eval, rawPolicyProbTemp);
		const size_t moveIdx = random_choice(eval.policyProbSmall);
		eval.bestMove = eval.legalMoves[moveIdx];

		gamePGN.gameMoves.push_back(state->format_action(eval.legalMoves[moveIdx]));
		speedcheck.track_next("make move");
		state->make_move(eval.bestMove,false,NOPLAYER,true);
		speedcheck.stop_track("make move");
	}
	return state;
}

unique_ptr<Node_switching_game> init_starting_state_from_random_moves(GamePGN &gamePGN, int num_actions, bool blue_starts)
{
	unique_ptr<Node_switching_game> state = make_unique<Node_switching_game>(Options["Hex_Size"],Options["Swap_Allowed"]);
	if (blue_starts){
		state->switch_onturn();
	}
	speedcheck.track_next("make move");
	for (int i=0;i<num_actions;++i) {
		int action = state->get_random_action();
		gamePGN.gameMoves.push_back(state->format_action(action));
		state->make_move(action,false,NOPLAYER,true);
	}
	speedcheck.stop_track("make move");
	return state;
}

unique_ptr<Node_switching_game> init_starting_state_from_fixed_moves(GamePGN &gamePGN, vector<int> actions, bool blue_starts)
{
	unique_ptr<Node_switching_game> state = make_unique<Node_switching_game>(Options["Hex_Size"],Options["Swap_Allowed"]);
	if (blue_starts){
		state->switch_onturn();
	}
	speedcheck.track_next("make move");
	for (int action : actions) {
		gamePGN.gameMoves.push_back(state->format_action(action));
		state->make_move(action,false,NOPLAYER,true);
	}
	speedcheck.stop_track("make move");
	return state;
}

size_t clip_ply(size_t ply, size_t maxPly)
{
	if (ply > maxPly) {
		return size_t(rand()) % maxPly;
	}
	return ply;
}

void apply_raw_policy_temp(EvalInfo &eval, float rawPolicyProbTemp)
{
	if (float(rand()) / RAND_MAX < rawPolicyProbTemp) {
		float temp = 2.0f;
		const float prob = float(rand()) / INT_MAX;
		if (prob < 0.05f) {
			temp = 10.0f;
		}
		else if (prob < 0.25f) {
			temp = 5.0f;
		}
		apply_temperature(eval.policyProbSmall, temp);
	}
}

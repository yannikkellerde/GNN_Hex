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

#include "config/searchlimits.h"
#include <iostream>
#include <fstream>
#include "util/blazeutil.h"
#include "util/randomgen.h"
#include "rl/gamepgn.h"
#include "util.h"
#include "util/speedcheck.h"


void play_move_and_update(const EvalInfo& evalInfo, Node_switching_game* state, GamePGN& gamePGN, Onturn& gameResult, bool make_random_move)
{
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
	rlSettings(rlSettings), gameIdx(0), gamesPerMin(0), samplesPerMin(0), options(options), folder("data/"), exporter("data/torch")
{
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
	filenamePGNSelfplay = folder+string("games") + string(".pgn");
	filenamePGNArena = folder+string("arena_games") + string(".pgn");
	fileNameGameIdx = folder + string("gameIdx") + string(".txt");

	// delete content of files
	ofstream pgnFile;
	pgnFile.open(filenamePGNSelfplay, std::ios_base::trunc);
	pgnFile.close();
	ofstream idxFile;
	idxFile.open(fileNameGameIdx, std::ios_base::trunc);
	idxFile.close();

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

void generate_parallel_games(int num_games, NN_api * net, vector<unique_ptr<TrainDataExporter>> * exporters, map<string,double> * stats, int total_games_to_generate, SelfPlay * sp){
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
				agents[i]->set_search_settings(states[i].get(), &searchLimits, &evalInfos[i]);
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
		do{
			changed = false;
			for (int i=0;i<num_games;++i){
				if (gameResults[i] == NOPLAYER && agents[i]->do_more_eval()){
					int randInt = rand();
					sp->adjust_node_count(&searchLimits, randInt); //ensure each game explores slightly different amount of nodes. (For variance)
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
		for (int i=0;i<num_games;++i){
			if (gameResults[i] == NOPLAYER){
				agents[i]->eval_stop();
				if (sp->rlSettings->lowPolicyClipThreshold > 0) {
					sharpen_distribution(evalInfos[i].policyProbSmall, sp->rlSettings->lowPolicyClipThreshold);
				}
				(*exporters)[i]->save_sample(states[i].get(), evalInfos[i]);
				++generatedSamples;
				play_move_and_update(evalInfos[i], states[i].get(), pgns[i], gameResults[i], false);
				sp->searchLimits->nodes = sp->backupNodes;
				if (gameResults[i]!=NOPLAYER){
					(*stats)["red_wins"]+=(gameResults[i]==RED);
					(*stats)["blue_wins"]+=(gameResults[i]==BLUE);
					(*stats)["first_player_wins"]+=((gameResults[i]==RED&&(i%2==1))||(gameResults[i]==BLUE&&(i%2==0)));
					(*stats)["second_player_wins"]+=((gameResults[i]==RED&&(i%2==0))||(gameResults[i]==BLUE&&(i%2==1)));
					(*stats)["num_moves"]+=states[i]->move_num;
					if (game_restarts_left>0){
						(*exporters)[i]->new_game(gameResults[i]);
						sp->set_game_result_to_pgn(gameResults[i],i%2==0,pgns[i]);
#ifdef DO_DEBUG
						sp->write_game_to_pgn(sp->filenamePGNSelfplay, true, pgns[i]);
#else
						sp->write_game_to_pgn(sp->filenamePGNSelfplay, false, pgns[i]);
#endif
						clean_up(pgns[i], agents[i].get());
						game_restarts_left-=1;
						gameResults[i] = NOPLAYER;
						states[i]->reset();
						if (game_restarts_left%2==0){
							states[i]->switch_onturn();
						}
					}
				}
			}
		}
		speedcheck.stop_track("actual_moving");
	}
	while(any_of(gameResults.begin(),gameResults.end(),[](Onturn res){return res==NOPLAYER;}));
	for (int i=0;i<num_games;++i){
		(*exporters)[i]->new_game(gameResults[i]);
		sp->set_game_result_to_pgn(gameResults[i],i%2==0,pgns[i]);
#ifdef DO_DEBUG
		sp->write_game_to_pgn(sp->filenamePGNSelfplay, true, pgns[i]);
#else
		sp->write_game_to_pgn(sp->filenamePGNSelfplay, false, pgns[i]);
#endif
		clean_up(pgns[i], agents[i].get());
	}
}

void SelfPlay::generate_game(bool verbose)
{
	chrono::steady_clock::time_point gameStartTime = chrono::steady_clock::now();

	size_t ply = size_t(random_exponential<float>(1.0f/playSettings->meanInitPly) + 0.5f);
	ply = clip_ply(ply, playSettings->maxInitPly);

	srand(unsigned(int(time(nullptr))));
	// load position from file if epd filepath was set
	/* unique_ptr<Node_switching_game> state = init_starting_state_from_raw_policy(*rawAgent, ply, gamePGN, rlSettings->rawPolicyProbabilityTemperature); */
	// random starting move
	unique_ptr<Node_switching_game> state = init_starting_state_from_random_moves(gamePGN,0,gameIdx%2==0);
	gamePGN.starting_color = gameIdx%2==0?"Blue":"Red";
	assert (state->who_won()==NOPLAYER); // If this fails, ply is to high.
	EvalInfo evalInfo;
	Onturn gameResult;

	size_t generatedSamples = 0;
	const bool allowResignation = is_resignation_allowed();
	do {
		searchLimits->startTime = current_time();
		const int randInt = rand();
		const bool isQuickSearch = is_quick_search();

		if (isQuickSearch) {
			searchLimits->nodes = rlSettings->quickSearchNodes;
			mctsAgent->update_q_value_weight(rlSettings->quickSearchQValueWeight);
			mctsAgent->update_dirichlet_epsilon(rlSettings->quickDirichletEpsilon);
		}
		adjust_node_count(searchLimits, randInt);
		mctsAgent->set_search_settings(state.get(), searchLimits, &evalInfo);
		mctsAgent->perform_action();
		if (rlSettings->reuseTreeForSelpay) {
			mctsAgent->apply_move_to_tree(evalInfo.bestMove, true);
		}

		if (!isQuickSearch) {
			if (rlSettings->lowPolicyClipThreshold > 0) {
				sharpen_distribution(evalInfo.policyProbSmall, rlSettings->lowPolicyClipThreshold);
			}
			exporter.save_sample(state.get(), evalInfo);
			++generatedSamples;
		}
		play_move_and_update(evalInfo, state.get(), gamePGN, gameResult, false);
		reset_search_params(isQuickSearch);
		check_for_resignation(allowResignation, evalInfo, state.get(), gameResult);
	}
	while(gameResult == NOPLAYER);
	stats["red_wins"]+=(gameResult==RED);
	stats["blue_wins"]+=(gameResult==BLUE);
	stats["first_player_wins"]+=((gameResult==RED&&(gameIdx%2==1))||(gameResult==BLUE&&(gameIdx%2==0)));
	stats["second_player_wins"]+=((gameResult==RED&&(gameIdx%2==0))||(gameResult==BLUE&&(gameIdx%2==1)));
	stats["num_moves"]+=state->move_num;

	// Finish up exporter work. Does not export yet.
	exporter.new_game(gameResult);

	set_game_result_to_pgn(gameResult,gameIdx%2==0,gamePGN);
	write_game_to_pgn(filenamePGNSelfplay, verbose, gamePGN);
	clean_up(gamePGN, mctsAgent);
	/* print_info(__LINE__,__FILE__,"Nodes per second",evalInfo.calculate_nps()); */

	// measure time statistics
	if (verbose) {
		const float elapsedTimeMin = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - gameStartTime).count() / 60000.f;
		speed_statistic_report(elapsedTimeMin, generatedSamples);
	}
	if (gameIdx%50==0){
		cout << "50 games" << endl;
	}
	++gameIdx;
}

void SelfPlay::print_stats(){
	cout << "STATISTIC: " << "red_blue_winrate " << stats["red_wins"]/(stats["red_wins"]+stats["blue_wins"]) << endl;
	cout << "STATISTIC: " << "first_player_winrate " << stats["first_player_wins"]/(stats["first_player_wins"]+stats["second_player_wins"]) << endl;
	cout << "STATISTIC: " << "avg_game_length " << stats["num_moves"]/gameIdx << endl;
	stats.clear();
}

Onturn SelfPlay::generate_arena_game(MCTSAgent* redPlayer, MCTSAgent* bluePlayer, bool verbose, vector<int>& starting_moves, bool blue_starts)
{
	gamePGN.red = "Red";
	gamePGN.blue = "Blue";
	gamePGN.starting_color = blue_starts?"Blue":"Red";
	unique_ptr<Node_switching_game> state = init_starting_state_from_fixed_moves(gamePGN,starting_moves,blue_starts);
	EvalInfo evalInfo;

	MCTSAgent* activePlayer;
	MCTSAgent* passivePlayer;
	// preserve the current active states
	Onturn gameResult;
	do {
		searchLimits->startTime = current_time();
		if (state->onturn == RED) {
			activePlayer = redPlayer;
			passivePlayer = bluePlayer;
		}
		else {
			activePlayer = bluePlayer;
			passivePlayer = redPlayer;
		}
		activePlayer->set_search_settings(state.get(), searchLimits, &evalInfo);
		activePlayer->perform_action();
		activePlayer->apply_move_to_tree(evalInfo.bestMove, true);
		if (state->move_num != 0) {
			passivePlayer->apply_move_to_tree(evalInfo.bestMove, false);
		}
		play_move_and_update(evalInfo, state.get(), gamePGN, gameResult, false);
	}
	while(gameResult == NOPLAYER);
	set_game_result_to_pgn(gameResult,blue_starts,gamePGN);
	write_game_to_pgn(filenamePGNArena, verbose, gamePGN);
	clean_up(gamePGN, redPlayer);
	bluePlayer->clear_game_history();
	return gameResult;
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

void SelfPlay::reset_speed_statistics()
{
	gameIdx = 0;
	gamesPerMin = 0;
	samplesPerMin = 0;
}

void SelfPlay::speed_statistic_report(float elapsedTimeMin, size_t generatedSamples)
{
	// compute running cumulative average
	gamesPerMin = (gameIdx * gamesPerMin + (1 / elapsedTimeMin)) / (gameIdx + 1);
	samplesPerMin = (gameIdx * samplesPerMin + (generatedSamples / elapsedTimeMin)) / (gameIdx + 1);

	cout << "    games    |  games/min  | samples/min " << endl
		<< "-------------+-------------+-------------" << endl
		<< std::setprecision(5)
		<< setw(13) << gameIdx << '|'
		<< setw(13) << gamesPerMin << '|'
		<< setw(13) << samplesPerMin << endl << endl;
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
	reset_speed_statistics();
	vector<vector<unique_ptr<TrainDataExporter>>> exporters(num_threads);
	vector<map<string,double>> all_stats(num_threads); 
	thread** threads = new thread*[num_threads];

	for (size_t i = 0; i<num_threads;++i){
		for (size_t j = 0; j<parallel_games_per_thread;++j){
			exporters[i].push_back(make_unique<TrainDataExporter>(exporter.output_folder));
		}
		threads[i] = new thread(generate_parallel_games, parallel_games_per_thread, netBatches[i].get(), &exporters[i], &all_stats[i], total_games_per_thread, this);
	}

	for (size_t i = 0; i < num_threads; ++i) {
			threads[i]->join();
	}
	print_stats();
	speedcheck.track_next("file_export");
	ofstream gameIdxFile;
	gameIdxFile.open(fileNameGameIdx);
	gameIdxFile << exporter.gameStartPtr.size();
	gameIdxFile.close();
	exporter = TrainDataExporter::merged_from_many(exporters,exporter.output_folder);
	exporter.export_game_samples();
	delete[] threads;
	speedcheck.stop_track("file_export");
}

TournamentResult SelfPlay::go_arena(MCTSAgent *mctsContender, size_t numberOfGames)
{
	unique_ptr<Node_switching_game> tmp_game = make_unique<Node_switching_game>(Options["Hex_Size"]);

	// generate starting moves to ensure that both agents get the same starting moves.
	vector<int> actions = tmp_game->get_actions();
	std::random_device rd = std::random_device {}; 
	std::default_random_engine rng = std::default_random_engine { rd() };
	std::shuffle(std::begin(actions), std::end(actions), rng);

	TournamentResult tournamentResult;
	tournamentResult.playerA = mctsContender->get_name();
	tournamentResult.playerB = mctsAgent->get_name();
	Onturn gameResult;

	for (size_t idx = 0; idx < numberOfGames; ++idx) {
		vector<int> starting_moves = {actions[(int)std::floor(idx/4)%actions.size()]};
		if (idx % 2 == 0) {
			// use default or in case of chess960 a random starting position
			gameResult = generate_arena_game(mctsContender, mctsAgent, true, starting_moves,idx%4==0);
			if (gameResult == RED) {
				++tournamentResult.numberWins;
			}
			else if (gameResult == BLUE){
				++tournamentResult.numberLosses;
			}
		}
		else {
			// use same starting position as before stored via gamePGN.fen
			gameResult = generate_arena_game(mctsAgent, mctsContender, true, starting_moves,(idx+1)%4==0);
			if (gameResult == RED) {
				++tournamentResult.numberLosses;
			}
			else if (gameResult == BLUE){
				++tournamentResult.numberWins;
			}
		}
		assert (gameResult!=NOPLAYER);
	}
	return tournamentResult;
}

unique_ptr<Node_switching_game> init_starting_state_from_raw_policy(RawNetAgent &rawAgent, size_t plys, GamePGN &gamePGN, float rawPolicyProbTemp)
{
	unique_ptr<Node_switching_game> state= make_unique<Node_switching_game>(Options["Hex_Size"]);

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
	unique_ptr<Node_switching_game> state = make_unique<Node_switching_game>(Options["Hex_Size"]);
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
	unique_ptr<Node_switching_game> state = make_unique<Node_switching_game>(Options["Hex_Size"]);
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

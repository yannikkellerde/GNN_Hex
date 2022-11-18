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


void play_move_and_update(const EvalInfo& evalInfo, Node_switching_game* state, GamePGN& gamePGN, Onturn& gameResult)
{
    string sanMove = state->format_action(evalInfo.bestMove);
		print_info(__LINE__,__FILE__,"Playing move",evalInfo.bestMove);
		speedcheck.track_next("make move");
    state->make_move(evalInfo.bestMove,false,noplayer,true);
		speedcheck.stop_track("make move");
    gameResult = state->who_won();

		print_info(__LINE__,__FILE__,"gameResult",gameResult);
		print_info(__LINE__,__FILE__,"makerwon",state->maker_won);

    if (gameResult!=noplayer) {
			sanMove += "#";
    }
    gamePGN.gameMoves.emplace_back(sanMove);
}

SelfPlay::SelfPlay(RawNetAgent* rawAgent, MCTSAgent* mctsAgent, SearchLimits* searchLimits, PlaySettings* playSettings,
                   RLSettings* rlSettings, OptionsMap& options):
    rawAgent(rawAgent), mctsAgent(mctsAgent), searchLimits(searchLimits), playSettings(playSettings),
    rlSettings(rlSettings), gameIdx(0), gamesPerMin(0), samplesPerMin(0), options(options)
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
		string folder = "data/";
    this->exporter = new TrainDataExporter(folder+"torch",
                                           rlSettings->numberChunks, rlSettings->chunkSize);
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
    delete exporter;
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
        if (state->onturn == maker) {
						print_info(__LINE__,__FILE__,"Breaker resigned");
            gameResult = maker;
        }
        else {
						print_info(__LINE__,__FILE__,"Maker resigned");
            gameResult = breaker;
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

void SelfPlay::generate_game(bool verbose)
{
    chrono::steady_clock::time_point gameStartTime = chrono::steady_clock::now();

    size_t ply = size_t(random_exponential<float>(1.0f/playSettings->meanInitPly) + 0.5f);
    ply = clip_ply(ply, playSettings->maxInitPly);

    srand(unsigned(int(time(nullptr))));
    // load position from file if epd filepath was set
    unique_ptr<Node_switching_game> state = init_starting_state_from_raw_policy(*rawAgent, ply, gamePGN, rlSettings->rawPolicyProbabilityTemperature);
		assert (state->who_won()==noplayer); // If this fails, ply is to high.
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

        if (!isQuickSearch && !exporter->is_file_full()) {
            if (rlSettings->lowPolicyClipThreshold > 0) {
                sharpen_distribution(evalInfo.policyProbSmall, rlSettings->lowPolicyClipThreshold);
            }
            exporter->save_sample(state.get(), evalInfo);
            ++generatedSamples;
        }
        play_move_and_update(evalInfo, state.get(), gamePGN, gameResult);
        reset_search_params(isQuickSearch);
        check_for_resignation(allowResignation, evalInfo, state.get(), gameResult);
    }
    while(gameResult == noplayer);

		// Finish up exporter work. Does not export yet.
    exporter->new_game(gameResult);

    set_game_result_to_pgn(gameResult);
    write_game_to_pgn(filenamePGNSelfplay, verbose);
    clean_up(gamePGN, mctsAgent);

    // measure time statistics
    if (verbose) {
        const float elapsedTimeMin = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - gameStartTime).count() / 60000.f;
        speed_statistic_report(elapsedTimeMin, generatedSamples);
    }
    ++gameIdx;
}

Onturn SelfPlay::generate_arena_game(MCTSAgent* makerPlayer, MCTSAgent* breakerPlayer, bool verbose)
{
    gamePGN.white = "Maker";
    gamePGN.black = "Breaker";
    unique_ptr<Node_switching_game> state = make_unique<Node_switching_game>(Options["Hex_Size"]);
    EvalInfo evalInfo;

    MCTSAgent* activePlayer;
    MCTSAgent* passivePlayer;
    // preserve the current active states
    Onturn gameResult;
    do {
        searchLimits->startTime = current_time();
        if (state->onturn == maker) {
            activePlayer = makerPlayer;
            passivePlayer = breakerPlayer;
        }
        else {
            activePlayer = breakerPlayer;
            passivePlayer = makerPlayer;
        }
        activePlayer->set_search_settings(state.get(), searchLimits, &evalInfo);
        activePlayer->perform_action();
        activePlayer->apply_move_to_tree(evalInfo.bestMove, true);
        if (state->move_num != 0) {
            passivePlayer->apply_move_to_tree(evalInfo.bestMove, false);
        }
        play_move_and_update(evalInfo, state.get(), gamePGN, gameResult);
    }
    while(gameResult == noplayer);
    set_game_result_to_pgn(gameResult);
    write_game_to_pgn(filenamePGNArena, verbose);
    clean_up(gamePGN, makerPlayer);
    breakerPlayer->clear_game_history();
    return gameResult;
}

void clean_up(GamePGN& gamePGN, MCTSAgent* mctsAgent)
{
    gamePGN.new_game();
    mctsAgent->clear_game_history();
}

void SelfPlay::write_game_to_pgn(const std::string& pgnFileName, bool verbose)
{
    ofstream pgnFile;
    pgnFile.open(pgnFileName, std::ios_base::app);
    if (verbose) {
        cout << endl << gamePGN << endl;
    }
    pgnFile << gamePGN << endl;
    pgnFile.close();
}

void SelfPlay::set_game_result_to_pgn(Onturn res)
{
    gamePGN.result = result[res];
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


void SelfPlay::go(size_t numberOfGames)
{
    reset_speed_statistics();
    gamePGN.white = mctsAgent->get_name();
    gamePGN.black = mctsAgent->get_name();

    if (numberOfGames == 0) {
        while(!exporter->is_file_full()) {
            generate_game(true);
        }
    }
    else {
        for (size_t idx = 0; idx < numberOfGames; ++idx) {
            generate_game(true);
        }
    }
    exporter->export_game_samples();
    export_number_generated_games();
}

TournamentResult SelfPlay::go_arena(MCTSAgent *mctsContender, size_t numberOfGames)
{
    TournamentResult tournamentResult;
    tournamentResult.playerA = mctsContender->get_name();
    tournamentResult.playerB = mctsAgent->get_name();
    Onturn gameResult;

    for (size_t idx = 0; idx < numberOfGames; ++idx) {
        if (idx % 2 == 0) {
            // use default or in case of chess960 a random starting position
            gameResult = generate_arena_game(mctsContender, mctsAgent, true);
            if (gameResult == maker) {
                ++tournamentResult.numberWins;
            }
            else if (gameResult == breaker){
                ++tournamentResult.numberLosses;
            }
        }
        else {
            // use same starting position as before stored via gamePGN.fen
            gameResult = generate_arena_game(mctsAgent, mctsContender, true);
            if (gameResult == breaker) {
                ++tournamentResult.numberWins;
            }
            else if (gameResult == maker){
                ++tournamentResult.numberLosses;
            }
        }
				assert (gameResult!=noplayer);
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

				print_info(__LINE__, __FILE__, "init best move ",eval.bestMove);
				print_info(__LINE__, __FILE__, "init move idx",eval.legalMoves);
				print_info(__LINE__, __FILE__, "pps size ",eval.policyProbSmall.size());
				print_info(__LINE__, __FILE__, "legal moves ",eval.legalMoves);
				gamePGN.gameMoves.push_back(state->format_action(eval.legalMoves[moveIdx]));
				speedcheck.track_next("make move");
				state->make_move(eval.bestMove,false,noplayer,true);
				speedcheck.stop_track("make move");
    }
    return state;
}

unique_ptr<Node_switching_game> init_starting_state_from_fixed_move(GamePGN &gamePGN, bool is960, const vector<int>& actions)
{
    unique_ptr<Node_switching_game> state= make_unique<Node_switching_game>(Options["Hex_Size"]);
		speedcheck.track_next("make move");
    for (int action : actions) {
        gamePGN.gameMoves.push_back(state->format_action(action));
        state->make_move(action,false,noplayer,true);
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

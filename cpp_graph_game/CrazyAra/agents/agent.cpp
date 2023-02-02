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
 * @file: agent.cpp
 * Created on 17.06.2019
 * @author: queensgambit
 */

#include <iostream>
#include <chrono>

#include "agent.h"
#include "util/blazeutil.h"
#include "util.h"

using namespace std;
using namespace crazyara;


void Agent::set_best_move(size_t moveCounter)
{
    if (moveCounter < playSettings->temperatureMoves && playSettings->initTemperature > 0.01) {
        /* DynamicVector<double> policyProbSmall = evalInfo->policyProbSmall; */
        apply_temperature(evalInfo->policyProbSmall, get_current_temperature(*playSettings, moveCounter));
        if (playSettings->quantileClipping != 0) {
            apply_quantile_clipping(playSettings->quantileClipping, evalInfo->policyProbSmall);
        }
        size_t moveIdx = random_choice(evalInfo->policyProbSmall);
        evalInfo->bestMove = evalInfo->legalMoves[moveIdx];
    }
    else {
			evalInfo->bestMove = evalInfo->pv[0][0];
    }
		print_single_pv(cout,*evalInfo,0,evalInfo->calculate_elapsed_time_ms());
}

Agent::Agent(NN_api * net, PlaySettings* playSettings, bool verbose):
    playSettings(playSettings), verbose(verbose), isRunning(false), net(net)
{
}

void Agent::set_search_settings(Node_switching_game *pos, SearchLimits *searchLimits, EvalInfo* evalInfo)
{
    this->state = pos;
    this->searchLimits = searchLimits;
    this->evalInfo = evalInfo;
}

int Agent::get_best_action()
{
    return evalInfo->bestMove;
}

void Agent::lock()
{
    runnerMutex.lock();
}

void Agent::unlock()
{
    runnerMutex.unlock();
}

void Agent::perform_action()
{
    isRunning = true;
    evalInfo->start = chrono::steady_clock::now();
    this->evaluate_board_state();
    evalInfo->end = chrono::steady_clock::now();
		assert(evalInfo->policyProbSmall.size()==state->get_num_actions());
    set_best_move(state->move_num);
		print_info(__LINE__, __FILE__,"Best move set to ", evalInfo->bestMove );
    isRunning = false;
    runnerMutex.unlock();
}

void run_agent_thread(Agent* agent)
{
    agent->perform_action();
    // inform the agent of the move, so the tree can potentially be reused later
    agent->apply_move_to_tree(agent->get_best_action(), true);
}

void apply_quantile_clipping(float quantile, DynamicVector<double>& policyProbSmall)
{
    double thresh = get_quantile(policyProbSmall, quantile);
    for (size_t idx = 0; idx < policyProbSmall.size(); ++idx) {
        if (policyProbSmall[idx] < thresh) {
            policyProbSmall[idx] = 0;
        }
    }
    policyProbSmall /= sum(policyProbSmall);
}

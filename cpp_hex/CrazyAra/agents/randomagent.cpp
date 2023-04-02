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
 * @file: mctsagentrandom.cpp
 * Created on 05.2021
 * @author: BluemlJ
 */

#include <thread>
#include <fstream>
#include <vector>
#include "randomagent.h"
#include "../evalinfo.h"
#include "../constants.h"
#include "../util/blazeutil.h"
#include "../manager/treemanager.h"
#include "../manager/threadmanager.h"
#include "../node.h"
#include "util/gcthread.h"


MCTSAgentRandom::MCTSAgentRandom(NN_api *netSingle,
                     SearchSettings* searchSettings, PlaySettings* playSettings):
    MCTSAgent(netSingle, searchSettings, playSettings)
    {

    }

MCTSAgentRandom::~MCTSAgentRandom()
{
			delete searchThread;
}

string MCTSAgentRandom::get_name() const
{
    return "MCTSRandom";
}

void MCTSAgentRandom::perform_action()
{
    vector<int> lM  = state->get_actions();
    if (lM.size() != 0){
        int randomIndex = rand() % lM.size();
        
        evalInfo->bestMove = lM[randomIndex];
    }
}

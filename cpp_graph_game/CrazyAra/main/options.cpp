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
 * @file: optionsuci.cpp
 * Created on 13.07.2019
 * @author: queensgambit
 */

#include "options.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cstring>
#include "../../hex_graph_game/util.h"

using namespace std;

// method is based on 3rdparty/Stockfish/misc.cpp
inline TimePoint current_time() {
  return std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::steady_clock::now().time_since_epoch()).count();
}

void OptionsUCI::init(OptionsMap &o)
{
		o["Hex_Size"]											 << Option(5);
    o["Allow_Early_Stopping"]          << Option(true);
#ifdef USE_RL
    o["Batch_Size"]                    << Option(8, 1, 8192);
#else
    o["Batch_Size"]                    << Option(16, 1, 8192);
#endif
    o["Centi_CPuct_Init"]              << Option(250, 1, 99999);
#ifdef USE_RL
    o["Centi_Dirichlet_Epsilon"]       << Option(25, 0, 99999);
#else
    o["Centi_Dirichlet_Epsilon"]       << Option(0, 0, 99999);
#endif
    o["Centi_Dirichlet_Alpha"]         << Option(20, 1, 99999);
    o["Centi_Epsilon_Checks"]          << Option(1, 0, 100);
    o["Centi_Epsilon_Greedy"]          << Option(5, 0, 100);
#ifdef USE_RL
    o["Centi_Node_Temperature"]        << Option(100, 1, 99999);
#else
    o["Centi_Node_Temperature"]        << Option(170, 1, 99999);
#endif
    o["Centi_Q_Value_Weight"]          << Option(100, 0, 99999);
    o["Centi_Q_Veto_Delta"]            << Option(40, 0, 99999);
#ifdef USE_RL
    o["Centi_Quantile_Clipping"]       << Option(0, 0, 100);
#else
    o["Centi_Quantile_Clipping"]       << Option(25, 0, 100);
#endif
    o["Centi_Random_Move_Factor"]      << Option(0, 0, 99);
#ifdef USE_RL
    o["Centi_Temperature"]             << Option(80, 0, 99999);
#else
    o["Centi_Temperature"]             << Option(170, 0, 99999);
#endif
    o["Centi_Temperature_Decay"]       << Option(92, 0, 100);
    o["Centi_U_Init_Divisor"]          << Option(100, 1, 99999);
    o["Centi_Virtual_Loss"]            << Option(100, 0, 99999);
    o["Context"]                       << Option("gpu", {"cpu", "gpu"});
    o["CPuct_Base"]                    << Option(19652, 1, 99999);
//    o["Enhance_Captures"]              << Option(false);         currently disabled
    o["First_Device_ID"]               << Option(0, 0, 99999);
    o["Fixed_Movetime"]                << Option(0, 0, 99999999);
    o["Last_Device_ID"]                << Option(0, 0, 99999);
    o["MCTS_Solver"]                   << Option(true);
    o["Model_Path"]               		 << Option(string("/home/kappablanca/github_repos/Gabor_Graph_Networks/GN0/alpha_zero/saved_models/traced.pt").c_str());
    o["Move_Overhead"]                 << Option(20, 0, 5000);
    o["MultiPV"]                       << Option(1, 1, 99999);
#ifdef USE_RL
    o["Nodes"]                         << Option(800, 0, 99999999);
#else
    o["Nodes"]                         << Option(0, 0, 99999999);
    o["Nodes_Limit"]                   << Option(0, 0, 999999999);
#endif
    o["Precision"]                     << Option("float32", {"float32", "int8"});
#ifdef USE_RL
    o["Reuse_Tree"]                    << Option(false);
#else
    o["Reuse_Tree"]                    << Option(true);
#endif
#ifdef USE_RL
    o["Temperature_Moves"]             << Option(15, 0, 99999);
#else
    o["Temperature_Moves"]             << Option(0, 0, 99999);
#endif
    o["Use_NPS_Time_Manager"]          << Option(true);
    o["Search_Type"]                   << Option("mcts", {"mcgs", "mcts"});
#ifdef USE_RL
    o["Simulations"]                   << Option(3200, 0, 99999999);
#else
    o["Simulations"]                   << Option(0, 0, 99999999);
#endif
    o["Threads"]                       << Option(1, 1, 512);
    o["Timeout_MS"]                    << Option(0, 0, 99999999);
    // we repeat e.g. "crazyhouse" in the list because of problem in XBoard/Winboard CrazyAra#23
    o["UCI_Variant"]                   << Option(string("HEX").c_str(), {"HEX"});
    o["Use_Raw_Network"]               << Option(false);
    // additional UCI-Options for RL only
#ifdef USE_RL
    o["Centi_Node_Random_Factor"]      << Option(10, 0, 100);
    o["Centi_Quick_Dirichlet_Epsilon"] << Option(0, 0, 99999);
    o["Centi_Quick_Probability"]       << Option(0, 0, 100);
    o["Centi_Quick_Q_Value_Weight"]    << Option(70, 0, 99999);
    o["Centi_Raw_Prob_Temperature"]    << Option(25, 0, 100);
    o["Centi_Resign_Probability"]      << Option(90, 0, 100);
    o["Centi_Resign_Threshold"]        << Option(-90, -100, 100);
    o["EPD_File_Path"]                 << Option("<empty>");
    o["MaxInitPly"]                    << Option(30, 0, 99999);
    o["MeanInitPly"]                   << Option(15, 0, 99999);
    o["Model_Directory_Contender"]     << Option(string("model_contender/" + engineName + "/" + StateConstants::DEFAULT_UCI_VARIANT()).c_str());
    o["Selfplay_Number_Chunks"]        << Option(640, 1, 99999);
    o["Selfplay_Chunk_Size"]           << Option(128, 1, 99999);
    o["Milli_Policy_Clip_Thresh"]      << Option(0, 0, 100);
    o["Quick_Nodes"]                   << Option(100, 0, 99999);
#endif
}

void OptionsUCI::setoption(istringstream &is, int& variant, Node_switching_game& state)
{

    string token, name, value;
    is >> token; // Consume "name" token

    // Read option name (can contain spaces)
    while (is >> token && token != "value")
        name += (name.empty() ? "" : " ") + token;

    // Read option value (can contain spaces)
    while (is >> token)
        value += (value.empty() ? "" : " ") + token;

    if (Options.find(name) != Options.end()) {
        const string givenName = name;
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        Options[name] = value;
        if (name != "uci_variant" && name != "uci_chess960") {
            info_string("Updated option", givenName, value);
        } else {
            bool is960 = false;
            string uciVariant = Options["UCI_Variant"];
            if (name == "uci_variant") {
                std::transform(value.begin(), value.end(), value.begin(), ::tolower);
                uciVariant = check_uci_variant_input(value, &is960);
                Options["UCI_Variant"] << Option(uciVariant.c_str());
                info_string("Updated option", givenName, uciVariant);
                if (Options["UCI_Chess960"] != is960) {
                    Options["UCI_Chess960"] << Option(is960);
                    info_string("Updated option UCI_Chess960 to", (string)Options["UCI_Chess960"]);
                }
            } else { // name == "uci_chess960"
                info_string("Updated option", givenName, value);
                is960 = Options["UCI_Chess960"];
            }
            variant = 0;

            string suffix_960 = (is960) ? "960" : "";
            info_string("variant", (string)Options["UCI_Variant"] + suffix_960, "startpos");
        }
    }
    else {
        info_string("Given option", name, "does not exist");
    }
}

string OptionsUCI::check_uci_variant_input(const string &value, bool *is960) {
    return "HEX";
}

void OptionsUCI::init_new_search(SearchLimits& searchLimits, OptionsMap &options)
{
    searchLimits.reset();
    searchLimits.startTime = current_time();
    searchLimits.moveOverhead = TimePoint(options["Move_Overhead"]);
    searchLimits.nodes = options["Nodes"];
    searchLimits.nodesLimit = options["Nodes_Limit"];
    searchLimits.movetime = options["Fixed_Movetime"];
    searchLimits.simulations = options["Simulations"];
}

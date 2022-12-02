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
 * @file: rawnetagent.cpp
 * Created on 12.06.2019
 * @author: queensgambit
 */

#include <blaze/Math.h>
#include "rawnetagent.h"
#include "util/blazeutil.h"
#include "util.h"
#include "util/speedcheck.h"

RawNetAgent::RawNetAgent(NN_api * net, PlaySettings* playSettings, bool verbose):
    Agent(net, playSettings, verbose)
{
}

void RawNetAgent::evaluate_board_state()
{
    evalInfo->legalMoves = state->get_actions();
    evalInfo->init_vectors_for_multi_pv(1UL);

    // sanity check
    assert(evalInfo->legalMoves.size() >= 1);

    // immediately stop the search if there's only one legal move
    if (evalInfo->legalMoves.size() == 1) {
        evalInfo->policyProbSmall.resize(1UL);
        evalInfo->policyProbSmall = 1;
        evalInfo->depth = 0;
        evalInfo->nodes = 0;
        evalInfo->pv[0] = {evalInfo->legalMoves[0]};
        return;
    }
		speedcheck.track_next("convert_graph");
		vector<torch::Tensor> tens = state->convert_graph(net->device);
		speedcheck.stop_track("convert_graph");
		node_features.clear();
		edge_indices.clear();
		node_features.push_back(tens[0]);
		edge_indices.push_back(tens[1]);

		std::vector<torch::jit::IValue> inputs;
		torch::Tensor batch_ptr;

		speedcheck.track_next("collate");
		inputs = collate_batch(node_features,edge_indices);
		speedcheck.stop_track("collate");

		speedcheck.track_next("nn predict");
    vector<at::Tensor> tvec = net->predict(inputs);
		speedcheck.stop_track("nn predict");
		probOutputs = tvec[0].exp(); // We expect the output from net to be log-softmax
		valueOutputs = tvec[1];
		batch_ptr = tvec[3];

    /* evalInfo->policyProbSmall.resize(evalInfo->legalMoves.size()); */
		evalInfo->policyProbSmall = torch_to_blaze<double>(probOutputs);

    size_t selIdx = argmax(evalInfo->policyProbSmall);
    int bestmove = evalInfo->legalMoves[selIdx];

    evalInfo->movesToMate[0] = 0;
    evalInfo->depth = 1;
    evalInfo->selDepth = 1;
    evalInfo->nodes = 1;
    evalInfo->pv[0] = { bestmove };
		cout << "Raw net evaluated board state " << valueOutputs[0] << endl;
}

void RawNetAgent::stop()
{
    // pass
}

void RawNetAgent::apply_move_to_tree(int move, bool ownMove)
{
    // pass
}

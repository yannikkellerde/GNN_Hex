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
 * @file: traindataexporter.cpp
 * Created on 12.09.2019
 * @author: queensgambit
 */

#include "util.h"
#include "traindataexporter.h"
#include <filesystem>
#include <inttypes.h>
#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/api/include/torch/serialize.h>

TrainDataExporter::TrainDataExporter(const string& output_folder, size_t numberChunks, size_t chunkSize):
    numberChunks(numberChunks),
    chunkSize(chunkSize),
    numberSamples(numberChunks * chunkSize),
    firstMove(true),
    gameIdx(0),
    startIdx(0),
    curSampleIdx(0),
		device(torch::kCPU),
		output_folder(output_folder)
{
		if (std::filesystem::is_directory(output_folder)) {
      info_string("Warning: Export folder already exists. Contents will be overwritten");
    }
		else{
			system(("mkdir -p "+output_folder).c_str()); // Not cross platform, but oh well
		}
		gameStartPtr.push_back(startIdx); // 0
}

void TrainDataExporter::save_sample(const Node_switching_game* pos, const EvalInfo& eval)
{
    if (startIdx+curSampleIdx >= numberSamples) {
        info_string("Extended number of maximum samples");
        return;
    }
    save_planes(pos);
    save_policy(eval.legalMoves, eval.policyProbSmall);
    save_best_move_q(eval);
    save_side_to_move(pos->onturn);
    ++curSampleIdx;
    firstMove = false;
}

void TrainDataExporter::save_best_move_q(const EvalInfo &eval)
{
    // Q value of "best" move (a.k.a selected move after mcts search)
		gameBestMoveQ.push_back(eval.bestMoveQ[0]);
}

void TrainDataExporter::save_side_to_move(Onturn col)
{
		gameValue.push_back(-(col * 2 - 1)); // Save 1 for maker and -1 for breaker initially. Multiply with -1 in the end if breaker wins.
}

void TrainDataExporter::export_game_samples() {
		assert (curSampleIdx == 0); // Call new_game first
    if (startIdx >= numberSamples) {
        info_string("Exceeded number of maximum samples");
        return;
    }
		torch::TensorOptions options_int = torch::TensorOptions().dtype(torch::kInt).device(device);
		torch::TensorOptions options_int8 = torch::TensorOptions().dtype(torch::kInt8).device(device);
		torch::TensorOptions options_float = torch::TensorOptions().dtype(torch::kFloat).device(device);
		torch::save(node_features,output_folder+"/node_features.pt");
		torch::save(edge_indices,output_folder+"/edge_indices.pt");
		torch::save(gamePolicy,output_folder+"/policy.pt");
		torch::save(torch::from_blob(gameValue.data(),gameValue.size(),options_int8),output_folder+"/value.pt");
		torch::save(torch::from_blob(gameBestMoveQ.data(),gameValue.size(),options_float),output_folder+"/best_q.pt");
		torch::save(torch::from_blob(gamePlysToEnd.data(),gamePlysToEnd.size(),options_int),output_folder+"/plys.pt");
		torch::save(torch::from_blob(gameStartPtr.data(),gameStartPtr.size(),options_int),output_folder+"/game_start_ptr.pt");
#ifdef DO_DEBUG
		torch::save(board_indices,output_folder+"/board_indices.pt");
#endif

    startIdx += curSampleIdx;
    gameIdx++;
}


size_t TrainDataExporter::get_number_samples() const
{
    return numberSamples;
}

bool TrainDataExporter::is_file_full()
{
    return startIdx >= numberSamples;
}

void TrainDataExporter::new_game(Onturn last_result)
{
		extend_plys_vector(curSampleIdx);
		apply_result_to_value(last_result,startIdx);
    firstMove = true;
		startIdx += curSampleIdx;
		gameStartPtr.push_back(startIdx);
		
    curSampleIdx = 0;
		++gameIdx;
}

void TrainDataExporter::save_planes(const Node_switching_game *pos)
{
		vector<torch::Tensor> tens = pos->convert_graph(device);
		node_features.push_back(tens[0]);
		edge_indices.push_back(tens[1]);
#ifdef DO_DEBUG
		board_indices.push_back(torch::tensor(pos->graph.lprops[board_location]));
#endif
}

void TrainDataExporter::save_policy(const vector<int>& legalMoves, const DynamicVector<float>& policyProbSmall)
{
    assert(legalMoves.size() == policyProbSmall.size());
		gamePolicy.push_back(torch::tensor(vector<float>(policyProbSmall.begin(),policyProbSmall.end())));
}

void TrainDataExporter::open_dataset_from_folder(const string& folder)
{
		torch::load(node_features,output_folder+"/node_features.pt");
		torch::load(edge_indices,output_folder+"/edge_indices.pt");
		torch::load(gamePolicy,output_folder+"/policy.pt");
		torch::Tensor vt,qt,pt,st;
		torch::load(vt,output_folder+"/value.pt");
		torch::load(qt,output_folder+"/best_q.pt");
		torch::load(pt,output_folder+"/plys.pt");
		torch::load(st,output_folder+"/game_start_ptr.pt");
		vt.contiguous();qt.contiguous();pt.contiguous();st.contiguous();
		gameValue = vector<int8_t>(vt.data_ptr<int8_t>(),vt.data_ptr<int8_t>()+vt.numel());
		gameBestMoveQ = vector<float>(qt.data_ptr<float>(),qt.data_ptr<float>()+qt.numel());
		gamePlysToEnd = vector<int>(qt.data_ptr<int>(),qt.data_ptr<int>()+qt.numel());
		gameStartPtr = vector<int>(qt.data_ptr<int>(),qt.data_ptr<int>()+qt.numel());
}


void TrainDataExporter::apply_result_to_value(Onturn result, int startIdx)
{
    // value
    if (result == breaker) {
			for (vector<int8_t>::iterator start = gameValue.begin()+startIdx;start!=gameValue.end();++start){
				(*start) = -(*start);
			}
    }
}

void TrainDataExporter::extend_plys_vector(int game_length)
{
		gamePlysToEnd.resize(gamePlysToEnd.size()+game_length);
		generate(gamePlysToEnd.end()-game_length,gamePlysToEnd.end(),[&game_length]{return game_length--;});
}

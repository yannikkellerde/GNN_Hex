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
}

void TrainDataExporter::save_sample(const Node_switching_game* pos, const EvalInfo& eval)
{
    if (startIdx+curSampleIdx >= numberSamples) {
        info_string("Extended number of maximum samples");
        return;
    }
    save_planes(pos);
    save_policy(eval.legalMoves, eval.policyProbSmall, false);
    save_best_move_q(eval);
    save_side_to_move(pos->onturn);
    save_cur_sample_index();
    ++curSampleIdx;
    // value will be set later in export_game_result()
    firstMove = false;
}

void TrainDataExporter::save_best_move_q(const EvalInfo &eval)
{
    // Q value of "best" move (a.k.a selected move after mcts search)
		gameBestMoveQ[curSampleIdx] = eval.bestMoveQ[0];
}

void TrainDataExporter::save_side_to_move(Onturn col)
{
		gameValue[curSampleIdx] = -(col * 2 - 1);
}

void TrainDataExporter::save_cur_sample_index()
{
    // curSampleIdx aka ply/half-move, starting from 0
    xt::xarray<int16_t> idxArray({ 1 }, curSampleIdx);

    if (firstMove) {
        gamePlysToEnd = idxArray;
    }
    else {
        // concatenate the sample to array for the current game
        gamePlysToEnd = xt::concatenate(xtuple(gamePlysToEnd, idxArray));
    }
}

void TrainDataExporter::export_game_samples(Onturn result) {
    if (startIdx >= numberSamples) {
        info_string("Exceeded number of maximum samples");
        return;
    }
		torch::save(node_features,output_folder+"/node_features.pt");
		torch::save(edge_indices,output_folder+"/edge_indices.pt");
		torch::save(gamePolicy,output_folder+"/policy.pt");
		torch::save(torch::from_blob(gameValue.data(),gameValue.size()),output_folder+"/value.pt");
		torch::save(torch::from_blob(gameBestMoveQ.data(),gameValue.size()),output_folder+"/best_q.pt");
		torch::save(torch::from_blob(gamePlysToEnd.data(),gamePlysToEnd.size()),output_folder+"/plys.pt");
		torch::save(torch::from_blob(gameStartPtr.data(),gameStartPtr.size()),output_folder+"/game_start_ptr.pt");

    // write value to roi
    z5::types::ShapeType offset = { startIdx };
    z5::types::ShapeType offsetPlanes = { startIdx, 0, 0, 0 };
    z5::multiarray::writeSubarray<int16_t>(dx, gameX, offsetPlanes.begin());
    z5::multiarray::writeSubarray<int16_t>(dValue, gameValue, offset.begin());
    z5::multiarray::writeSubarray<float>(dbestMoveQ, gameBestMoveQ, offset.begin());
    z5::types::ShapeType offsetPolicy = { startIdx, 0 };
    z5::multiarray::writeSubarray<float>(dPolicy, gamePolicy, offsetPolicy.begin());
    z5::multiarray::writeSubarray<int16_t>(dPlysToEnd, gamePlysToEnd, offset.begin());

    startIdx += curSampleIdx;
    gameIdx++;
    save_start_idx();
}


size_t TrainDataExporter::get_number_samples() const
{
    return numberSamples;
}

bool TrainDataExporter::is_file_full()
{
    return startIdx >= numberSamples;
}

void TrainDataExporter::new_game()
{
    firstMove = true;
    curSampleIdx = 0;
}

void TrainDataExporter::save_planes(const Node_switching_game *pos)
{
		vector<torch::Tensor> tens = pos->convert_graph(device);
    // write array to roi
    xt::xarray<int16_t>::shape_type planesShape = { 1, StateConstants::NB_CHANNELS_TOTAL(), StateConstants::BOARD_HEIGHT(), StateConstants::BOARD_WIDTH()};
    xt::xarray<int16_t> planes(planesShape);
    for (size_t idx = 0; idx < StateConstants::NB_VALUES_TOTAL(); ++idx) {
        planes.data()[idx] = int16_t(inputPlanes[idx]);
    }

    if (firstMove) {
        gameX = planes;
    }
    else {
        // concatenate the sample to array for the current game
        gameX = xt::concatenate(xtuple(gameX, planes));
    }
}

void TrainDataExporter::save_policy(const vector<Action>& legalMoves, const DynamicVector<float>& policyProbSmall, bool mirrorPolicy)
{
    assert(legalMoves.size() == policyProbSmall.size());

    xt::xarray<float>::shape_type shapePolicy = { 1, StateConstants::NB_LABELS() };
    xt::xarray<float> policy(shapePolicy, 0);

    for (size_t idx = 0; idx < legalMoves.size(); ++idx) {
        size_t policyIdx;
        if (mirrorPolicy) {
            policyIdx = StateConstants::action_to_index<classic, mirrored>(legalMoves[idx]);
        }
        else {
            policyIdx = StateConstants::action_to_index<classic, notMirrored>(legalMoves[idx]);
        }
        policy[policyIdx] = policyProbSmall[idx];
    }

    if (firstMove) {
        gamePolicy = policy;
    }
    else {
        // concatenate the sample to array for the current game
        gamePolicy = xt::concatenate(xtuple(gamePolicy, policy));
    }
}

void TrainDataExporter::save_start_idx()
{
    // gameStartIdx
    // write value to roi
    z5::types::ShapeType offsetStartIdx = { gameIdx };
    xt::xarray<int32_t> arrayGameStartIdx({ 1 }, int32_t(startIdx));
    z5::multiarray::writeSubarray<int32_t>(dStartIndex, arrayGameStartIdx, offsetStartIdx.begin());
}

void TrainDataExporter::open_dataset_from_file(const z5::filesystem::handle::File& file)
{
    dStartIndex = z5::openDataset(file, "start_indices");
    dx = z5::openDataset(file, "x");
    dValue = z5::openDataset(file, "y_value");
    dPolicy = z5::openDataset(file, "y_policy");
    dbestMoveQ = z5::openDataset(file, "y_best_move_q");
    dPlysToEnd = z5::openDataset(file, "plys_to_end");
}

void TrainDataExporter::create_new_dataset_file(const z5::filesystem::handle::File &file)
{
    // create the file in zarr format
    const bool createAsZarr = true;
    z5::createFile(file, createAsZarr);

    // create a new zarr dataset
    std::vector<size_t> shape = { numberSamples, StateConstants::NB_CHANNELS_TOTAL(), StateConstants::BOARD_HEIGHT(), StateConstants::BOARD_WIDTH() };
    std::vector<size_t> chunks = { chunkSize, StateConstants::NB_CHANNELS_TOTAL(), StateConstants::BOARD_HEIGHT(), StateConstants::BOARD_WIDTH() };
    dStartIndex = z5::createDataset(file, "start_indices", "int32", { numberSamples }, { chunkSize });
    dx = z5::createDataset(file, "x", "int16", shape, chunks);
    dValue = z5::createDataset(file, "y_value", "int16", { numberSamples }, { chunkSize });
    dPolicy = z5::createDataset(file, "y_policy", "float32", { numberSamples, StateConstants::NB_LABELS() }, { chunkSize, StateConstants::NB_LABELS() });
    dbestMoveQ = z5::createDataset(file, "y_best_move_q", "float32", { numberSamples }, { chunkSize });
    dPlysToEnd = z5::createDataset(file, "plys_to_end", "int16", { numberSamples }, { chunkSize });

    save_start_idx();
}

void TrainDataExporter::apply_result_to_value(Result result)
{
    // value
    if (result == BLACK_WIN) {
        gameValue *= -1;
    }
    else if (result == DRAWN) {
        gameValue *= 0;
    }
}

void TrainDataExporter::apply_result_to_plys_to_end()
{
    gamePlysToEnd -= curSampleIdx;
    gamePlysToEnd *= -1;
}

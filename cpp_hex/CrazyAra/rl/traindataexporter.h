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
 * @file: traindataexporter.h
 * Created on 12.09.2019
 * @author: queensgambit
 *
 * Exporter class which saves the board position in planes (x) and the target values (y) for NN training
 */

#ifndef TRAINDATAEXPORTER_H
#define TRAINDATAEXPORTER_H

#include <string>

/* #include "nlohmann/json.hpp" */

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include "constants.h"
#include "node.h"
#include "evalinfo.h"

class TrainDataExporter
{
public:
		torch::Device device;

#ifdef DO_DEBUG
		vector<torch::Tensor> board_indices;
		vector<int> moves;
#endif
		vector<torch::Tensor> node_features;
		vector<torch::Tensor> edge_indices;
		vector<torch::Tensor> gamePolicy;
		vector<int8_t> gameValue;
		vector<float> gameBestMoveQ;
		vector<int> gamePlysToEnd;
		vector<int> gameStartPtr;
    bool firstMove;

    // current number of games - 1
    size_t gameIdx;
    // current sample index to insert
    size_t startIdx;
    // current sample index of the current game
    size_t curSampleIdx;

#ifdef DO_DEBUG
	void save_best_move(const EvalInfo &eval,const Node_switching_game* pos);
#endif

    /**
     * @brief export_planes Exports the board in plane representation (x)
     * @param pos Board position to export
     */
    void save_planes(const Node_switching_game* pos);

    /**
     * @brief save_policy Saves the policy (e.g. mctsPolicy) to the matrix
     * @param legalMoves List of legal moves
     * @param policyProbSmall Probability for each move
     * @param mirrorPolicy Decides if the policy should be mirrored
     */
    void save_policy(const vector<int>& legalMoves, const DynamicVector<float>& policyProbSmall);

    /**
     * @brief save_best_move_q Saves the Q-value of the move which was selected after MCTS search(Optional training sample feature)
     * @param eval Filled EvalInfo struct after mcts search
     */
    void save_best_move_q(const EvalInfo& eval);

    /**
     * @brief save_side_to_move Saves the current side to move as a +1 for WHITE and -1 for BLACK.
     * The current side to move is either WHITE(0) or BLACK(1).
     * Later if WHITE won the game the value array is inverted.
     * For a draw it will be multiplied by 0.
     * @param col current side to move
     */
    void save_side_to_move(Onturn col);

    /**
     * @brief save_cur_sample_index Saves the current sample index, i.e. the ply index which is resetted to 0 before each game.
     */
    void save_cur_sample_index();

    /**
     * @brief open_dataset_from_file Reads a previously exported training set back into memory
     * @param file filesystem handle
     */
    void open_dataset_from_folder(const string& folder);

    /**
     * @brief open_dataset_from_file Creates a new zarr data set given a filesystem handle
     * @param file filesystem handle
     */
    void create_new_dataset_file(const string& file);

    /**
     * @brief apply_result_to_value Inverts the gameValue array if WHITE lost the game.
     * In the case of a draw, all entries are set to 0.
     * @param result Possible values DRAWN, WHITE_WIN, BLACK_WIN,
     */
    void apply_result_to_value(Onturn result, int startIdx);

    /**
     * @brief apply_result_to_plys_to_end Converts the ply index information into plys-to-end
     *  by subtracting the final ply and multiplying by -1.
     */
    void extend_plys_vector(int game_length);

		string output_folder;
    /**
     * @brief TrainDataExporter
     * @param fileNameExport File name of the uncompressed data to be exported in (e.g. "data.zarr")
     * @param numberChunks Defines how many chunks a single file should contain.
     * The product of the number of chunks and its chunk size yields the total number of samples of a file.
     * @param chunkSize Defines the chunk size of a single chunk
     */
    TrainDataExporter(const string& fileNameExport);

    /**
     * @brief export_pos Saves a given board position, policy and Q-value to the specific game arrays
     * @param pos Current board position
     * @param eval Filled EvalInfo struct after mcts search
     */
    void save_sample(const Node_switching_game* pos, const EvalInfo& eval);

		static TrainDataExporter merged_from_many(vector<unique_ptr<TrainDataExporter>> & exporters, const string & file_name_export);

		static TrainDataExporter merged_from_many(vector<vector<unique_ptr<TrainDataExporter>>> & exporters, const string & file_name_export);

    /**
     * @brief export_game_samples Assigns the game result, (Monte-Carlo value result) to every training sample.
     * The value is inversed after each step and export all training samples of a single game.
     * @param result Game match result: LOST, DRAW, WON
     */
    void export_game_samples();

    /**
     * @brief new_game Sets firstMove to true
     */
    void new_game(Onturn last_result);
};

#endif // TRAINDATAEXPORTER_H

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

#include "main/customuci.h"
#include "util.h"
#include "traindataexporter.h"
#include <filesystem>
#include <inttypes.h>
#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include "util/speedcheck.h"

TrainDataExporter::TrainDataExporter(const string& output_folder):
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
	speedcheck.track_next("save samples");
	save_planes(pos);
	save_policy(eval.legalMoves, eval.policyProbSmall);
	save_best_move_q(eval);
	save_side_to_move(pos->onturn);
#ifdef DO_DEBUG
	save_best_move(eval,pos);
#endif
	++curSampleIdx;
	firstMove = false;
	speedcheck.stop_track("save samples");
}

#ifdef DO_DEBUG
void TrainDataExporter::save_best_move(const EvalInfo &eval,const Node_switching_game* pos){
	if (eval.bestMove+2==pos->graph.num_vertices){
		moves.push_back(-1);
	}
	else{
		moves.push_back(pos->graph.lprops[BOARD_LOCATION][eval.bestMove+2]);
	}
}
#endif

void TrainDataExporter::save_best_move_q(const EvalInfo &eval)
{
	// Q value of "best" move (a.k.a selected move after mcts search)
	gameBestMoveQ.push_back(eval.bestMoveQ[0]);
}

void TrainDataExporter::save_side_to_move(Onturn col)
{
	gameValue.push_back(-(col * 2 - 1)); // Save 1 for red and -1 for blue initially. Multiply with -1 in the end if blue wins.
}

TrainDataExporter TrainDataExporter::merged_from_many(vector<unique_ptr<TrainDataExporter>> & exporters, const string& file_name_export){
	TrainDataExporter out(file_name_export);
	for (vector<unique_ptr<TrainDataExporter>>::iterator exp=exporters.begin();exp!=exporters.end();++exp){
		TrainDataExporter * exporter = exp->get();
		for_each(exporter->gameStartPtr.begin(), exporter->gameStartPtr.end(), [out](int &n){ n+=out.node_features.size(); });
		out.gameStartPtr.insert(out.gameStartPtr.end(),make_move_iterator(exporter->gameStartPtr.begin()),make_move_iterator(exporter->gameStartPtr.end()));
		out.node_features.insert(out.node_features.end(),make_move_iterator(exporter->node_features.begin()),make_move_iterator(exporter->node_features.end()));
		out.edge_indices.insert(out.edge_indices.end(),make_move_iterator(exporter->edge_indices.begin()),make_move_iterator(exporter->edge_indices.end()));
		out.gamePolicy.insert(out.gamePolicy.end(),make_move_iterator(exporter->gamePolicy.begin()),make_move_iterator(exporter->gamePolicy.end()));
		out.gameValue.insert(out.gameValue.end(),make_move_iterator(exporter->gameValue.begin()),make_move_iterator(exporter->gameValue.end()));
		out.gameBestMoveQ.insert(out.gameBestMoveQ.end(),make_move_iterator(exporter->gameBestMoveQ.begin()),make_move_iterator(exporter->gameBestMoveQ.end()));
		out.gamePlysToEnd.insert(out.gamePlysToEnd.end(),make_move_iterator(exporter->gamePlysToEnd.begin()),make_move_iterator(exporter->gamePlysToEnd.end()));

#ifdef DO_DEBUG
		out.board_indices.insert(out.board_indices.end(),make_move_iterator(exporter->board_indices.begin()),make_move_iterator(exporter->board_indices.end()));
		out.moves.insert(out.moves.end(),make_move_iterator(exporter->moves.begin()),make_move_iterator(exporter->moves.end()));
#endif
	}
	return out;
}

TrainDataExporter TrainDataExporter::merged_from_many(vector<vector<unique_ptr<TrainDataExporter>>> & exporters, const string& file_name_export){
	TrainDataExporter out(file_name_export);
	for (vector<vector<unique_ptr<TrainDataExporter>>>::iterator portvec=exporters.begin();portvec!=exporters.end();++portvec ){
		for (vector<unique_ptr<TrainDataExporter>>::iterator exp=portvec->begin();exp!=portvec->end();++exp){
			TrainDataExporter * exporter = exp->get();
			for_each(exporter->gameStartPtr.begin(), exporter->gameStartPtr.end(), [out](int &n){ n+=out.node_features.size(); });
			out.gameStartPtr.insert(out.gameStartPtr.end(),make_move_iterator(exporter->gameStartPtr.begin()),make_move_iterator(exporter->gameStartPtr.end()));
			out.node_features.insert(out.node_features.end(),make_move_iterator(exporter->node_features.begin()),make_move_iterator(exporter->node_features.end()));
			out.edge_indices.insert(out.edge_indices.end(),make_move_iterator(exporter->edge_indices.begin()),make_move_iterator(exporter->edge_indices.end()));
			out.gamePolicy.insert(out.gamePolicy.end(),make_move_iterator(exporter->gamePolicy.begin()),make_move_iterator(exporter->gamePolicy.end()));
			out.gameValue.insert(out.gameValue.end(),make_move_iterator(exporter->gameValue.begin()),make_move_iterator(exporter->gameValue.end()));
			out.gameBestMoveQ.insert(out.gameBestMoveQ.end(),make_move_iterator(exporter->gameBestMoveQ.begin()),make_move_iterator(exporter->gameBestMoveQ.end()));
			out.gamePlysToEnd.insert(out.gamePlysToEnd.end(),make_move_iterator(exporter->gamePlysToEnd.begin()),make_move_iterator(exporter->gamePlysToEnd.end()));

#ifdef DO_DEBUG
			out.board_indices.insert(out.board_indices.end(),make_move_iterator(exporter->board_indices.begin()),make_move_iterator(exporter->board_indices.end()));
			out.moves.insert(out.moves.end(),make_move_iterator(exporter->moves.begin()),make_move_iterator(exporter->moves.end()));
#endif
		}
	}
	return out;
}

void TrainDataExporter::export_game_samples() {
	assert (curSampleIdx == 0); // Call new_game first
	speedcheck.track_next("file_write");
	torch::TensorOptions options_int = torch::TensorOptions().dtype(torch::kInt).device(device);
	torch::TensorOptions options_int8 = torch::TensorOptions().dtype(torch::kInt8).device(device);
	torch::TensorOptions options_float = torch::TensorOptions().dtype(torch::kFloat).device(device);
	torch::save(node_features,output_folder+"/node_features.pt");
	cout << "Node Features exported ..." << endl;
	torch::save(edge_indices,output_folder+"/edge_indices.pt");
	cout << "Edge indices exported ..." << endl;
	torch::save(gamePolicy,output_folder+"/policy.pt");
	torch::save(torch::from_blob(gameValue.data(),gameValue.size(),options_int8),output_folder+"/value.pt");
	cout << "Value and Policy exported ..." << endl;
	torch::save(torch::from_blob(gameBestMoveQ.data(),gameValue.size(),options_float),output_folder+"/best_q.pt");
	torch::save(torch::from_blob(gamePlysToEnd.data(),gamePlysToEnd.size(),options_int),output_folder+"/plys.pt");
	torch::save(torch::from_blob(gameStartPtr.data(),gameStartPtr.size(),options_int),output_folder+"/game_start_ptr.pt");
#ifdef DO_DEBUG
	torch::save(board_indices,output_folder+"/board_indices.pt");
	torch::save(torch::from_blob(moves.data(),moves.size(),options_int),output_folder+"/moves.pt");
#endif

	speedcheck.stop_track("file_write");
	startIdx += curSampleIdx;
	gameIdx++;
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
	vector<torch::Tensor> tens;
	speedcheck.track_next("convert_graph");
	if (Options["CNN_Mode"]){
		tens = pos->convert_planes(device);
	}
	else{
		tens = pos->convert_graph(device);
		edge_indices.push_back(tens[1]);
	}
	node_features.push_back(tens[0]);
	speedcheck.stop_track("convert_graph");
#ifdef DO_DEBUG
	board_indices.push_back(torch::tensor(pos->graph.lprops[BOARD_LOCATION]));
#endif
}

void TrainDataExporter::save_policy(const vector<int>& legalMoves, const DynamicVector<float>& policyProbSmall)
{
	assert(legalMoves.size() == policyProbSmall.size());
	torch::Tensor policy = torch::empty(policyProbSmall.size());
	for (int i=0;i<policyProbSmall.size();++i){   // This is super slow??
		policy[legalMoves[i]] = policyProbSmall[i];
	}
	gamePolicy.push_back(policy);
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
	if (result == BLUE) {
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

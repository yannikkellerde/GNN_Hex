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
 * @file: mctsagent.cpp
 * Created on 13.05.2019
 * @author: queensgambit
 */

#include <thread>
#include <fstream>
#include "main/customuci.h"
#include "mctsagent.h"
#include "evalinfo.h"
#include "constants.h"
#include "util/blazeutil.h"
#include "manager/treemanager.h"
#include "manager/threadmanager.h"
#include "node.h"
#include "util.h"
#include "util/speedcheck.h"


MCTSAgent::MCTSAgent(NN_api *netSingle,
		SearchSettings* searchSettings, PlaySettings* playSettings):
	Agent(netSingle, playSettings, true),
	searchSettings(searchSettings),
	rootNode(nullptr),
	rootState(nullptr),
	ownNextRoot(nullptr),
	opponentsNextRoot(nullptr),
	lastValueEval(-1.0f),
	reusedFullTree(false),
	overallNPS(0.0f),
	nbNPSentries(0),
	threadManager(nullptr),
	reachedTablebases(false)
{
	mapWithMutex.hashTable.reserve(1e6);

	searchThread = new SearchThread(net, searchSettings, &mapWithMutex);
	timeManager = make_unique<TimeManager>(searchSettings->randomMoveFactor);
	generator = default_random_engine(r());
}

MCTSAgent::~MCTSAgent()
{
	delete searchThread;
}

Node* MCTSAgent::get_opponents_next_root() const
{
	return opponentsNextRoot.get();
}

Node* MCTSAgent::get_root_node() const
{
	return rootNode.get();
}

float MCTSAgent::get_dirichlet_noise() const
{
	return searchSettings->dirichletEpsilon;
}

float MCTSAgent::get_q_value_weight() const
{
	return searchSettings->qValueWeight;
}

void MCTSAgent::update_q_value_weight(float value)
{
	searchSettings->qValueWeight = value;
}

void MCTSAgent::update_dirichlet_epsilon(float value)
{
	searchSettings->dirichletEpsilon = value;
}

Node_switching_game *MCTSAgent::get_root_state() const
{
	return rootState.get();
}

bool MCTSAgent::is_running() const
{
	return isRunning;
}

size_t MCTSAgent::init_root_node(Node_switching_game *state)
{
	size_t nodesPreSearch;
	gcThread.oldRootNode = rootNode;
	rootNode = get_root_node_from_tree(state);

	/* if (rootNode != nullptr) { */
	/* 	// swap the states because now the old states are used */
	/* 	// This way the memory won't be freed for the next new move */
	/* 	nodesPreSearch = size_t(rootNode->get_visits()); */
	/* 	if (rootNode->is_playout_node()) { */
	/* 		nodesPreSearch -= rootNode->get_free_visits(); */
	/* 	} */
	/* 	info_string(nodesPreSearch, "nodes of former tree will be reused"); */
	/* } */
	/* else { */
		/* print_info(__LINE__,__FILE__,"Num Verts for new root:",state->graph.num_vertices); */
	create_new_root_node(state);
	nodesPreSearch = 0;
	/* } */
	return nodesPreSearch;
}

shared_ptr<Node> MCTSAgent::get_root_node_from_tree(Node_switching_game *state)
{
	reusedFullTree = false;

	if (rootNode == nullptr) {
		return nullptr;
	}
	if (!searchSettings->reuseTree) {
		delete_old_tree();
		return nullptr;
	}

	if (same_hash_key(rootNode.get(), state)) {
		reusedFullTree = true;
		return rootNode;
	}

	if (same_hash_key(ownNextRoot.get(), state) && ownNextRoot->is_playout_node() && ownNextRoot->get_number_of_nodes() > 0) {
		return ownNextRoot;
	}
	if (same_hash_key(opponentsNextRoot.get(), state) && opponentsNextRoot->is_playout_node() && opponentsNextRoot->get_number_of_nodes() > 0) {
		return opponentsNextRoot;
	}
	// the node wasn't found, clear the old tree
	delete_old_tree();

	return nullptr;
}

void MCTSAgent::create_new_root_node(Node_switching_game* state)
{
#ifdef MCTS_STORE_STATES
	rootNode = make_shared<Node>(state->clone(), searchSettings);
#else
	rootNode = make_shared<Node>(state, searchSettings);
#endif
	vector<torch::Tensor> tens;
	speedcheck.track_next("convert_graph");
	if (Options["CNN_Mode"]){
		tens = state->convert_planes(net->device);
	}
	else{
		tens = state->convert_graph(net->device);
		net->edge_indices.push_back(tens[1]);
	}
	net->node_features.push_back(tens[0]);
	speedcheck.stop_track("convert_graph");
	/* node_features.clear(); */
	/* edge_indices.clear(); */
	root_node_to_fill = true;
	root_node_batch_idx = net->edge_indices.size(); 

	/* std::vector<torch::jit::IValue> inputs; */
	/* torch::Tensor batch_ptr; */

	/* speedcheck.track_next("collate"); */
	/* inputs = collate_batch(node_features,edge_indices); */
	/* speedcheck.stop_track("collate"); */
	/* vector<at::Tensor> tvec = net->predict(inputs); */
	/* probOutputs = tvec[0].exp(); // We expect the output from net to be log-softmax */
	/* valueOutputs = tvec[1]; */
	/* batch_ptr = tvec[3]; */
	/* size_t tbHits = 0; */
	/* fill_nn_results(0, false, valueOutputs, probOutputs, batch_ptr, rootNode.get(), tbHits, */
	/* 		searchSettings, rootNode->is_tablebase()); */
	/* rootNode->prepare_node_for_visits(); */
}

void MCTSAgent::fill_root_nn_results(){
	assert(root_node_to_fill);
	root_node_to_fill = false;
	size_t tbHits = 0;
	fill_nn_results(root_node_batch_idx,false,net->valueOutputs,net->probOutputs,net->batch_ptr,rootNode.get(),tbHits,searchSettings,rootNode->is_tablebase());
	rootNode->prepare_node_for_visits();
	if (rootNode->get_number_child_nodes() == 1) {
		/* info_string("Only single move available -> early stopping"); */
		handle_single_move();
	}
	else if (rootNode->get_number_child_nodes() == 0) {
		info_string("The given position has no legal moves");
	}
	else{
		if (searchSettings->dirichletEpsilon > 0.009f) {
			/* info_string("apply dirichlet noise"); */
			// TODO: Check for dirichlet compability
			rootNode->apply_dirichlet_noise_to_prior_policy(searchSettings);
			rootNode->fully_expand_node();
		}

		if (!rootNode->is_root_node()) {
			rootNode->make_to_root();
		}
		searchThread->set_root_node(rootNode.get());
		searchThread->set_root_state(rootState.get());
		searchThread->set_search_limits(searchLimits);
		searchThread->set_is_running(true);
		searchThread->reset_stats();
	}
}

void MCTSAgent::delete_old_tree()
{
	// clear all remaining node of the former root node
	mapWithMutex.hashTable.clear();
	assert(mapWithMutex.hashTable.size() == 0);
}

void MCTSAgent::sleep_and_log_for(size_t timeMS, size_t updateIntervalMS)
{
	if (!isRunning) {
		return;
	}
	for (size_t var = 0; var < timeMS / updateIntervalMS && isRunning; ++var) {
		this_thread::sleep_for(chrono::milliseconds(updateIntervalMS));
		evalInfo->end = chrono::steady_clock::now();
		if (!searchThread->is_running()) {
			isRunning = false;
			return;
		}
	}
	this_thread::sleep_for(chrono::milliseconds(timeMS % 1000));
}

void MCTSAgent::update_nps_measurement(float curNPS)
{
	if (searchSettings->useNPSTimemanager) {
		++nbNPSentries;
		overallNPS += 1.0f/nbNPSentries * (curNPS - overallNPS);
	}
}

void MCTSAgent::apply_move_to_tree(int move, bool ownMove)
{
	if (!reusedFullTree && rootNode != nullptr && rootNode->is_playout_node()) {
		if (ownMove) {
			opponentsNextRoot = pick_next_node(move, rootNode.get());
			return;
		}
		else if (opponentsNextRoot != nullptr && opponentsNextRoot->is_playout_node()){
			ownNextRoot = pick_next_node(move, opponentsNextRoot.get());
			return;
		}
	}
	// the full tree will be deleted next search
	opponentsNextRoot = nullptr;
	ownNextRoot = nullptr;
}

void MCTSAgent::clear_game_history()
{
	delete_old_tree();
	ownNextRoot = nullptr;
	opponentsNextRoot = nullptr;
	rootNode = nullptr;
	lastValueEval = -1.0f;
	nbNPSentries = 0;
	overallNPS = 0;
	reachedTablebases = false;
}

bool MCTSAgent::is_policy_map()
{
	return false;
}

string MCTSAgent::get_name() const
{
	return net->model_name;
}

void MCTSAgent::update_stats()
{
	avgDepth = 0;
	maxDepth = searchThread->get_max_depth();
	tbHits = searchThread->get_tb_hits();
}

void MCTSAgent::handle_single_move()
{
	float targetEval = lastValueEval;
#ifndef MCTS_SINGLE_PLAYER
	if (lastSideToMove != state->onturn) {
		targetEval = -lastValueEval;
	}
#endif
	rootNode->set_value(targetEval);
	rootNode->set_q_value(0, targetEval);
}

void MCTSAgent::create_unexpanded_root_nodes(){
	rootState = unique_ptr<Node_switching_game>(state->clone());
	evalInfo->nodesPreSearch = init_root_node(state);
}

void MCTSAgent::eval_step_start(){
	searchThread->create_mini_batch();
}

bool MCTSAgent::do_more_eval(){
	return searchThread->is_running() && searchThread->nodes_limits_ok() && searchThread->is_root_node_unsolved();
}

void MCTSAgent::eval_step_stop(){
	searchThread->after_nn_results_are_in();
}

void MCTSAgent::eval_stop(){
	update_stats();
	update_eval_info(*evalInfo, rootNode.get(), tbHits, maxDepth, searchSettings);
	lastValueEval = evalInfo->bestMoveQ[0];
	lastSideToMove = state->onturn;
	update_nps_measurement(evalInfo->calculate_nps());
	set_best_move(state->move_num);
	searchThread->set_is_running(false);
};

void MCTSAgent::evaluate_board_state()
{
	create_unexpanded_root_nodes();
	net->predict_stored();
	fill_root_nn_results();
	while (do_more_eval()){
		eval_step_start();
		net->predict_stored();
		eval_step_stop();
	}
	eval_stop();
	update_nps_measurement(evalInfo->calculate_nps());
}

void MCTSAgent::run_mcts_search()
{
	searchThread->set_root_node(rootNode.get());
	searchThread->set_root_state(rootState.get());
	searchThread->set_search_limits(searchLimits);
	searchThread->set_reached_tablebases(reachedTablebases);
	run_search_thread(searchThread);
	/* int curMovetime = timeManager->get_time_for_move(searchLimits, rootState->onturn, rootNode->plies_from_null()/2); */
	/* ThreadManagerData tData(rootNode.get(), searchThread, evalInfo, lastValueEval); */
	/* ThreadManagerInfo tInfo(searchSettings, searchLimits, overallNPS, rootState->onturn); */
	/* ThreadManagerParams tParams(curMovetime, 250, is_game_sceneario(searchLimits), can_prolong_search(rootNode->plies_from_null()/2, timeManager->get_thresh_move())); */
	/* threadManager = make_unique<ThreadManager>(&tData, &tInfo, &tParams); */
	/* unique_ptr<thread> tManager = make_unique<thread>(run_thread_manager, threadManager.get()); */
	/* runnerMutex.unlock(); */
	/* print_info(__LINE__,__FILE__,rootNode->get_node_count()); */
}


void MCTSAgent::print_root_node()
{
	if (rootNode == nullptr) {
		info_string("You must do a search before you can print the root node statistics");
		return;
	}
	const vector<size_t> customOrdering = sort_permutation(evalInfo->policyProbSmall, std::greater<float>());
	rootNode->print_node_statistics(rootState.get(), customOrdering);
}

void print_child_nodes_to_file(const Node* parentNode, Node_switching_game* state, size_t parentId, size_t& nodeId, ostream& outFile, size_t depth, size_t maxDepth)
{
	int initialId = nodeId;
	if (maxDepth != 0 && depth > maxDepth) {
		return;
	}
	size_t childIdx = 0;
	for (auto it = parentNode->get_node_it_begin(); it != parentNode->get_node_it_end(); ++it) {
		const Node* node = it->get();
		if (node != nullptr) {
			int action = parentNode->get_action(childIdx);
			outFile << "N" << ++nodeId << " [label = \""
				<<  state->get_actions()
				<< "\"]" << endl;
			int perc = (float(parentNode->get_child_number_visits()[childIdx++]) / parentNode->get_visits()) * 100 + 0.5;
			perc = min(perc+10, 100);
			outFile << "N" << parentId << " -> " << "N" << nodeId
				<< " [color = gray" << 100-perc << "]"
				<<   ";" << endl;
		}
	}
	outFile  << "{ rank=same; ";
	for (size_t idx = initialId+1; idx < initialId+parentNode->get_no_visit_idx(); ++idx) {
		outFile << "N" << idx << " ";
	}
	outFile << "}" << endl;
	for (auto it = parentNode->get_node_it_begin(); it != parentNode->get_node_it_end(); ++it) {
		const Node* node = it->get();
		if (node != nullptr && node->is_playout_node()) {
			unique_ptr<Node_switching_game> state2 = unique_ptr<Node_switching_game>(state->clone());
			int action = parentNode->get_action(childIdx);
			state2->make_move(action,false,NOPLAYER,true);
			print_child_nodes_to_file(node, state2.get(), ++initialId, nodeId, outFile, depth+1, maxDepth);
		}
	}
}

void MCTSAgent::export_search_tree(size_t maxDepth, const string& filename)
{
	size_t nodeId = 0;
	ofstream outFile;
	outFile.open (filename);
	outFile << "digraph g {" << endl;
	outFile << "graph [layout = dot]" << endl << endl;
	outFile << "node [shape = circle," << endl
		<< " fontname = Helvetica," << endl
		<< " fontsize = 8.5," << endl
		<< " fixedsize = true," << endl
		<< " color = black," << endl
		<< " width = 0.3," << endl
		<< " height = 0.3," << endl
		<< " label = \"\"]" << endl << endl;

	outFile << "edge [" << endl
		<< "arrowhead = vee," << endl
		<< "arrowsize = 0.2," << endl
		<< "color = grey" << endl
		<< "]" << endl << endl;

	outFile << "N0 [label = \"root\"]" << endl << endl;
	print_child_nodes_to_file(rootNode.get(), rootState.get(), 0, nodeId, outFile, 1, maxDepth);
	outFile << "}" << endl;
	outFile.close();
}

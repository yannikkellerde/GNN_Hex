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
 * @file: crazyara.cpp
 * Created on 12.06.2019
 * @author: queensgambit
 */

#include <torchscatter/scatter.h>
#include <torchsparse/sparse.h>
#include "options.h"
#include "crazyara.h"
#include "playmode.h"
#include "convert_moves_to_training_data.h"

#include <thread>
#include <fstream>
#include "agents/mctsagent.h"
#include "search.h"
#include "timeoutreadythread.h"
#include "evalinfo.h"
#include "constants.h"
#include "util.h"
#include "util/speedcheck.h"
#include "util/statlogger.h"


CrazyAra::CrazyAra():
	rawAgent(nullptr),
	mctsAgent(nullptr),
	netSingle(nullptr),         // will be initialized in is_ready()
#ifdef USE_RL
	netSingleContender(nullptr),
	mctsAgentContender(nullptr),
#endif
	searchSettings(SearchSettings()),
	searchLimits(SearchLimits()),
	playSettings(PlaySettings()),
	variant(0),
	useRawNetwork(false),      // will be initialized in init_search_settings()
	networkLoaded(false),
	ongoingSearch(false),
	is960(false),
	changedUCIoption(false)
{
}

CrazyAra::~CrazyAra()
{
}

void CrazyAra::welcome()
{
	cout << intro << endl;
}

void CrazyAra::uci_loop(int argc, char *argv[])
{
	init();
	unique_ptr<Node_switching_game> state = make_unique<Node_switching_game>(Options["Hex_Size"]);
	string token, cmd;
	EvalInfo evalInfo;
	state->reset();

	for (int i = 1; i < argc; ++i)
		cmd += string(argv[i]) + " ";

	size_t it = 0;

	// this is debug vector which can contain uci commands which will be automatically processed when the executable is launched
	vector<string> commands = {
	};

	do {
		if (it < commands.size()) {
			cmd = commands[it];
			cout << ">>" << cmd << endl;
		}
		else if (argc == 1 && !getline(cin, cmd)) // Block here waiting for input or EOF
			cmd = "quit";

		istringstream is(cmd);

		token.clear(); // Avoid a stale if getline() returns empty or blank line
		is >> skipws >> token;

		if (token == "uci") {
			cout << engine_info()
				<< "uciok" << endl;
		}
		else if (token == "swapmap"){
			gen_swap_map(Options["Hex_Size"],netSingle.get());
			cout << "readyok" << endl;
		}

		else if (token == "starting_eval"){
			gen_starting_eval_file(Options["Hex_Size"],netSingle.get());
			cout << "readyok" << endl;
		}
		else if (token == "speedsummary") speedcheck.summarize(cout);
		else if (token == "statsummary") statlogger.summarize(cout);
		else if (token == "setoption")  set_uci_option(is, *state.get());
		else if (token == "go")         go(state.get(), is, evalInfo);
		else if (token == "ucinewgame") ucinewgame();
		else if (token == "isready")    is_ready<true>();
		else if (token == "play"){
			prepare_search_config_structs();
			playmode(mctsAgent.get(),rawAgent.get(),&searchLimits,&evalInfo);
		}

		// Additional custom non-UCI commands, mainly for debugging
		else if (token == "root")       mctsAgent->print_root_node();
		else if (token == "tree")      export_search_tree(is);
		else if (token == "flip")       state->switch_onturn();
		else if (token == "d") {
			state->graphviz_me("crazyarastate.dot");
			system("neato -Tpdf crazyarastate.dot -o crazyarastate.pdf");
		}
		else if (token == "activeuci") activeuci();
		else if (token == "make_training_data")   make_training_data(is);
#ifdef USE_RL
		else if (token == "selfplay")   selfplay(is);
		else if (token == "arena")      arena(is);
		// Test if the new modes are also usable for chess and others

		else if (token == "tournament")   roundrobin(is);
#endif   
		else
			cout << "Unknown command: " << cmd << endl;

		++it;
	} while (token != "quit" && argc == 1); // Command line args are one-shot

	wait_to_finish_last_search();
}

void CrazyAra::prepare_search_config_structs()
{
    OptionsUCI::init_new_search(searchLimits, Options);

    if (changedUCIoption) {
        init_search_settings();
        init_play_settings();
        changedUCIoption = false;
    }
}

void CrazyAra::go(Node_switching_game* state, istringstream &is,  EvalInfo& evalInfo)
{
	wait_to_finish_last_search();
	ongoingSearch = true;
	prepare_search_config_structs();

	string token;
	while (is >> token) {
		if (token == "searchmoves")
			while (is >> token);
		else if (token == "wtime")     is >> searchLimits.time[RED];
		else if (token == "btime")     is >> searchLimits.time[BLUE];
		else if (token == "winc")      is >> searchLimits.inc[RED];
		else if (token == "binc")      is >> searchLimits.inc[BLUE];
		else if (token == "movestogo") is >> searchLimits.movestogo;
		else if (token == "depth")     is >> searchLimits.depth;
		else if (token == "nodes")     is >> searchLimits.nodes;
		else if (token == "movetime")  is >> searchLimits.movetime;
		else if (token == "infinite")  searchLimits.infinite = true;
	}

	if (useRawNetwork) {
		cout << "Using raw network" << endl;
		rawAgent->set_search_settings(state, &searchLimits, &evalInfo);
		rawAgent->lock();  // lock() rawAgent to avoid calling stop() immediatly
		mainSearchThread = thread(run_agent_thread, rawAgent.get());
	}
	else {
		cout << "Using MCTS" << endl;
		mctsAgent->set_search_settings(state, &searchLimits, &evalInfo);
		mctsAgent->lock(); // lock() mctsAgent to avoid calling stop() immediatly
		mainSearchThread = thread(run_agent_thread, mctsAgent.get());
	}
}

void CrazyAra::wait_to_finish_last_search()
{
	if (ongoingSearch) {
		mainSearchThread.join();
		ongoingSearch = false;
	}
}

void CrazyAra::export_search_tree(istringstream &is)
{
	string depth, filename;
	is >> depth;
	is >> filename;
	if (depth == "") {
		mctsAgent->export_search_tree(2, "tree.gv");
		return;
	}
	if (filename == "") {
		mctsAgent->export_search_tree(std::stoi(depth), "tree.gv");
		return;
	}
	mctsAgent->export_search_tree(std::stoi(depth), filename);
}

void CrazyAra::activeuci()
{
	for (const auto& it : Options)
		cout << "option name " << it.first << " value " << string(Options[it.first]) << endl;
	cout << "readyok" << endl;
}

void CrazyAra::make_training_data(istringstream &is){
	string filename, output_folder;
	int hex_size;
	is >> filename;
	is >> hex_size;
	is >> output_folder;
	to_training_data(filename,hex_size,output_folder);
}

#ifdef USE_RL
void CrazyAra::selfplay(istringstream &is)
{
	prepare_search_config_structs();
	speedcheck.track_next("selfplay");
	SelfPlay selfPlay(rawAgent.get(), mctsAgent.get(), &searchLimits, &playSettings, &searchSettings, &rlSettings, Options);
	size_t numberOfGames;
	is >> numberOfGames;
	selfPlay.go(Options["Threads"],std::min((int)numberOfGames,(int)Options["Num_Parallel_Games"]),numberOfGames,netBatches);
	cout << "readyok" << endl;
	speedcheck.stop_track("selfplay");
}

void CrazyAra::arena(istringstream &is)
{
	assert (((string)Options["Model_Path_Contender"]).length()>0);
	int numberOfGames;
	int threads;
	int pgames = Options["Num_Parallel_Games"];
	is >> numberOfGames;
	is >> threads;
	prepare_search_config_structs();
	SelfPlay selfPlay(rawAgent.get(), mctsAgent.get(), &searchLimits, &playSettings, &searchSettings, &rlSettings, Options);
	netBatchesContender = create_new_net_batches(Options["Model_Path_Contender"], threads);
	if (netBatches.size()<threads){
		netBatches= create_new_net_batches(Options["Model_Path"], threads);
	}
	selfPlay.go_arena(netBatches,netBatchesContender,std::min(numberOfGames,pgames),numberOfGames,threads);

	cout << "Arena summary" << endl;
	cout << "Contender winrate: " << statlogger.mean_statistics["contender_wins"].first << endl;
	if (statlogger.mean_statistics["contender_wins"].first > 0.5f) {
		cout << "replace" << endl;
	}
	else {
		cout << "keep" << endl;
	}
}

void CrazyAra::roundrobin(istringstream &is)
{
	int type;
	int numberofgames;
	is >> numberofgames;
	struct modelstring
	{
		int number_of_mcts_agent;
		int number_of_model_folder;
	};

	int i = 0;
	std::vector<modelstring> agents;
	std::vector<int> numbers;
	while (!is.eof())
	{
		is >> type;
		int tmp1 = type;
		is >> type;

		modelstring tmp;
		tmp.number_of_mcts_agent = tmp1;
		tmp.number_of_model_folder = type;
		std::cout << "ini " << tmp1 << " " << type << std::endl;
		agents.push_back(tmp);
		numbers.push_back(i);
		i++;
	}

	std::vector<std::string> combinations = comb(numbers, 2);
	std::string delimiter = " ";
	for (int i = 0; i < combinations.size(); i++)
	{
		std::string s = combinations[i];

		int token1 = std::stoi(s.substr(0, s.find(delimiter)));
		int token2 = std::stoi(s.substr(2, s.find(delimiter)));
		std::string comb = std::to_string(agents[token1].number_of_mcts_agent) + " " + std::to_string(agents[token2].number_of_mcts_agent);
		std::string m1 = "m" + std::to_string(agents[token1].number_of_model_folder) + "/";
		std::string m2 = "m" + std::to_string(agents[token2].number_of_model_folder) + "/";
		std::istringstream iss(comb + " " + std::to_string(numberofgames));
	}

	exit(0);
}

void CrazyAra::init_rl_settings()
{
	rlSettings.numberChunks = Options["Selfplay_Number_Chunks"];
	rlSettings.chunkSize = Options["Selfplay_Chunk_Size"];
	rlSettings.quickSearchNodes = Options["Quick_Nodes"];
	rlSettings.quickSearchProbability = Options["Centi_Quick_Probability"] / 100.0f;
	rlSettings.quickSearchQValueWeight = Options["Centi_Quick_Q_Value_Weight"] / 100.0f;
	rlSettings.lowPolicyClipThreshold = Options["Milli_Policy_Clip_Thresh"] / 1000.0f;
	rlSettings.quickDirichletEpsilon = Options["Centi_Quick_Dirichlet_Epsilon"] / 100.0f;
	rlSettings.nodeRandomFactor = Options["Centi_Node_Random_Factor"] / 100.0f;
	rlSettings.rawPolicyProbabilityTemperature = Options["Centi_Raw_Prob_Temperature"] / 100.0f;
	rlSettings.resignProbability = Options["Centi_Resign_Probability"] / 100.0f;
	rlSettings.resignThreshold = Options["Centi_Resign_Threshold"] / 100.0f;
	rlSettings.reuseTreeForSelpay = Options["Reuse_Tree"];
	rlSettings.epdFilePath = string(Options["EPD_File_Path"]);
	if (rlSettings.epdFilePath != "<empty>" and rlSettings.epdFilePath != "") {
		std::ifstream epdFile (rlSettings.epdFilePath);
		if (!epdFile.is_open()) {
			throw invalid_argument("Given epd file: " + rlSettings.epdFilePath + " could not be opened.");
		}
	}
}
#endif

std::string read_string_from_file(const std::string &file_path){
	const std::ifstream input_stream(file_path, std::ios_base::binary);

	if (input_stream.fail()) {
		throw std::runtime_error("Failed to open file");
	}

	std::stringstream buffer;
	buffer << input_stream.rdbuf();

	return buffer.str();
}

void CrazyAra::init()
{
	OptionsUCI::init(Options);
}

	template<bool verbose>
bool CrazyAra::is_ready()
{
	bool hasReplied = false;
	if (!networkLoaded) {
		const size_t timeoutMS = Options["Timeout_MS"];
		TimeOutReadyThread timeoutThread(timeoutMS);
		thread tTimeoutThread;
		if (timeoutMS != 0) {
			tTimeoutThread = thread(run_timeout_thread, &timeoutThread);
		}
		init_search_settings();
		init_play_settings();
#ifdef USE_RL
		init_rl_settings();
#endif
		print_info(__LINE__,__FILE__,"loading model",string(Options["Model_Path"]));
		netSingle = create_new_net_single(string(Options["Model_Path"]),int(Options["First_Device_ID"]));
		netBatches = create_new_net_batches(string(Options["Model_Path"]));
		mctsAgent = create_new_mcts_agent(netSingle.get(), &searchSettings);
		rawAgent = make_unique<RawNetAgent>(netSingle.get(), &playSettings, false);
		/* StateConstants::init(mctsAgent->is_policy_map()); */
		timeoutThread.kill();
		if (timeoutMS != 0) {
			tTimeoutThread.join();
		}
		hasReplied = timeoutThread.has_replied();
		networkLoaded = true;
	}
	wait_to_finish_last_search();
	if (verbose && !hasReplied) {
		cout << "readyok" << endl;
	}
	return networkLoaded;
}

void CrazyAra::ucinewgame()
{
	if (networkLoaded) {
		wait_to_finish_last_search();
		mctsAgent->clear_game_history();
		cout << "info string newgame" << endl;
	}
}

string CrazyAra::engine_info()
{
	stringstream ss;
	ss << "id name " << "Hex graph ara" << " " << "0.0.1" << " (" << __DATE__ << ")" << "\n";
	return ss.str();
}

unique_ptr<NN_api> CrazyAra::create_new_net_single(const string& modelPath, int device_id)
{
	torch::Device device(torch::kCPU,0);
	if (Options["Context"] == "cpu"){
		device = torch::Device(torch::kCPU,0);
	}
	else if (Options["Context"] == "gpu"){
		device = torch::Device(torch::kCUDA,device_id);
	}
	else{
		throw std::logic_error("Invalid Context");
	}
	return make_unique<NN_api>(modelPath,device);
}

vector<unique_ptr<NN_api>> CrazyAra::create_new_net_batches(const string& modelPath, int threads)
{
	if (threads==0) threads = Options["Threads"];
	vector<unique_ptr<NN_api>> someNetBatches;
	for (int deviceId = int(Options["First_Device_ID"]); deviceId <= int(Options["Last_Device_ID"]); ++deviceId) {
		for (size_t i = 0; i < size_t(threads); ++i) {
			someNetBatches.push_back(create_new_net_single(modelPath,deviceId));
		}
	}
	return someNetBatches;
}

void CrazyAra::set_uci_option(istringstream &is, Node_switching_game& state)
{
	// these three UCI-Options may trigger a network reload, keep an eye on them
	const string prevModelDir = Options["Model_Path"];
	const int prevThreads = Options["Threads"];
	const string prevUciVariant = Options["UCI_Variant"];
	const int prevFirstDeviceID = Options["First_Device_ID"];
	const int prevLastDeviceID = Options["Last_Device_ID"];

	OptionsUCI::setoption(is, variant, state);
	changedUCIoption = true;
	if (networkLoaded) {
		if (string(Options["Model_Path"]) != prevModelDir || int(Options["Threads"]) != prevThreads || string(Options["UCI_Variant"]) != prevUciVariant ||
				int(Options["First_Device_ID"]) != prevFirstDeviceID || int(Options["Last_Device_ID"] != prevLastDeviceID)) {
			networkLoaded = false;
			is_ready<false>();
		}
	}
}

unique_ptr<MCTSAgent> CrazyAra::create_new_mcts_agent(NN_api* netSingle, SearchSettings* searchSettings, MCTSAgentType type)
{   
	switch (type) {
		case MCTSAgentType::kDefault:
			return make_unique<MCTSAgent>(netSingle, searchSettings, &playSettings);
		case MCTSAgentType::kRandom:
			info_string("TYP 7 -> Random");
			return make_unique<MCTSAgentRandom>(netSingle, searchSettings, &playSettings);
		default:
			info_string("Unknown MCTSAgentType");
			return nullptr;
	}
}

size_t get_num_gpus(OptionsMap& option)
{
	return size_t(option["Last_Device_ID"] - option["First_Device_ID"] + 1);
}

void CrazyAra::init_search_settings()
{
	searchSettings.multiPV = Options["MultiPV"];
	searchSettings.threads = Options["Threads"] * get_num_gpus(Options);
	searchSettings.batchSize = Options["Batch_Size"];
	searchSettings.useMCGS = Options["Search_Type"] == "mcgs";
	searchSettings.qValueWeight = Options["Centi_Q_Value_Weight"] / 100.0f;
	searchSettings.qVetoDelta = Options["Centi_Q_Veto_Delta"] / 100.0f;
	searchSettings.epsilonChecksCounter = round((1.0f / Options["Centi_Epsilon_Checks"]) * 100.0f);
	searchSettings.epsilonGreedyCounter = round((1.0f / Options["Centi_Epsilon_Greedy"]) * 100.0f);
	searchSettings.cpuctInit = Options["Centi_CPuct_Init"] / 100.0f;
	searchSettings.cpuctBase = Options["CPuct_Base"];
	searchSettings.dirichletEpsilon = Options["Centi_Dirichlet_Epsilon"] / 100.0f;
	searchSettings.dirichletAlpha = Options["Centi_Dirichlet_Alpha"] / 100.0f;
	searchSettings.nodePolicyTemperature = Options["Centi_Node_Temperature"] / 100.0f;
	searchSettings.virtualLoss = Options["Centi_Virtual_Loss"] / 100.0f;
	searchSettings.randomMoveFactor = Options["Centi_Random_Move_Factor"]  / 100.0f;
	searchSettings.allowEarlyStopping = Options["Allow_Early_Stopping"];
	useRawNetwork = Options["Use_Raw_Network"];
	searchSettings.useNPSTimemanager = Options["Use_NPS_Time_Manager"];
	searchSettings.useTablebase = false;
	searchSettings.reuseTree = Options["Reuse_Tree"];
	searchSettings.mctsSolver = Options["MCTS_Solver"];
}

void CrazyAra::init_play_settings()
{
	playSettings.initTemperature = Options["Centi_Temperature"] / 100.0f;
	playSettings.temperatureMoves = Options["Temperature_Moves"];
	playSettings.temperatureDecayFactor = Options["Centi_Temperature_Decay"] / 100.0f;
	playSettings.quantileClipping = Options["Centi_Quantile_Clipping"] / 100.0f;
#ifdef USE_RL
	playSettings.meanInitPly = Options["MeanInitPly"];
	playSettings.maxInitPly = Options["MaxInitPly"];
#endif
}

std::vector<std::string> comb(std::vector<int> N, int K)
{
	std::string bitmask(K, 1); // K leading 1's
	bitmask.resize(N.size(), 0); // N-K trailing 0's
	std::vector<std::string> p ;
	// print integers and permute bitmask

	do {
		std::string c = "";
		for (int i = 0; i < N.size(); ++i) // [0..N-1] integers
		{
			if (bitmask[i]){
				c.append(std::to_string(N[i])+ " ");
			} 
		}
		p.push_back(c);
	} while (std::prev_permutation(bitmask.begin(), bitmask.end()));

	return p;
}

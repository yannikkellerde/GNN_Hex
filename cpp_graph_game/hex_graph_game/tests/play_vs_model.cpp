#include "play_vs_model.h"
#include<iostream>
#include <unistd.h>
#include "../util.h"
#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include "../shannon_node_switching_game.h"
#include "../nn_api.h"
#include <sstream>
#include <string>

using namespace std;
using namespace torch::indexing;

void play_vs_model(string model_path){
	string action;
	int move;
	stringstream ss;
	std::vector<torch::Tensor> inputs, outputs;
	torch::Device device(torch::kCUDA);
	NN_api nn_api(model_path,device);
	/* for (auto m = nn_api.model.parameters().begin();m!=nn_api.model.parameters().end();++m){ */
	/* 	torch::NoGradGuard no_grad; */
	/* 	(*m).index_put_({None},torch::rand((*m).sizes())*5); */
	/* 	cout << *m << endl; */
	/* } */
	/* exit(0); */
	starting_eval_img(5,&nn_api);
	Node_switching_game game(5);
	while (true){
		inputs = game.convert_graph(device);
		torch::TensorOptions options_long = torch::TensorOptions().dtype(torch::kLong).device(device);
		inputs.push_back(torch::zeros(game.graph.num_vertices,options_long));
		outputs = nn_api.predict(inputs);
		print_info(__LINE__,__FILE__,"Value:",outputs[1].item<double>());
		print_info(__LINE__,__FILE__,"Policy:",outputs[0]);
		vector<string> nodetext(game.graph.num_vertices);
		for (int i=0;i<game.graph.num_vertices;++i){
			ss.str(string());
			ss << i << "(" << setprecision(3) << outputs[0][i].item<double>() << ")";
			nodetext[i] = ss.str();
		}
		game.graphviz_me(nodetext,"my_graph.dot",game.graph);
    system("pkill -f 'mupdf my_graph.pdf'");
		system("neato -Tpdf my_graph.dot -o my_graph.pdf");
		system("mupdf my_graph.pdf &");
		usleep(100000U);
		system("bspc node -f west");
		cin >> action;
		if (action=="e"){
			move = outputs[0].argmax().item<int>();
		}
		else{
			move = stoi(action);
		}
		game.make_move(move,false,NOPLAYER,true);
		Onturn winner = game.who_won();
		if (winner==RED){
			cout << "red won" << endl;
		}
		if (winner==BLUE){
			cout << "blue won" << endl;
		}
	}

}

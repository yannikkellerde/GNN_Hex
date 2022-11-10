#include "test_my_graph.h"

void play_around_with_graph(){
	string user_command;
	vector<string> command_parts;
	Graph g(10);
	while (true){
		g.graphviz_me("my_graph.dot",false);
		g.do_complete_dump("graph_dump.txt");
    system("pkill -f 'mupdf my_graph.pdf'");
		system("neato -Tpdf my_graph.dot -o my_graph.pdf");
		system("mupdf my_graph.pdf &");
		usleep(100000U);
		system("bspc node -f west");
		getline(cin,user_command);
		boost::split(command_parts,user_command,boost::is_any_of(" "));
		if (command_parts[0] == "add_edge"){
			cout << g.add_edge(stoi(command_parts[1]),stoi(command_parts[2])) << endl;
		}
		else if (command_parts[0] == "clear_vertex"){
			g.clear_vertex(stoi(command_parts[1]));
		}
		else if (command_parts[0] == "delete_edge_onesided"){
			cout << g.delete_edge_onesided(stoi(command_parts[1]),stoi(command_parts[2])) << endl;
		}
		else if (command_parts[0] == "remove_vertex"){
			g.remove_vertex(stoi(command_parts[1]));
		}
	}
}

#include "../graph.cpp"
#include "../shannon_node_switching_game.cpp"

void test_copy(){
	Graph g(10);
	g.add_lprop(0);
	Graph f = g;
	g.add_edge(1,2);
	cout << g.sources.size() << " " << f.sources.size() << endl;
	g.lprops[0][0] = 1;
	cout << g.lprops[0][0] << " " << f.lprops[0][0] << endl;
	
	Node_switching_game game(11);
	game.make_move(5);
	Node_switching_game fame = game;
	cout << fame.graph.num_vertices << " " << game.graph.num_vertices << endl;
	fame.make_move(6);
	cout << fame.graph.num_vertices << " " << game.graph.num_vertices << endl;
	cout << fame.onturn << " " << game.onturn << endl;
}

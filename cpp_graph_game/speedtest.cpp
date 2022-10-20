#include "shannon_node_switching_game.cpp"
#include<iostream>
#include <unistd.h>

int main(){
	int move;
	const int size=11;
	Hex_board<size> board;
	Node_switching_game<size> game(board);


	
	return 0;
}

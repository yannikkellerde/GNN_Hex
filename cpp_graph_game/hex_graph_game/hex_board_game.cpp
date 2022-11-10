#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include "hex_board_game.h"

using namespace std;

Hex_board::Hex_board(int size, Hex_color c):onturn(c),size(size),num_squares(size*size){
	position = vector<Hex_color>(num_squares,empty_square);
};
vector<int> Hex_board::get_actions(){
	vector <int> actions = {};
	for (int i=0; i<position.size(); ++i){
		if (position[i]==empty_square){
			actions.push_back(i);
		}
	}
	return actions;
};


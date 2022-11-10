#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include <vector>

#if !defined(HEX_BOARD_H)
#define HEX_BOARD_H

using namespace std;

enum Hex_color{empty_square,red,blue};

class Hex_board {
	private:
		Hex_color onturn;
		vector<Hex_color> position;
	public:
		int size;
		int num_squares;
		Hex_board(int size=11, Hex_color c=red);
		vector<int> get_actions();
};
#endif

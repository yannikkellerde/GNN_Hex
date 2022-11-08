#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include <vector>

#if !defined(HEX_BOARD)
#define HEX_BOARD

using namespace std;

enum Hex_color{empty_square,red,blue};

class Hex_board {
	private:
		Hex_color onturn;
		vector<Hex_color> position;
	public:
		int size;
		int num_squares;
		Hex_board(int size=11, Hex_color c=red):onturn(c),size(size),num_squares(size*size){
			position = vector<Hex_color>(num_squares,empty_square);
		};
		vector<int> get_actions(){
			vector <int> actions = {};
			for (int i=0; i<position.size(); ++i){
				if (position[i]==empty_square){
					actions.push_back(i);
				}
			}
			return actions;
		};

};
#endif

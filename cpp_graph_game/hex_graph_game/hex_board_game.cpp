#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include <vector>

#if !defined(HEX_BOARD)
#define HEX_BOARD

using namespace std;

enum Hex_color{empty_square,red,blue};

template <int S>
class Hex_board {
	private:
		Hex_color position[S*S];
		Hex_color onturn;
	public:
		static const int size = S;
		static const int num_squares = S*S;
		Hex_board(Hex_color c=red){
			onturn = c;
			fill(begin(position),begin(position)+num_squares,empty_square);
		};
		vector<int> get_actions(){
			vector <int> actions = {};
			for (int i=0; i<sizeof(position)/sizeof(*position); ++i){
				if (position[i]==empty_square){
					actions.push_back(i);
				}
			}
			return actions;
		};

};
#endif

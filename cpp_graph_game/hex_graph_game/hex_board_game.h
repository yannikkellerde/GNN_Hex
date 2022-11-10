#include <vector>

#if !defined(HEX_BOARD_H)
#define HEX_BOARD_H

enum Hex_color{empty_square,red,blue};

class Hex_board {
	private:
		Hex_color onturn;
		std::vector<Hex_color> position;
	public:
		int size;
		int num_squares;
		Hex_board(int size=11, Hex_color c=red);
		std::vector<int> get_actions();
};
#endif

#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include <vector>

using namespace std;

struct String_template{
	vector<string> p1_stones;
	vector<string> p2_stones;
	vector<string> to_capture;
};

template<class BigInt>
struct Captured_template{
	BigInt p1_stones;
	BigInt p2_stones;
	BigInt to_capture;
	BigInt all_relevant_locations;
	int width;
	int height;
	vector<int> captured_board_locations;
};

template<class BigInt, int S>
vector<vector<Captured_template<BigInt>>> precompute_templates(){
	// Not finsihed, because rotations/mirroring result in very many templates
	// might still be faster than local search in graph though. Not sure.
	vector<Captured_template<BigInt>> basic_templates;
	vector<String_template> str_templates = {
		{
			{"100",
			 "000",
			 "111"},
			{"000",
			 "000",
			 "000"},
			{"000",
			 "110",
			 "000"}
		},
		{
			{"100",
			 "000",
			 "110"
			},
			{"000",
			 "001",
			 "000"
			},
			{"000",
			 "110",
			 "000"
			},
		},
		{
			{"0100",
			 "0000",
			 "0010"
			},
			{"0000",
			 "1001",
			 "0000"
			},
			{"0000",
			 "0110",
			 "0000"
			},
		},
		{
			{"1001",
			 "0111"
			},
			{"0000",
			 "0000"
			},
			{"0110",
			 "0000"
			},
		},
		{
			{"0000",
			 "0001",
			 "0111"
			},
			{"1000",
			 "0000",
			 "0000"
			},
			{"0000",
			 "0110",
			 "0000"
			},
		},
		{
			{"0000",
			 "0001",
			 "0011"
			},
			{"1000",
			 "1000",
			 "0000"
			},
			{"0000",
			 "0110",
			 "0000"
			},
		},
		{
			{"110",
			 "101"
			},
			{"000",
			 "000"
			},
			{"000",
			 "010"
			},
		},
		{
			{"110",
			 "001",
			 "000"
			},
			{"000",
			 "000",
			 "010"
			},
			{"000",
			 "010",
			 "000"
			},
		},
		{
			{"010",
			 "001",
			 "000"
			},
			{"000",
			 "100",
			 "010"
			},
			{"000",
			 "010",
			 "000"
			},
		},
	};

	for (const String_template str_temp:str_templates){

	}
}


enum Hex_color{empty_square,red,blue};

template <int S>
class Hex_board {
	private:
		Hex_color position[S*S];
		Hex_color onturn;
	public:
		int size = S;
		int num_squares = S*S;
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

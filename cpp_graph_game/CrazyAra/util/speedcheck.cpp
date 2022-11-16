#include "util/speedcheck.h"
#include <string>
#include <vector>
#include <chrono>
#include <cassert>
#include <iomanip>

using namespace std;

Speedcheck speedcheck; // Single object design

Speedcheck::Speedcheck(){
	/* vector<string> track_things = {"files write","make moves","save samples","nn predict","convert_graph","collate"}; */
	/* for (string t:track_things){ */
	/* 	movetimes[t] = 0; */
	/* 	is_running[t] = false; */
	/* } */
}

void Speedcheck::track_next(const string& what){
	assert (!is_running[what]);
	starts[what] = chrono::steady_clock::now();
	is_running[what] = true;
}

void Speedcheck::stop_track(const string& what){
	assert (is_running[what]);
	movetimes[what] += chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now()-starts[what]).count();
	is_running[what] = false;
}

void Speedcheck::summarize(ostream& write_here){
	write_here << "|      task      |  total time (μs) |" << endl
			 <<       "| -------------- | ---------------- |"<< endl
			 << std::setprecision(5);
	for (map<string,long>::iterator it=movetimes.begin();it!=movetimes.end();++it){
		write_here << "|" << std::setw(16) << it->first << "|"
			<< std::setw(18) << it->second << "|" << endl;
	}
}

#include "util/speedcheck.h"
#include <string>
#include <vector>
#include <chrono>
#include <cassert>
#include <iomanip>
#include <thread>

using namespace std;

Speedcheck speedcheck; // Single object design

Speedcheck::Speedcheck(){
}

void Speedcheck::track_next(const string& what){
	thread::id tid = this_thread::get_id();
	assert (!is_running[tid][what]);
	starts[tid][what] = chrono::steady_clock::now();
	is_running[tid][what] = true;
}

void Speedcheck::stop_track(const string& what){
	thread::id tid = this_thread::get_id();
	assert (is_running[tid][what]);
	threadlock[what].lock();
	movetimes[what] += chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now()-starts[tid][what]).count();
	threadlock[what].unlock();
	is_running[tid][what] = false;
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

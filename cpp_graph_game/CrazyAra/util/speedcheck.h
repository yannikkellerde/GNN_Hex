#include <map>
#include <string>
#include <chrono>
#include <iostream>
#include <thread>
#include <mutex>

#if !defined(SPEEDCHECK_H)
#define SPEEDCHECK_H

class Speedcheck{
	public:
		std::mutex threadlock;
		std::map<std::string,long> movetimes;
		std::map<std::thread::id,std::map<std::string,std::chrono::steady_clock::time_point>> starts;
		std::map<std::thread::id,std::map<std::string,bool>> is_running;
		
		Speedcheck();
		void track_next(const std::string& what);
		void stop_track(const std::string& what);
		void stop_track_mean(const std::string& what);
		void summarize(std::ostream& write_here);
};

extern Speedcheck speedcheck;

#endif

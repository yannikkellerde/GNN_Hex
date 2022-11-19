#include <map>
#include <string>
#include <chrono>
#include <iostream>
#include <thread>
#include <mutex>

class Speedcheck{
	public:
		std::map<std::string,std::mutex> threadlock;
		std::map<std::string,long> movetimes;
		std::map<std::thread::id,std::map<std::string,std::chrono::steady_clock::time_point>> starts;
		std::map<std::thread::id,std::map<std::string,bool>> is_running;
		
		Speedcheck();
		void track_next(const std::string& what);
		void stop_track(const std::string& what);
		void summarize(std::ostream& write_here);
};

extern Speedcheck speedcheck;

#include <map>
#include <string>
#include <iostream>
#include <thread>
#include <mutex>

#if !defined(STATLOGGER_H)
#define STATLOGGER_H

class Statlogger{
	public:
		std::map<std::string,std::mutex> threadlock;
		std::map<std::string,std::pair<double,int>> mean_statistics;
		std::map<std::string,double> sum_statistics;
		std::map<std::string,double> max_statistics;
		std::map<std::string,double> min_statistics;
		
		Statlogger();
		void reset_key(const std::string & what);

		void log_mean_statistic(const std::string& what, double number);
		void log_sum_statistic(const std::string& what, double number);
		void log_max_statistic(const std::string& what, double number);
		void log_min_statistic(const std::string& what, double number);
		void summarize(std::ostream& write_here);
		void print_statistics(std::ostream& write_here);
};

extern Statlogger statlogger;

#endif

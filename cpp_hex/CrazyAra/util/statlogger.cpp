#include "util/statlogger.h"
#include <algorithm>
#include <string>
#include <vector>
#include <chrono>
#include <cassert>
#include <iomanip>
#include <thread>

using namespace std;
Statlogger statlogger;

Statlogger::Statlogger(){
}

void Statlogger::reset_key(const string & what){
	mean_statistics.erase(what);
	max_statistics.erase(what);
	min_statistics.erase(what);
	sum_statistics.erase(what);
}

void Statlogger::log_mean_statistic(const string & what, double number){
	threadlock[what].lock();
	if (mean_statistics[what].second == 0){
		mean_statistics[what].first = number;
		mean_statistics[what].second = 1;
	}
	else{
		mean_statistics[what].first = (mean_statistics[what].first*mean_statistics[what].second+number)/(mean_statistics[what].second+1);
		++mean_statistics[what].second;
	}
	threadlock[what].unlock();
}

void Statlogger::log_sum_statistic(const string & what, double number){
	threadlock[what].lock();
	sum_statistics[what]+=number;
	threadlock[what].unlock();
}

void Statlogger::log_max_statistic(const string & what, double number){
	threadlock[what].lock();
	if (max_statistics[what]<number){
		max_statistics[what] = number;
	}
	threadlock[what].unlock();
}

void Statlogger::log_min_statistic(const string & what, double number){
	threadlock[what].lock();
	if (min_statistics[what]>number){
		min_statistics[what] = number;
	}
	threadlock[what].unlock();
}

void Statlogger::print_statistics(ostream& write_here){
	write_here << std::setprecision(5);
	for (map<string,double>::iterator it=sum_statistics.begin();it!=sum_statistics.end();++it){
		string modi = it->first;
		replace(modi.begin(),modi.end(),' ','_');
		write_here << "Statistic: " << modi << " " << it->second<<endl;
	}
	for (map<string,double>::iterator it=min_statistics.begin();it!=min_statistics.end();++it){
		string modi = it->first;
		replace(modi.begin(),modi.end(),' ','_');
		write_here << "Statistic: " << modi << " " << it->second<<endl;
	}
	for (map<string,double>::iterator it=max_statistics.begin();it!=max_statistics.end();++it){
		string modi = it->first;
		replace(modi.begin(),modi.end(),' ','_');
		write_here << "Statistic: " << modi << " " << it->second<<endl;
	}
	for (map<string,pair<double,int>>::iterator it=mean_statistics.begin();it!=mean_statistics.end();++it){
		string modi = it->first;
		replace(modi.begin(),modi.end(),' ','_');
		write_here << "Statistic: " << modi << " " << it->second.first<<endl;
	}
}

void Statlogger::summarize(ostream& write_here){
	write_here << "|        statistic         |       value      |" << endl
			 <<       "| ------------------------ | ---------------- |"<< endl
			 << std::setprecision(5);
	for (map<string,double>::iterator it=sum_statistics.begin();it!=sum_statistics.end();++it){
		write_here << "|" << std::setw(26) << it->first << "|"
			<< std::setw(18) << it->second << "|" << endl;
	}
	for (map<string,double>::iterator it=min_statistics.begin();it!=min_statistics.end();++it){
		write_here << "|" << std::setw(26) << it->first << "|"
			<< std::setw(18) << it->second << "|" << endl;
	}
	for (map<string,double>::iterator it=max_statistics.begin();it!=max_statistics.end();++it){
		write_here << "|" << std::setw(26) << it->first << "|"
			<< std::setw(18) << it->second << "|" << endl;
	}
	for (map<string,pair<double,int>>::iterator it=mean_statistics.begin();it!=mean_statistics.end();++it){
		write_here << "|" << std::setw(26) << it->first << "|"
			<< std::setw(18) << it->second.first << "|" << endl;
	}
}

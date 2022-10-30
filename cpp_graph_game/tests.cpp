#include <iostream>
#include <fstream>
#include <iterator>
#include <utility>
#include <algorithm>
#include <vector>
using namespace std;

int main() {
	vector<int> a;
	vector<int>::iterator it,wit;
	a.insert(a.begin(),8);
	a.insert(a.end(),9);
	a.push_back(1);
	a.push_back(2);
	a.push_back(3);
	for (int i:a)
			cout << i << ' ';
	cout << endl;
	it = a.begin()+1;
	cout << *it << endl;
	it = a.insert(it,4);
	for (int i:a)
			cout << i << ' ';
	cout << endl;
	cout << *it << endl;
	it = a.insert(it,5);
	for (int i:a)
			cout << i << ' ';
	cout << endl;
	it++;
	it = a.insert(it,6);
	for (int i:a)
			cout << i << ' ';
	cout << endl;
	cout << *it << endl;
	a.erase(it,it+2);
	for (int i:a)
			cout << i << ' ';
	cout << endl;
	*(a.end()-1)--;
	for (int i:a)
			cout << i << ' ';
	cout << endl;
	return 0;
} 

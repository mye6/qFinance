#include "Solver.h"
#include "Finance.h"
#include "Puzzle.h"
#include "Leetcode.h"


/*
int count_lines(const string& filename) {
	int res = 0;
	ifstream in(filename);
	string line;
	while (getline(in, line)) ++res;	
	in.close();
	return res;
}

int count_lines(const vector<string>& files) {
	int total = 0;
	for (size_t i = 0; i < files.size(); ++i) {
		total += count_lines(files[i]);
	}
	return total;
}
*/


int main() {
	PRINT(count_lines());
	system("pause");
	return 0;
}
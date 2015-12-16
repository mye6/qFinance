#include "Solver.h"
#include "Finance.h"
#include "Puzzle.h"
#include "Leetcode.h"






int main() {
	vector<string> vec{ "deer", "door", "cake", "card" };
	ValidWordAbbr vwa(vec);
	PRINT(vwa.isUnique("deer"));
	PRINT(vwa.isUnique("de"));


	system("pause");
	return 0;
}
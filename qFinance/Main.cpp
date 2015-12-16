#include "Solver.h"
#include "Finance.h"
#include "Puzzle.h"
#include "Leetcode.h"


class ValidWordAbbr {
public:
	ValidWordAbbr(vector<string> &dictionary) {
		for (string& d : dictionary) {
			int n = d.length();
			string abbr = d[0] + to_string(n) + d[n - 1];
			mp[abbr].insert(d);
		}
	}

	bool isUnique(string word) {
		int n = word.length();
		string abbr = word[0] + to_string(n) + word[n - 1];
		PRINT(mp[abbr].count(word));
		PRINT(mp[abbr].size());
		return mp[abbr].count(word) == mp[abbr].size();
	}
private:
	unordered_map<string, unordered_set<string>> mp;
};



int main() {
	vector<string> vec{ "deer", "door", "cake", "card" };
	ValidWordAbbr vwa(vec);
	PRINT(vwa.isUnique("deer"));
	PRINT(vwa.isUnique("de"));


	system("pause");
	return 0;
}
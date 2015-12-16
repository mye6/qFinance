#include "Solver.h"
#include "Finance.h"
#include "Puzzle.h"
#include "Leetcode.h"

bool wordPattern(string pattern, string str) {
	map<char, int> p2i; // map char to int
	map<string, int> w2i; // map string to int
	istringstream in(str); // parse the word strings
	int i = 0, n = pattern.size();
	for (string word; in >> word; ++i) {
		if (p2i[pattern[i]] != w2i[word] || i == n)
			return false; // if str is longer, or no match, return with false, before recording
		p2i[pattern[i]] = w2i[word] = i + 1; // record each char/string mapping
	}
	for (map<char, int>::iterator it = p2i.begin(); it != p2i.end(); ++it) {
		cout << *it << endl;
	}
	//cout << p2i << endl;
	SEP;
	cout << p2i << endl;
	SEP;
	cout << w2i << endl;
	return i == n;
}


int main() {
	string pattern("abba"), str("dog cat cat dog");
	PRINT(wordPattern(pattern, str));


	system("pause");
	return 0;
}
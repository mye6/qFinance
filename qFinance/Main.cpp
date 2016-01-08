#include "Solver.h"
#include "Leetcode.h"
/*
#include "Finance.h"
#include "Puzzle.h"

#include "ThinkCPP.h"
#include "FaqCPP.h"
*/

/*Section: Hash Table*/

// 170. Two Sum III - Data Structure Design
class TwoSum {
	unordered_map<int, int> map;
public:
	void add(int number) { map[number]++; }
	bool find(int value) {
		for (auto it : map) {
			int i = it.first, j = value - i;
			if ((i == j && it.second>1) 
				|| (i != j && map.find(j) != map.end()) )
				return true;
		}
		return false;
	}
};

// 299. Bulls and Cows
string getHint(string secret, string guess) {
	if (secret.empty() || secret.size() != guess.size()) return "0A0B";
	int na = 0, nb = 0;
	vector<int> svec(10, 0), gvec(10, 0);
	for (int i = 0; i < (int)secret.size(); ++i) {
		char c1 = secret[i], c2 = guess[i];
		if (c1 == c2) { ++na; }
		else { ++svec[c1 - '0']; ++gvec[c2 - '0']; }
	}
	for (int i = 0; i < (int)svec.size(); ++i) nb += min(svec[i], gvec[i]);
	return to_string(na) + "A" + to_string(nb) + "B";
}


// 202. Happy Number
int digitSquareSum(int n) {
	int res = 0;
	while (n > 0) {
		int t = n % 10;
		res += t*t;
		n /= 10;
	}
	return res;
}
bool isHappy(int n) {
	int i1 = n, i2 = digitSquareSum(n);
	while (i2 != i1) {
		i1 = digitSquareSum(i1);
		i2 = digitSquareSum(digitSquareSum(i2));
	}
	return i1 == 1;
}

// 242. Valid Anagram
bool isAnagram(string s, string t) {
	sort(s.begin(), s.end());
	sort(t.begin(), t.end());
	return s == t;
}

// 242. Valid Anagram
bool isAnagram2(string s, string t) {
	if (s.size() != t.size()) return false;
	int n = s.length();
	vector<int> counts(256, 0);
	for (int i = 0; i < n; ++i) {
		++counts[s[i]];
		--counts[t[i]];
	}
	for (int count : counts)
		if (count != 0) return false;
	return true;
}

// 49. Group Anagram
vector<vector<string>> groupAnagrams(vector<string>& strs) {
	unordered_map<string, vector<string> > smap;
	for (string s : strs) {
		string t = s;
		sort(s.begin(), s.end());
		smap[s].push_back(t);
	}
	vector<vector<string> > res;
	for (auto it : smap) {
		sort(it.second.begin(), it.second.end());
		res.push_back(it.second);
	}
	return res;
}

// 205. Isomorphic Strings
bool isIsomorphic(string s, string t) {
	if (s.size() != t.size()) return false;
	vector<int> smap(256, -1), tmap(256, -1);
	int n = s.size();
	for (int i = 0; i < n; ++i) {
		if (smap[s[i]] != tmap[t[i]]) return false;
		smap[s[i]] = tmap[t[i]] = i;
	}
	return true;
}

// 3. Longest Substring Without Repeating Characters
int lengthOfLongestSubstring(string s) {
	vector<int> smap(256, -1);
	int res = 0, pos = -1;
	for (int i = 0; i < (int)s.size(); ++i) {
		pos = max(pos, smap[s[i]]);
		smap[s[i]] = i;
		res = max(res, i - pos);
	}
	return res;
}

// 36. Valid Sudoku
bool isValidSudoku(vector<vector<char> >& board) {
	bool used1[]
}


int main(){
	string s("egg"), t("add");
	PRINT(isIsomorphic(s, t));
	

	cout << endl;
	system("pause");
	return 0;
}  
#include "Solver.h"
#include "Leetcode.h"
/*
#include "Finance.h"
#include "Puzzle.h"

#include "ThinkCPP.h"
#include "FaqCPP.h"
*/

/*Section: Math*/
// 231. Power of Two, bit
bool isPowerOfTwo0(int n) {
	if (n <= 0) return false;
	return !(n&(n - 1));
}

// 231. Power of Two, module
bool isPowerOfTwo2(int n) {
	return (n > 0) && (1073741824 % n == 0);
	// 2^30 = 1073741824
}

// 231. Power of Two, log
bool isPowerOfTwo3(int n) {
	if (n <= 0) return false;
	double x = log10(n) / log10(2);
	return x == floor(x); // only x=2^n will give true
}

// 231. Power of Two, iterative
bool isPowerOfTwo4(int n) {
	if (n > 1) {
		while (n % 2 == 0) n /= 2;
	}
	return n == 1;
}

// 231. Power of Two, recursive
bool isPowerOfTwo5(int n) {
	return n>0 && (n == 1 || (n % 2 == 0 && isPowerOfTwo5(n / 2)));
}

// 326. Power of Three, module
bool isPowerOfThree1(int n) {
	return (n > 0) && (1162261467 % n == 0);
	// 3^19 = 1162261467
}

// 326. Power of Three, log
bool isPowerOfThree2(int n) {
	if (n <= 0) return false;
	double x = log10(n) / log10(3);
	return x == floor(x); // only x=3^n will give true
}

// 326. Power of Three, iterative O(h), n=3^h
bool isPowerOfThree3(int n) {
	if (n > 1) {
		while (n % 3 == 0) n /= 3;
	}	
	return n == 1;
}

// 326. Power of Three, recursive
bool isPowerOfThree4(int n) {
	return n>0 && (n == 1 || (n % 3 == 0 && isPowerOfThree4(n/3)) );
}

// 67. Add Binary
string addBinary0(string a, string b) {
	string s = "";
	int c = 0, i = a.size() - 1, j = b.size() - 1;
	while (i >= 0 || j >= 0 || c == 1) {
		c += (i >= 0 ? a[i--] - '0' : 0);
		c += (j >= 0 ? b[j--] - '0' : 0);		
		s = char(c % 2 + '0') + s;		
		c /= 2;
	}
	return s;
}

// 8. String to Integer (atoi)
int atoi0(const char *str) {
	int sign = 1, base = 0, i = 0;
	while (str[i]==' ') ++i;
	if (str[i]=='-' || str[i]=='+') sign = 1 - 2*(str[i++]=='-');
	while (str[i] >= '0' && str[i] <= '9') {
		if (base > INT_MAX/10 || (base == INT_MAX/10 && str[i]-'0'>7)) {
			if (sign == 1) return INT_MAX;
			else return INT_MIN;
		}
		base = 10 * base + (str[i++] - '0');
	}
	return base * sign;
}

// 223. Rectangle Area
int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
	int left = max(A, E), right = max(min(C, G), left);
	int bottom = max(B, F), top = max(min(D, H), bottom);
	return (C - A)*(D - B) - (right - left)*(top - bottom) + (G - E)*(H - F);
}

// 263. Ugly Number
bool isUgly(int num) {
	for (int i = 2; i < 6 && num>0; ++i) {
		while (num%i == 0) num /= i;
	}
	return num == 1;
}

// 264. Ugly Number II
int nthUglyNumber(int n) {
	if (n <= 0) return 0;
	if (n == 1) return 1;
	int t2 = 0, t3 = 0, t5 = 0;
	vector<int> d(n);
	d[0] = 1;
	for (int i = 1; i < n; ++i) {
		d[i] = min(d[t2] * 2, min(d[t3] * 3, d[t5] * 5) );
		if (d[i] == d[t2] * 2) ++t2;
		if (d[i] == d[t3] * 3) ++t3;
		if (d[i] == d[t5] * 5) ++t5;
	}
	return d[n - 1];
}

// 313. Super Ugly Number
int nthSuperUglyNumber(int n, vector<int>& primes) {
	int k = primes.size();
	vector<int> index(k, 0), d(n, INT_MAX);
	d[0] = 1;
	for (int i = 1; i < n; ++i){
		for (int j = 0; j < k; ++j)
			d[i] = min(d[i], d[index[j]] * primes[j]);
		for (int j = 0; j < k; ++j)
			index[j] += (d[i] == d[index[j]] * primes[j]);
	}
	return d[n - 1];
}

// 2. Add Two Numbers
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
	ListNode dummy(-1), *tail = &dummy;
	int carry = 0;
	while (l1 || l2 || carry > 0) {
		int tmp = ((l1 ? l1->val : 0) + (l2 ? l2->val : 0) + carry);
		tail->next = new ListNode(tmp % 10);
		tail = tail->next;
		carry = tmp / 10;
		l1 ? l1 = l1->next : NULL;
		l2 ? l2 = l2->next : NULL;
	}
	return dummy.next;
}

// 43. Multiply Strings
string multiply(string num1, string num2) {
	int n1 = num1.size(), n2 = num2.size();
	string res(n1 + n2, '0');
	for (int i = n1 - 1; i >= 0; --i) {
		int carry = 0;
		for (int j = n2 - 1; j >= 0; --j) {
			int tmp = (res[i + j + 1] - '0') + (num1[i] - '0')*(num2[j] - '0') + carry;
			res[i + j + 1] = tmp % 10 + '0';
			carry = tmp / 10;
		}
		res[i] += carry;
	}
	size_t startpos = res.find_first_not_of('0');
	return (startpos == string::npos ? "0" : res.substr(startpos));
}

// 258. Add Digits, conditions
int addDigits(int num) {
	if (num == 0) return 0;
	if ((num != 0) && (num % 9 == 0)) return 9;
	else return (num % 9); // if ((num != 0) && (num % 9 != 0))
}

// 258. Add Digits, simplication
int addDigits2(int num) {
	return 1 + (num - 1) % 9;
}



int main(){
	PRINT(count_lines());


	system("pause");
	return 0;
}  
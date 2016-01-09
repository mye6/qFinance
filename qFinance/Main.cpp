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




int main(){
	string a = "1"; string b = "101";
	string res = addBinary0(a, b);
	PRINT(res);

	system("pause");
	return 0;
}  
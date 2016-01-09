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

// 326. Power of Three, 




int main(){
	PRINT(fractionToDecimal(2, 3));
	PRINT(count_lines());
	
	system("pause");
	return 0;
}  
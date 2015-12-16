#include "Solver.h"
#include "Finance.h"
#include "Puzzle.h"
#include "Leetcode.h"
#include "ThinkCPP.h"

class Integer {
	long i;
	Integer* This() { return this; }
public:
	Integer(long ll = 0) : i(ll) {}
	
	/* unary operators */ 
	friend const Integer& operator+(const Integer& a) {
		cout << "+Integer\n";
		return a; // Unary + has no effect
	}

	friend const Integer& operator++(Integer& a);	// ++a returns the const reference, notice the difference!!!
	friend const Integer operator++(Integer& a, int); // a++, returns the const object, notice the difference!!!

	/* binary operators */
	friend const Integer operator+(const Integer& left, const Integer& right); // create new, modified value
	
	friend Integer& operator+=(Integer& left, const Integer& right); // assignments modify & return lvalue

	friend ostream& operator<<(std::ostream& os, const Integer& dat) {
		os << dat.i << endl;
		return os;
	}	
};

/* unary operators */
// Prefix; return incremented value
const Integer& operator++(Integer& a) {
	cout << "++Integer\n";
	a.i++;
	return a;
}
// Postfix; return the value before increment:
const Integer operator++(Integer& a, int) {
	cout << "Integer++\n";
	Integer before(a.i);
	a.i++;
	return before;
}

/* binary operators */
const Integer operator+(const Integer& left, const Integer& right) {
	// create new, modified value
	return Integer(left.i + right.i);
}


Integer& operator+=(Integer& left, const Integer& right) {
	// assignments modify & return lvalue
	if (&left == &right) { /* self-assignment*/ }
	left.i += right.i;
	return left;
}

int main() {
	Integer a(2);
	PRINT(+a);
	PRINT(a);
	PRINT(a++);
	PRINT(++a);

	PRINT(a);
	Integer b(1);
	Integer c(2);
	a = b + c;
	PRINT(a);
	a += c;
	PRINT(a);
	a += a;
	PRINT(a);

	PRINT(count_lines());
	system("pause");
	return 0;
}
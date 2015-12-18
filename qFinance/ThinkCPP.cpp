#include "ThinkCPP.h"

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

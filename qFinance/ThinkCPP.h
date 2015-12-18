#ifndef THINKCPP_H
#define THINKCPP_H

#include <iostream>
using namespace std;

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



#endif
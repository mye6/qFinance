#include "Solver.h"

namespace dsaa{
	// class IntCell
	IntCell::IntCell(int dat) : storedValue(dat) {}
	IntCell::IntCell(const IntCell& rhs) : storedValue(rhs.storedValue) {}
	IntCell& IntCell::operator=(const IntCell& rhs) {
		if (&rhs == this) return *this;
		storedValue = rhs.storedValue;
		return *this;
	}
	int IntCell::read() const { return storedValue; };
	void IntCell::write(int x) { storedValue = x; }

	// class IntPtCell
	IntPtCell::IntPtCell(int dat) : storedValue(new int(dat)) {}
	IntPtCell::IntPtCell(const IntPtCell& rhs) : storedValue(new int(*rhs.storedValue)){}
	IntPtCell& IntPtCell::operator = (const IntPtCell& rhs) {
		if (&rhs == this) return *this;
		storedValue = new int(*rhs.storedValue);
		return *this;
	}
	IntPtCell::~IntPtCell() { delete storedValue; }
	int IntPtCell::read() const { return *storedValue; }
	void IntPtCell::write(int x) { *storedValue = x; }

	// class Employee
	Employee::Employee(const string& name_, double salary_, int seniority_) 
		: name(name_), salary(salary_), seniority(seniority_) {}
	void Employee::setValue(const string& n, double s) { name = n; salary = s; }
	const string& Employee::getName() const { return name; }
	void Employee::print(ostream& out) const { out << name << " (" << salary << ", " << seniority << ")"; }
	bool Employee::operator<(const Employee& rhs) const { return salary < rhs.salary; }
	bool Employee::operator==(const Employee& rhs) const { return getName() == rhs.getName(); }
	bool Employee::operator!=(const Employee& rhs) const { return !(*this == rhs); }
	int Employee::hash() const {
		int hashVal = 0;
		for (size_t i = 0; i < name.length(); ++i)
			hashVal = 37 * hashVal + name[i];		
		return hashVal;
	}

	// see url: stackoverflow.com/questions/171862/namespaces-and-operator-overloading-in-c
	ostream& operator<<(ostream& out, const Employee& rhs) { rhs.print(out); return out; }	
}

/**
* A hash routine for string objects.
*/
int hash0(const string & key, int tableSize) {
	int hashVal = 0;
	for (int i = 0; i < key.length(); i++)
		hashVal = 37 * hashVal + key[i];

	hashVal %= tableSize;
	if (hashVal < 0) hashVal += tableSize;

	return hashVal;
}


/**
* A hash routine for ints.
*/
int hash0(int key, int tableSize) {
	if (key < 0) key = -key;
	return key % tableSize;
}

int hash0(const dsaa::Employee& item) {
	int hashVal = 0;
	string name = item.getName();
	for (size_t i = 0; i < name.length(); ++i)
		hashVal = 37 * hashVal + name[i];
	return hashVal;
}

/**
* Internal method to test if a positive number is prime.
* Not an efficient algorithm.
*/
bool isPrime(int n) {
	if (n == 2 || n == 3) return true;
	if (n == 1 || n % 2 == 0) return false;
	for (int i = 3; i * i <= n; i += 2)
		if (n % i == 0) return false;
	return true;
}

int nextPrime(int n) {
	if (n % 2 == 0) n++;
	for (; !isPrime(n); n += 2)
		;
	return n;
}
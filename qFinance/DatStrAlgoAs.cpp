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
	void Employee::setValue(const string& n, double s) { name = n; salary = s; }
	const string& Employee::getName() const { return name; }
	void Employee::print(ostream& out) const { out << name << " (" << salary << ")"; }
	bool Employee::operator<(const Employee& rhs) const { return salary < rhs.salary; }	
	// see url: stackoverflow.com/questions/171862/namespaces-and-operator-overloading-in-c
	ostream& operator<<(ostream& out, const Employee& rhs) { rhs.print(out); return out; }	
}
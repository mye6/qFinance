#ifndef FAQCPP_H
#define FAQCPP_H

#include "Solver.h"

class Open_error {};

// wrap a raw C file handle and put the resource acquisition and release
// in the C++ type's constructor and destructor, respectively
class File_handle {
	FILE* p;
public:
	File_handle(const char* n, const char* a) {
		p = fopen(n, a);
		if (p == 0) throw Open_error();
	}
	File_handle(FILE* pp) {
		p = pp;
		if (p == 0) throw Open_error();
	}
	~File_handle() { fclose(p); }
	operator FILE*() { return p; }   // if desired
	// ...
};
// use File_handle: uses vastly outnumber the above code
void f(const char* fn) {
	File_handle f(fn, "rw"); // open fn for reading and writing
	// use file through f
} // automatically destroy f here, calls fclose automatically with no extra effort
// (even if there's an exception, so this is exception-safe by construction)

#endif
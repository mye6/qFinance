#include "Solver.h"
#include "Finance.h"
#include "Leetcode.h"
#include "Vector.h"
#include "mystring.h"

/*
class string {
public:
	string(const char *cstring = "");               // Constructor
	string(const string & str);                     // Copy constructor
	~string() { delete[] buffer; }					// Destructor
	const string& operator=(const string & rhs);	// Copy
	const string& operator+=(const string & rhs);	// Append
	const char* c_str() const { return buffer; }	// Return C-style string
	int length() const { return strLength; }		// Return string length
	char operator[](int k) const;					// Accessor operator[]
	char& operator[](int k);						// Mutator  operator[]
	enum { MAX_LENGTH = 1024 };						// Maximum length for input string
private:
	char *buffer;                  // storage for characters
	int strLength;                 // length of string (# of characters)
	int bufferLength;              // capacity of buffer
};

std::ostream& operator<<(std::ostream& out, const string& str);    // Output
std::istream& operator>>(std::istream& in, string& str);           // Input
std::istream& getline(std::istream& in, string& str);              // Read line

bool operator==(const string& lhs, const string& rhs);    // Compare ==
bool operator!=(const string& lhs, const string& rhs);    // Compare !=
bool operator< (const string& lhs, const string& rhs);    // Compare <
bool operator<=(const string& lhs, const string& rhs);    // Compare <=
bool operator> (const string& lhs, const string& rhs);    // Compare >
bool operator>=(const string& lhs, const string& rhs);    // Compare >=

*/

int main() {
	string s("bingtuo");
	cout << s << endl;
	s = "maomao";
	cout << s << endl;
	string s2("xiaobao");
	cout << (s < s2) << endl;
	cout << s.length() << endl;


	system("pause");
	return 0;
}
#ifndef MY_STRING_H_
#define MY_STRING_H_

#include <iostream>

/*
* The next line is used because Codewarrior has a conflict with
* the STL string. Make sure to put the #include of this file
* AFTER all the system includes.
*/
#define string String

class StringIndexOutOfBounds {};

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


/* definition*/
string::string(const char* cstring) {
	if (cstring == NULL) cstring = "";
	strLength = strlen(cstring);
	bufferLength = strLength + 1;
	buffer = new char[bufferLength];
	strcpy(buffer, cstring);
}

string::string(const string& str) {
	strLength = str.length();
	bufferLength = strLength + 1;
	buffer = new char[bufferLength];
	strcpy(buffer, str.buffer);
}

const string& string::operator=(const string& rhs) {
	if (this != &rhs) {
		if (bufferLength < rhs.length() + 1) {
			delete[] buffer;
			bufferLength = rhs.length() + 1;
			buffer = new char[bufferLength];
		}
		strLength = rhs.length();
		strcpy(buffer, rhs.buffer);
	}
	return *this;
}

const string& string::operator+=(const string& rhs) {
	if (this == &rhs) {
		string copy(rhs);
		return *this += copy;
	}
	int newLength = length() + rhs.length();
	if (newLength >= bufferLength) {
		bufferLength = 2 * (newLength + 1);
		char *oldBuffer = buffer;
		buffer = new char[bufferLength];
		strcpy(buffer, oldBuffer);
		delete[] oldBuffer;
	}
	strcpy(buffer + length(), rhs.buffer);
	strLength = newLength;
	return *this;
}

char& string::operator[](int k) {
	if (k < 0 || k >= strLength)
		throw StringIndexOutOfBounds();
	return buffer[k];
}

char string::operator[](int k) const {
	if (k < 0 || k >= strLength)
		throw StringIndexOutOfBounds();
	return buffer[k];
}

std::ostream& operator<<(std::ostream& out, const string& str) {
	return out << str.c_str();
}

std::istream& operator>>(std::istream& in, string& str) {
	char buf[string::MAX_LENGTH + 1];
	in >> buf;
	if (!in.fail())
		str = buf;
	return in;
}

std::istream& getline(std::istream& in, string& str) {
	char buf[string::MAX_LENGTH + 1];
	in.getline(buf, string::MAX_LENGTH);
	if (!in.fail())
		str = buf;
	return in;
}

bool operator==(const string& lhs, const string& rhs) {
	return strcmp(lhs.c_str(), rhs.c_str()) == 0;
}

bool operator!=(const string & lhs, const string & rhs) {
	return strcmp(lhs.c_str(), rhs.c_str()) != 0;
}

bool operator<(const string & lhs, const string & rhs) {
	return strcmp(lhs.c_str(), rhs.c_str()) < 0;
}

bool operator<=(const string & lhs, const string & rhs) {
	return strcmp(lhs.c_str(), rhs.c_str()) <= 0;
}

bool operator>(const string & lhs, const string & rhs) {
	return strcmp(lhs.c_str(), rhs.c_str()) > 0;
}

bool operator>=(const string & lhs, const string & rhs) {
	return strcmp(lhs.c_str(), rhs.c_str()) >= 0;
}

#endif

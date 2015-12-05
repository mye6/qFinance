#ifndef MATRIX_H
#define MATRIX_H

#include "Vector.h"

template <class Object>
class matrix {
public:
	matrix(int rows, int cols);
	matrix(const matrix& rhs);
	
	const vector<Object>& operator[](int row) const;
	vector<Object>& operator[](int row);

	int numrows() const;
	int numcols() const;
private:
	vector<vector<Object> > array;
};

// implementations
template <class Object>
matrix<Object>::matrix(int rows, int cols) : array(rows) {
	for (int i = 0; i < rows; i++)
		array[i].resize(cols);
}

template <class Object>
matrix<Object>::matrix(const matrix& rhs) : array(rhs.array) { }

template <class Object>
const vector<Object>& matrix<Object>::operator[](int row) const {
	return array[row];
}

template <class Object>
vector<Object>& matrix<Object>::operator[](int row) {
	return array[row];
}

template <class Object>
int matrix<Object>::numrows() const {
	return array.size();
}

template <class Object>
int matrix<Object>::numcols() const {
	return numrows() ? array[0].size() : 0;
}




#endif
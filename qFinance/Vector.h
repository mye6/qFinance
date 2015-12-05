#ifndef VECTOR_H
#define VECTOR_H

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include "dsexceptions.h"


template <typename Object>
class Vector {
public:
	explicit Vector(int initSize = 0); // constructor (default)
	Vector(const Vector& rhs); // copy constructor
	Vector& operator=(const Vector& rhs); // assignment operator
	~Vector(); // destructor
	Vector(Vector&& rhs); // move constructor
	Vector& operator= (Vector&& rhs); // move assignment operator
	bool empty() const; // is empty?
	int size() const; // size?
	int capacity() const; // capacity?
	Object& operator[](int index); // index'th element, non-constant version
	const Object& operator[](int index) const; // index'th element, constant version
	void resize(int newSize); // resize the Vector
	void reserve(int newCapacity); // allocate new space
	// Stacky stuff
	void push_back(const Object& x); // add one element to the back by coping
	void push_back(Object&& x); // add one element to the back by moving
	void pop_back(); // remove one element from the back
	const Object& back() const; // retrieve the element in the back

	// Iterator stuff: not bounds checked
	typedef Object * iterator;
	typedef const Object * const_iterator;
	iterator begin() { return &objects[0]; }
	const_iterator begin() const { return &objects[0]; }
	iterator end() { return &objects[size()]; }
	const_iterator end() const { return &objects[size()]; }	
private:
	static const int SPARE_CAPACITY = 2;
	int theSize;
	int theCapacity;
	Object* objects;
};

// constructor
template <typename Object>
Vector<Object>::Vector(int initSize)
	: theSize{ initSize }, theCapacity{ initSize + SPARE_CAPACITY } {

	objects = new Object[theCapacity];
}

// copy constructor
template <typename Object>
Vector<Object>::Vector(const Vector<Object>& rhs)
	: theSize{ rhs.theSize }, theCapacity{ rhs.theCapacity }, objects{ nullptr } {

	objects = new Object[theCapacity];
	for (int k = 0; k < theSize; ++k)
		objects[k] = rhs.objects[k];
}

// assignment operator
template <typename Object>
Vector<Object>& Vector<Object>::operator=(const Vector<Object>& rhs) {
	Vector<Object> copy = rhs;
	std::swap(*this, copy);
	return *this;
}

// destructor
template <typename Object>
Vector<Object>::~Vector() {
	delete[] objects;
}

// move constructor
template <typename Object>
Vector<Object>::Vector(Vector<Object> && rhs)
	: theSize{ rhs.theSize }, theCapacity{ rhs.theCapacity }, objects{ rhs.objects } {
	rhs.objects = nullptr;
	rhs.theSize = 0;
	rhs.theCapacity = 0;
}

// move assignment operator
template <typename Object>
Vector<Object>& Vector<Object>::operator= (Vector<Object>&& rhs) {
	std::swap(theSize, rhs.theSize);
	std::swap(theCapacity, rhs.theCapacity);
	std::swap(objects, rhs.objects);
	return *this;
}

template <typename Object>
bool Vector<Object>::empty() const {
	return size() == 0;
}

template <typename Object>
int Vector<Object>::size() const {
	return theSize;
}

template <typename Object>
int Vector<Object>::capacity() const {
	return theCapacity;
}

template <typename Object>
Object& Vector<Object>::operator[](int index) {
#ifndef NO_CHECK
	if (index < 0 || index >= size())
		throw ArrayIndexOutOfBoundsException{};
#endif
	return objects[index];
}

template <typename Object>
const Object& Vector<Object>::operator[](int index) const {
#ifndef NO_CHECK
	if (index < 0 || index >= size())
		throw ArrayIndexOutOfBoundsException{};
#endif
	return objects[index];
}

template <typename Object>
void Vector<Object>::resize(int newSize) {
	if (newSize > theCapacity)
		reserve(newSize * 2);
	theSize = newSize;
}

template <typename Object>
void Vector<Object>::reserve(int newCapacity) {
	if (newCapacity < theSize)
		return;

	Object *newArray = new Object[newCapacity];
	for (int k = 0; k < theSize; ++k)
		newArray[k] = std::move(objects[k]);

	theCapacity = newCapacity;
	std::swap(objects, newArray);
	delete[] newArray;
}

template <typename Object>
void Vector<Object>::push_back(const Object& x) {
	if (theSize == theCapacity)
		reserve(2 * theCapacity + 1);
	objects[theSize++] = x;
}

template <typename Object>
void Vector<Object>::push_back(Object&& x) {
	if (theSize == theCapacity)
		reserve(2 * theCapacity + 1);
	objects[theSize++] = std::move(x);
}

template <typename Object>
void Vector<Object>::pop_back() {
	if (empty())
		throw UnderflowException{};
	--theSize;
}

template <typename Object>
const Object& Vector<Object>::back() const {
	if (empty())
		throw UnderflowException{};
	return objects[theSize - 1];
}

#endif
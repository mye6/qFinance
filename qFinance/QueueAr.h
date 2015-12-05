#ifndef QUEUEAR_H
#define QUEUEAR_H

#include "Vector.h"
#include "dsexceptions.h"

// Queue class -- array implementation
//
// CONSTRUCTION: with or without a capacity; default is 10
//
// ******************PUBLIC OPERATIONS*********************
// void enqueue( x )      --> Insert x
// void dequeue( )        --> Return and remove least recently inserted item
// Object getFront( )     --> Return least recently inserted item
// bool isEmpty( )        --> Return true if empty; else false
// bool isFull( )         --> Return true if full; else false
// void makeEmpty( )      --> Remove all items
// ******************ERRORS********************************
// Overflow and Underflow thrown as needed

template <class Object>
class Queue {
public:
	explicit Queue(int capacity = 10);

	bool isEmpty() const;
	bool isFull() const;
	const Object& getFront() const;

	void makeEmpty();
	Object dequeue();
	void enqueue(const Object & x);

private:
	vector<Object> theArray;
	int            currentSize;
	int            front;
	int            back;

	void increment(int & x);
};

// definitions
/**
* Construct the queue.
*/
template <class Object>
Queue<Object>::Queue(int capacity) : theArray(capacity) {
	makeEmpty();
}

/**
* Test if the queue is logically empty.
* Return true if empty, false otherwise.
*/
template <class Object>
bool Queue<Object>::isEmpty() const {
	return currentSize == 0;
}

/**
* Test if the queue is logically full.
* Return true if full, false otherwise.
*/
template <class Object>
bool Queue<Object>::isFull() const {
	return currentSize == theArray.size();
}

/**
* Make the queue logically empty.
*/
template <class Object>
void Queue<Object>::makeEmpty() {
	currentSize = 0;
	front = 0;
	back = -1;
}

/**
* Get the least recently inserted item in the queue.
* Return the least recently inserted item in the queue
* or throw Underflow if empty.
*/
template <class Object>
const Object& Queue<Object>::getFront() const {
	if (isEmpty())
		throw Underflow();
	return theArray[front];
}

/**
* Return and remove the least recently inserted item from the queue.
* Throw Underflow if empty.
*/
template <class Object>
Object Queue<Object>::dequeue() {
	if (isEmpty())
		throw Underflow();

	currentSize--;
	Object frontItem = theArray[front];
	increment(front);
	return frontItem;
}

/**
* Insert x into the queue.
* Throw Overflow if queue is full
*/
template <class Object>
void Queue<Object>::enqueue(const Object& x) {
	if (isFull())
		throw Overflow();
	increment(back);
	theArray[back] = x;
	currentSize++;
}

/**
* Internal method to increment x with wraparound.
*/
template <class Object>
void Queue<Object>::increment(int& x) {
	if (++x == theArray.size()) // no need to take module every time, only when x reaches the size
		x = 0;
}


#endif
// QuadraticProbing.h
#ifndef QUADRATIC_PROBING_H_
#define QUADRATIC_PROBING_H_

#include "Vector.h"
#include "String.h"
#include <iostream>

// QuadraticProbing Hash table class
//
// CONSTRUCTION: an initialization for ITEM_NOT_FOUND
//               and an approximate initial size or default of 101
//
// ******************PUBLIC OPERATIONS*********************
// void insert( x )       --> Insert x
// void remove( x )       --> Remove x
// Hashable find( x )     --> Return item that matches x
// void makeEmpty( )      --> Remove all items

template <class HashedObj>
class HashTable {
public:
	explicit HashTable(const HashedObj & notFound, int size = 101);
	HashTable(const HashTable & rhs)
		: ITEM_NOT_FOUND(rhs.ITEM_NOT_FOUND),
		array(rhs.array), currentSize(rhs.currentSize) { }

	const HashedObj& find(const HashedObj& x) const;

	void makeEmpty();
	void insert(const HashedObj& x);
	void remove(const HashedObj& x);

	const HashTable& operator=(const HashTable& rhs);

	enum EntryType { ACTIVE, EMPTY, DELETED };
private:
	struct HashEntry {
		HashedObj element;
		EntryType info;
		HashEntry(const HashedObj& e = HashedObj(), EntryType i = EMPTY)
			: element(e), info(i) { }
	};

	vector<HashEntry> array;
	int currentSize;
	const HashedObj ITEM_NOT_FOUND;

	bool isActive(int currentPos) const;
	int findPos(const HashedObj& x) const;
	void reHash();
};

int Hash(const string& key, int tableSize); // hash a string
int Hash(int key, int tableSize); // hash an integer

// definitions
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

/**
* Internal method to return a prime number at least as large as n.
* Assumes n > 0.
*/
int nextPrime(int n) {
	if (n % 2 == 0) n++;
	for (; !isPrime(n); n += 2)
		;

	return n;
}

/**
* Construct the hash table.
*/
template <class HashedObj>
HashTable<HashedObj>::HashTable(const HashedObj& notFound, int size)
	: ITEM_NOT_FOUND(notFound), array(nextPrime(size)) { makeEmpty(); }

/**
* Insert item x into the hash table. If the item is
* already present, then do nothing.
*/
template <class HashedObj>
void HashTable<HashedObj>::insert(const HashedObj& x) {
	// Insert x as active
	int currentPos = findPos(x);
	if (isActive(currentPos)) // x already exists, do nothing
		return;

	if (array[currentPos].info != DELETED)
		++currentSize; 
		// only increase the size where a new space is taken (not one that was previously deleted)

	array[currentPos] = HashEntry(x, ACTIVE);

	// Rehash; see Section 5.5
	if (currentSize > (int)array.size() / 2)
		reHash();
	// main the occupied size is always less than half, to ensure not too full
	// when the load factor is above 0.5
}

/**
* Expand the hash table.
*/
template <class HashedObj>
void HashTable<HashedObj>::reHash() {
	vector<HashEntry> oldArray = array;

	// Create new double-sized, empty table
	array.resize(nextPrime(2 * oldArray.size()));
	// always use prime numbers for the table size

	for (int j = 0; j < (int)array.size(); j++)
		array[j].info = EMPTY;

	// Copy table over
	currentSize = 0;
	for (int i = 0; i < (int)oldArray.size(); i++)
		if (oldArray[i].info == ACTIVE)
			insert(oldArray[i].element);
}

/**
* Method that performs quadratic probing resolution.
* Return the position where the search for x terminates.
*/
template <class HashedObj>
int HashTable<HashedObj>::findPos(const HashedObj& x) const {
	int collisionNum = 0;
	int currentPos = Hash(x, array.size());

	while (array[currentPos].info != EMPTY && array[currentPos].element != x) {
		// order is important here, don't switch
		currentPos += 2 * ++collisionNum - 1;  // Compute ith probe
		// note: i^2 = (i-1)^2 + (2*i - 1)
		
		if (currentPos >= (int)array.size())
			currentPos -= array.size();
	}
	return currentPos;
}


/**
* Remove item x from the hash table.
*/
template <class HashedObj>
void HashTable<HashedObj>::remove(const HashedObj& x) {
	int currentPos = findPos(x);
	if (isActive(currentPos))
		array[currentPos].info = DELETED;
}

/**
* Find item x in the hash table.
* Return the matching item or ITEM_NOT_FOUND if not found
*/
template <class HashedObj>
const HashedObj& HashTable<HashedObj>::find(const HashedObj& x) const {
	int currentPos = findPos(x);
	if (isActive(currentPos))
		return array[currentPos].element;
	else
		return ITEM_NOT_FOUND;
}

/**
* Make the hash table logically empty.
*/
template <class HashedObj>
void HashTable<HashedObj>::makeEmpty() {
	currentSize = 0;
	for (int i = 0; i < (int)array.size(); i++)
		array[i].info = EMPTY;
}

/**
* Deep copy.
*/
template <class HashedObj>
const HashTable<HashedObj>& 
HashTable<HashedObj>::operator=(const HashTable<HashedObj>& rhs) {
	if (this != &rhs) {
		array = rhs.array;
		currentSize = rhs.currentSize;
	}
	return *this;
}

/**
* Return true if currentPos exists and is active.
*/
template <class HashedObj>
bool HashTable<HashedObj>::isActive(int currentPos) const {
	return array[currentPos].info == ACTIVE;
}

/**
* A hash routine for string objects.
*/
int Hash(const string& key, int tableSize) {
	int hashVal = 0;

	for (int i = 0; i < (int)key.length(); i++)
		hashVal = 37 * hashVal + key[i];

	hashVal %= tableSize;
	if (hashVal < 0)
		hashVal += tableSize;

	return hashVal;
}

/**
* A hash routine for ints.
*/
int Hash(int key, int tableSize) {
	if (key < 0) key = -key;
	return key % tableSize;
}

#endif
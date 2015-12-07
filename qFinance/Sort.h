#ifndef SORT_H_
#define SORT_H_

#define merge Merge
#define swap Swap

/**
* Several sorting routines.
* Arrays are rearranged with smallest item first.
*/
#include "Vector.h"

class Random {
public:
	explicit Random(int initialValue = 1);

	int randomInt();
	int randomIntWRONG();
	double random0_1();
	int randomInt(int low, int high);
private:
	int state;
};

/**
* Simple insertion sort.
*/
template <class Comparable>
void insertionSort(vector<Comparable>& a) {
	for (size_t p = 1; p < a.size(); p++) {
		Comparable tmp = a[p];
		int j;
		for (j = p; j > 0 && tmp < a[j - 1]; j--)
			a[j] = a[j - 1];		
		a[j] = tmp;
	}
}

/**
* Shellsort, using Shell's (poor) increments.
*/
template <class Comparable>
void shellsort(vector<Comparable> & a) {
	for (int gap = (int)a.size() / 2; gap > 0; gap /= 2)
		for (int i = gap; i < (int)a.size(); i++) {
			Comparable tmp = a[i];
			
			int j = i;
			for (; j >= gap && tmp < a[j - gap]; j -= gap)
				a[j] = a[j - gap];
			
			a[j] = tmp;
		}
}

/**
* Standard heapsort.
*/
template <class Comparable>
void heapsort(vector<Comparable>& a) {
	/* 1*/      for (int i = a.size() / 2; i >= 0; i--)  /* buildHeap */
		/* 2*/          percDown(a, i, a.size());
	/* 3*/      for (int j = a.size() - 1; j > 0; j--) {
		/* 4*/          swap(a[0], a[j]);                  /* deleteMax */
		/* 5*/          percDown(a, 0, j);
	}
}

/**
* Internal method for heapsort.
* i is the index of an item in the heap.
* Returns the index of the left child.
*/
inline int leftChild(int i) {
	return 2 * i + 1;
}

/**
* Internal method for heapsort that is used in
* deleteMax and buildHeap.
* i is the position from which to percolate down.
* n is the logical size of the binary heap.
*/
template <class Comparable>
void percDown(vector<Comparable> & a, int i, int n) {
	int child;
	Comparable tmp;

	for (tmp = a[i]; leftChild(i) < n; i = child) {
		child = leftChild(i);
		if (child != n - 1 && a[child] < a[child + 1])
			child++;
		if (tmp < a[child])
			a[i] = a[child];
		else
			break;
	}
	a[i] = tmp;
}

/**
* Mergesort algorithm (driver).
*/
template <class Comparable>
void mergeSort(vector<Comparable>& a) {
	vector<Comparable> tmpArray(a.size());
	mergeSort(a, tmpArray, 0, a.size() - 1);
}

/**
* Internal method that makes recursive calls.
* a is an array of Comparable items.
* tmpArray is an array to place the merged result.
* left is the left-most index of the subarray.
* right is the right-most index of the subarray.
*/
template <class Comparable>
void mergeSort(vector<Comparable>& a,
	vector<Comparable> & tmpArray, int left, int right) {
	if (left < right) {
		int center = (left + right) / 2;
		mergeSort(a, tmpArray, left, center);
		mergeSort(a, tmpArray, center + 1, right);
		merge(a, tmpArray, left, center + 1, right);
	}
}

/**
* Internal method that merges two sorted halves of a subarray.
* a is an array of Comparable items.
* tmpArray is an array to place the merged result.
* leftPos is the left-most index of the subarray.
* rightPos is the index of the start of the second half.
* rightEnd is the right-most index of the subarray.
*/
template <class Comparable>
void merge(vector<Comparable> & a, vector<Comparable> & tmpArray,
	int leftPos, int rightPos, int rightEnd) {

	int leftEnd = rightPos - 1;
	int tmpPos = leftPos;
	int numElements = rightEnd - leftPos + 1;

	// Main loop
	while (leftPos <= leftEnd && rightPos <= rightEnd)
		if (a[leftPos] <= a[rightPos])
			tmpArray[tmpPos++] = a[leftPos++];
		else
			tmpArray[tmpPos++] = a[rightPos++];

	while (leftPos <= leftEnd)    // Copy rest of first half
		tmpArray[tmpPos++] = a[leftPos++];

	while (rightPos <= rightEnd)  // Copy rest of right half
		tmpArray[tmpPos++] = a[rightPos++];

	// Copy tmpArray back
	for (int i = 0; i < numElements; i++, rightEnd--)
		a[rightEnd] = tmpArray[rightEnd];
}

/**
* Quicksort algorithm (driver).
*/
template <class Comparable>
void quicksort(vector<Comparable> & a)
{
	quicksort(a, 0, a.size() - 1);
}

/**
* Standard swap
*/
template <class Comparable>
inline void swap(Comparable & obj1, Comparable & obj2)
{
	Comparable tmp = obj1;
	obj1 = obj2;
	obj2 = tmp;
}

/**
* Return median of left, center, and right.
* Order these and hide the pivot.
*/
template <class Comparable>
const Comparable & median3(vector<Comparable> & a, int left, int right)
{
	int center = (left + right) / 2;
	if (a[center] < a[left])
		swap(a[left], a[center]);
	if (a[right] < a[left])
		swap(a[left], a[right]);
	if (a[right] < a[center])
		swap(a[center], a[right]);

	// Place pivot at position right - 1
	swap(a[center], a[right - 1]);
	return a[right - 1];
}

/**
* Internal quicksort method that makes recursive calls.
* Uses median-of-three partitioning and a cutoff of 10.
* a is an array of Comparable items.
* left is the left-most index of the subarray.
* right is the right-most index of the subarray.
*/
template <class Comparable>
void quicksort(vector<Comparable> & a, int left, int right)
{
	/* 1*/      if (left + 10 <= right)
	{
		/* 2*/          Comparable pivot = median3(a, left, right);

		// Begin partitioning
		/* 3*/          int i = left, j = right - 1;
		/* 4*/          for (;;)
		{
			/* 5*/              while (a[++i] < pivot) {}
			/* 6*/              while (pivot < a[--j]) {}
			/* 7*/              if (i < j)
				/* 8*/                  swap(a[i], a[j]);
			else
				/* 9*/                  break;
		}

		/*10*/          swap(a[i], a[right - 1]);  // Restore pivot

		/*11*/          quicksort(a, left, i - 1);     // Sort small elements
		/*12*/          quicksort(a, i + 1, right);    // Sort large elements
	}
	else  // Do an insertion sort on the subarray
		/*13*/          insertionSort(a, left, right);
}

/**
* Internal insertion sort routine for subarrays
* that is used by quicksort.
* a is an array of Comparable items.
* left is the left-most index of the subarray.
* right is the right-most index of the subarray.
*/
template <class Comparable>
void insertionSort(vector<Comparable> & a, int left, int right)
{
	for (int p = left + 1; p <= right; p++)
	{
		Comparable tmp = a[p];
		int j;

		for (j = p; j > left && tmp < a[j - 1]; j--)
			a[j] = a[j - 1];
		a[j] = tmp;
	}
}

/**
* Quick selection algorithm.
* Places the kth smallest item in a[k-1].
* a is an array of Comparable items.
* k is the desired rank (1 is minimum) in the entire array.
*/
template <class Comparable>
void quickSelect(vector<Comparable> & a, int k)
{
	quickSelect(a, 0, a.size() - 1, k);
}


/**
* Internal selection method that makes recursive calls.
* Uses median-of-three partitioning and a cutoff of 10.
* Places the kth smallest item in a[k-1].
* a is an array of Comparable items.
* left is the left-most index of the subarray.
* right is the right-most index of the subarray.
* k is the desired rank (1 is minimum) in the entire array.
*/
template <class Comparable>
void quickSelect(vector<Comparable> & a, int left, int right, int k)
{
	/* 1*/      if (left + 10 <= right)
	{
		/* 2*/          Comparable pivot = median3(a, left, right);

		// Begin partitioning
		/* 3*/          int i = left, j = right - 1;
		/* 4*/          for (;;)
		{
			/* 5*/              while (a[++i] < pivot) {}
			/* 6*/              while (pivot < a[--j]) {}
			/* 7*/              if (i < j)
				/* 8*/                  swap(a[i], a[j]);
			else
				/* 9*/                  break;
		}

		/*10*/          swap(a[i], a[right - 1]);  // Restore pivot

		// Recurse; only this part changes
		/*11*/          if (k <= i)
			/*12*/              quickSelect(a, left, i - 1, k);
		/*13*/          else if (k > i + 1)
			/*14*/              quickSelect(a, i + 1, right, k);
	}
	else  // Do an insertion sort on the subarray
		/*15*/          insertionSort(a, left, right);
}

/**
* Class that wraps a pointer variable.
*/
template <class Comparable>
class Pointer
{
public:
	Pointer(Comparable *rhs = NULL) : pointee(rhs) { }

	bool operator<(const Pointer & rhs) const
	{
		return *pointee < *rhs.pointee;
	}
	operator Comparable * () const
	{
		return pointee;
	}
private:
	Comparable *pointee;
};

/**
* Sort objects -- even large ones --
* with only N + ln N Comparable moves on average.
*/
template <class Comparable>
void largeObjectSort(vector<Comparable>& a) {
	vector<Pointer<Comparable> > p(a.size());
	int i, j, nextj;

	for (i = 0; i < (int)a.size(); i++)
		p[i] = &a[i];

	quicksort(p);

	// Shuffle items in place
	for (i = 0; i < (int)a.size(); i++) {
		if (p[i] != &a[i]) {
			Comparable tmp = a[i];
			for (j = i; p[j] != &a[i]; j = nextj) {
				nextj = p[j] - &a[0];
				a[j] = *p[j];
				p[j] = &a[j];
			}

			a[j] = tmp;
			p[j] = &a[j];
		}
	}
}



// Random definitions
static const int A = 48271;
static const int M = 2147483647;
static const int Q = M / A;
static const int R = M % A;

/**
* Construct with initialValue for the state.
*/
Random::Random(int initialValue)
{
	if (initialValue < 0)
		initialValue += M;

	state = initialValue;
	if (state == 0)
		state = 1;
}

/**
* Return a pseudorandom int, and change the
* internal state.
*/
int Random::randomInt()
{
	int tmpState = A * (state % Q) - R * (state / Q);
	if (tmpState >= 0)
		state = tmpState;
	else
		state = tmpState + M;

	return state;
}

/**
* Return a pseudorandom int, and change the
* internal state. DOES NOT WORK.
*/
int Random::randomIntWRONG()
{
	return state = (A * state) % M;
}

/**
* Return a pseudorandom double in the open range 0..1
* and change the internal state.
*/
double Random::random0_1()
{
	return (double)randomInt() / M;
}

/**
* Return an int in the closed range [low,high], and
* change the internal state.
*/
int Random::randomInt(int low, int high)
{
	double partitionSize = (double)M / (high - low + 1);

	return (int)(randomInt() / partitionSize) + low;
}


#endif
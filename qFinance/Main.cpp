#include "Solver.h"
#include "Finance.h"
#include "Leetcode.h"

/*
* Bubble sort: 
* 1. each iteration, compare each pair of adjacent items and swap them if wrong order
* 2. i: 0 to n-1, j: 0 to n-1-i; compare a[j] and a[j+1] and swap if needed
* Complexity: O(n^2)
*/
template<class T>
void swapT(T& x, T& y) {
	T tmp = x;
	x = y;
	y = tmp;
}

template<class T>
void bubble_sort(vector<T>& a) {
	int n = a.size();
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n-1-i; ++j)
			if (a[j]>a[j + 1]) swapT<T>(a[j], a[j + 1]);
}

/*
* Insertion sort:
* 1. each iteration, comparison sort in which the sorted array (or list) is built (0..i)
* 2. implemented by comparing key with each element
* Complexity: O(n^2)
*/
template <class T>
void insertion_sort(vector<T>& a) {
	int n = a.size();
	T key;
	for (int i = 1; i<n; ++i){
		key = a[i];
		int j = i - 1;
		while (j >= 0 && a[j]>key){
			a[j + 1] = a[j];
			j = j - 1;
		}
		a[j + 1] = key;
	}
}

/*
* Selection sort:
* 1. each iteration, get the minimum element in (i..n-1) to ith place
* 2. use min_elm to find the location of the minimum element
* Complexity: O(n^2)
*/
template <class T>
int min_elm(vector<T>& a, int low, int up) {
	int min = low;
	while (low<up) {
		if (a[low]<a[min]) min = low;
		low++;
	}
	return min;
}
template <class T> 
void selection_sort(vector<T>& a) {
	int n = a.size();	
	for (int i = 0; i<n; i++) {		
		swapT<T>(a[min_elm(a, i, n)], a[i]);
	}
}

/*
* Counting sort:
* requires input: [0..m], small m
* 1. m: max element of the nums
* 2. tmp: contains the count for each element, size: m+1
* 3. output tmp back to a
* Complexity: O(n)
*/
template <class T>
void counting_sort(vector<int>& a) {
	int n = a.size(), m = 0;
	for (int i = 0; i < n; ++i) m = max(m, a[i]); // 1. m: max element of the nums
	vector<int> tmp(m+1, 0);
	for (int i = 0; i < n; ++i) ++tmp[a[i]]; 
	//2. tmp: contains the count for each element, size : m + 1
	int k = 0;
	for (int i = 0; i <= m; ++i) {
		for (int j = 0; j < tmp[i]; ++j) a[k++] = i;		
	}
	// 3. output tmp back to a
}

/*
* Heap sort:
* Max-heap: all nodes are greater than or equal to each of its children,
* a[1] is the root for a[1..n], maximum element in max-heap a[1..n]
* 1. insert INT_MIN at a[0], and build max-heap on a[1..n]
* 2. to build max-heap, iterate i=n/2 to 1, max_heapify ith node recursively
* 3. to max_heapify the ith node, find the index of i, leftChild, rightChild as lar; swap a[i], a[lar]; max_heapify lar's node
* 4. after max-heap is built, switch a[1] to the ith element, and perform max_heapify through a[1..i]
* Complexity: O(nlog(n))
*/
template <class T>
void max_heapify(vector<T>& a, int i, int n) {
	// l: index of left child, r: index of right child
	// a[1..n], no need to consider a[0]==INT_MIN
	int l = 2*i, r = (2*i)+1, lar;

	// find the biggest element among i, l, r
	if (l <= n && a[l]>a[i]) lar = l;
	else lar = i;
	if (r <= n && a[r]>a[lar]) lar = r;
	
	// swap a[i] and a[lar], and keep max_heapify a[lar]
	if (lar != i) {
		swapT<T>(a[i], a[lar]);
		max_heapify(a, lar, n);
	}
}
template <class T>
void build_max_heap(vector<T>& a, int n) {
	for (int i = n / 2; i >= 1; i--)
		max_heapify(a, i, n);
}
template <class T>
void heap_sort(vector<T>& a) {
	// insert 0 at front to make the indexing easier, l = 2*i, r = (2*i)+1
	a.insert(a.begin(), INT_MIN);
	int n = a.size() - 1;
	build_max_heap(a, n);
	// each iteration, put the maximum element at the end
	// a[1] is always the maximum element
	for (int i = n; i >= 2; --i) {
		swapT<T>(a[1], a[i]);
		n = n - 1;
		max_heapify(a, 1, n);
	}
	a.erase(a.begin());
}

/*
* Quick sort:
* not only for educational purposes, but widely applied in practice, O(nlog(n))
* divide-and-conquer strategy is used.
* Recursion steps:
* 1. Choose a pivot value, can be any value. Here choose the last element as pivot
* 2. Partition. Rearrange elements in such a way, that all elements which are lesser than the pivot 
*    go to the left part of the array and all elements greater than the pivot, go to the right part of the array.
*    Values equal to the pivot can stay in any part of the array. Notice, that array may be divided in non-equal parts.
* 3. Sort both parts. Apply quicksort algorithm recursively to the left and the right parts.
* Worst: O(n^2), Best: O(nlog(n)), Average: O(nlog(n))
*/
template <class T>
int partition(vector<T>& a, int p, int r) {	
	T x = a[r]; // pivot
	int i = p - 1; // Index of smaller element
	for (int j = p; j <= r - 1; j++) {
		// If current element is smaller than or equal to pivot 
		if (a[j] <= x) {
			++i;  // increment index of smaller element
			swapT<T>(a[i], a[j]); // Swap current element with index
		}
	}
	swapT<T>(a[i + 1], a[r]);
	return i + 1;
}

template <class T>
void quick_sort(vector<T>& a, int p, int r) {	
	if (p<r) {
		int q = partition(a, p, r); /* Partitioning index */
		quick_sort(a, p, q - 1);
		quick_sort(a, q + 1, r);
	}
}
template <class T>
void quick_sort(vector<T>& a) {
	int n = a.size();
	quick_sort(a, 0, n - 1);
}




int main() {	
	vector<int> vec{ 10, 20, 15, 30, 20, 10, 10, 20, 9, 21 };
	PRINT(vec); SEP;
	quick_sort<int>(vec);

	PRINT(vec);
	





	system("pause");
	return 0;
}
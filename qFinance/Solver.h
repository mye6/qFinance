#ifndef SOLVER_H
#define SOLVER_H

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <numeric>
#include <cassert>
#include <map>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
using namespace std;
using namespace Eigen;

const double PI = 3.141592653589793238462643;


// Section: Data
// handle data input and output
vector<double> read_rev_csv(const string& file_path);
// read a sequence of data from csv file into a vector and reverse

class OnlineQuantile {
	vector<double> Q, Np, dNp, N;	
	double med;
public:
	OnlineQuantile(vector<double> dat, double p = 0.5); // constructor uses a 5-element dat vector
	void addNumber(double dat);
	double getValue() const;
};

class OnlineQuantileB {
	vector<double> Q, N; // Np, dNp, N;
	int B;
	double p;
public:
	OnlineQuantileB(vector<double> dat, double p = 0.5); // constructor uses a 5-element dat vector
	void addNumber(double dat);
	double getValue() const;
};

class DataFrame {
	// DataFrame implementation like Pandas in Python
	map<string, vector<double> > mp;
public:
	DataFrame(const string& file_path, char sep = ',');
	DataFrame(const vector<string>& keys, const vector<vector<double> >& data);
	DataFrame(const DataFrame& df);
	DataFrame& operator=(const DataFrame& rhs);
	vector<string> keys() const;
	vector<vector<double> > data();
	vector<double> getCol(const string& col);
	vector<int> dim() const;	
	void to_csv(const string& file_path);
};

// Section: Random
double unif(double lower = 0.0, double upper = 1.0); // generate a unif random number
double normal(double mu = 0.0, double sigma = 1.0); // generate a normal random number
// box-muller method
double N(double z); // cumulative density function of normal distribution
double N(double a, double b, double rho);
double n(double z); // probability density function of normal distribution



// Section: Processor
bool avg_lower(const vector<float>& prices, int M = 5, float P = 1500.);
// compare average value of the M smallest elements in vector dat is <= P
// if avg <= P, return true; otherwise return false

// Section: EigenIO
VectorXd vecGen(const vector<double> dat, int nrow = -1); // generate a Vector through a 1-d std::vector

MatrixXd matGen(vector<double> dat, int nrow = 2); // generate a matrix through a 1-d std::vector with number of columns

MatrixXd matGen(vector<vector<double>> dat);

vector<double> vec2vec1d(const VectorXd& vec);

vector<double> mat2vec1d(const MatrixXd& mat);

vector<vector<double> > mat2vec2d(const MatrixXd& mat);

void lu_fp_decomp(const MatrixXd& A, MatrixXd& P, MatrixXd& Q, MatrixXd& L, MatrixXd& U);
// P*A*Q = L*U, full pivoting

void lu_pp_decomp(const MatrixXd& A, MatrixXd& P, MatrixXd& L, MatrixXd& U);
// P*A = L*U, partial pivoting

void ldlt_decomp(const MatrixXd& mat, MatrixXd& L, MatrixXd& D);
void ldlt_decomp(const MatrixXd& mat);
void ldlt_decomp(vector<vector<double> > dat, vector<vector<double> >& L1, vector<vector<double> >& D1);
void ldlt_decomp(vector<vector<double> > dat);

// 

// Section: Utility
// print helper
#define PRINT(EX) cout << #EX << ": " << endl << EX << endl
#define SEP cout << string(60, '-') << endl

int sgn(double x);

// common and simple functions
template <class T>
void print_vec(const vector<T>& vec, string s = " ", size_t sz = 10) {
	// print the first sz elements of a vector; if sz > vector size, just the vector
	if (sz > vec.size()) sz = vec.size();
	for (size_t i = 0; i < sz - 1; ++i) cout << vec.at(i) << s;
	cout << vec.at(sz-1) << endl;
}

template <class T>
void print_vec2d(const vector<vector<T> >& vec2d, string s = " ", size_t m = 10, size_t n = 10) {
	// print Row: 1-to-min(m, mat2d.size), Col: 1-to-min(n, mat2d[i].size) of a mat2d; 
	if (m > vec2d.size()) m = vec2d.size();
	for (size_t i = 0; i < m; ++i) {
		if (n > vec2d[i].size()) n = vec2d[i].size();
		for (size_t j = 0; j < n - 1; ++j) {
			cout << vec2d[i][j] << s;
		}
		cout << vec2d[i][n - 1] << endl;
	}
}

// cout << vec;
template <class T>
ostream& operator<<(ostream& os, const vector<T>& vec) {
	for (size_t i = 0; i < vec.size(); ++i) os << vec.at(i) << " ";
	return os;
}

// 
template <class T>
ostream& operator<<(ostream& os, const vector<vector<T> >& vec2d) {
	for (size_t i = 0; i < vec2d.size() - 1; ++i) os << vec2d[i] << endl;
	os << vec2d.back();
	return os;
}

vector<vector<double> > reshape_vec1d(const vector<double>& vec, int nrow = 2); // reshape 1-d vector into 2-d

vector<double> flatten_vec2d(const vector<vector<double> >& dat); // flatten 2-d vector into 1-d

void print_map(const map<string, vector<double>>& mp, size_t sz = 10);

// output a generic pair
template<typename Key, typename Value>
std::ostream& operator<<(std::ostream& os, const std::pair<const Key, Value>& p) {
	os << p.first << " => " << p.second;
	return os;
}

// output a generic map
template<typename Key, typename Value>
std::ostream& operator<<(std::ostream& os, const std::map<Key, Value>& mp) {
	for (map<Key, Value>::const_iterator it = mp.begin(); it != mp.end(); ++it)
		os << *it << endl;
	return os;
}


vector<double> sub_vec(const vector<double>& vec, int first = 0, int len = 1);
// obtain the sub vector with length len starting from the first element
	
double power_rec(double base, int n);
// obtain the power function through recursion
// use memorization to cache intermediate results (static vector) to achieve higher efficiency

double exp_taylor(double x);
// obtain the exp function by multiplication/addition/power through Taylor expansion

vector<int> prime_vec(int N);
// obtain all the prime numbers <= N

/*
convert a 0-255 unsigned char to 8-digit-binary string
val: 255
output: 11111111
*/
string toBinary(const unsigned char val = 255);

void printBinary(const unsigned char val = 255);

/*
* Fisher–Yates shuffle Algorithm
* O(n) time complexity
* given a function rand()
* The idea is to start from the last element,
* swap it with a randomly selected element from the whole array (including last)
*/
void swap(int& x, int& y);
void randomize(vector<int>& a);

/*
* XOR swap uses XOR bitwise operation
* A^B = B^A
* (A^B)^C = A^(B^C)
* A^0 = A
* A^A = 0
*/
void swap_xor(int& x, int& y);

/* generate a random number within [0..n] */
int randi(int n);

/*
* For example for 4 ( 100) and 16(10000), we get following after subtracting 1
* 3 –> 011
* 15 –> 01111
* !(x&(x - 1))
* x&(x-1) will get an integer, whose digits are determined by x
*
*/
bool isPowerOfTwo(int x);

/*
count how many lines in a file
input: filename
output: number of lines
*/
int count_lines(const string& filename);

/*
count how many lines in a number of files
input: vector of filenames
output: number of lines
*/
int count_lines(const vector<string>& files);

/*
count how many lines in current directory
input: no, may require to change inside the function body
output: number of lines
*/
int count_lines();


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
		for (int j = 0; j < n - 1 - i; ++j)
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
	vector<int> tmp(m + 1, 0);
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
	int l = 2 * i, r = (2 * i) + 1, lar;

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



/*
* Merge sort: using Divide and Conquer
* merge_sort(vector<T>& a, low,  high)
* If low > high
*    1. Find the middle point to divide the array into two halves:
*            middle mid = (low+high)/2
*    2. Call merge_sort for first half [low..mid]:
*            Call merge_sort(a, low, mid)
*    3. Call merge_sort for second half [mid+1..high]:
*            Call merge_sort(arr, m+1, r)
*    4. Merge the two halves sorted in step 2 and 3:
*            Call merge(a, low, high, mid)
* Auxiliary Space: O(n), Complexity: O(nlog(n))
*/
template<class T>
void merge(vector<T>& a, int low, int high, int mid) {
	vector<T> c(a);
	int i = low, j = mid + 1, k = low;
	while (i <= mid && j <= high) {
		if (a[i] < a[j]) c[k++] = a[i++];
		else c[k++] = a[j++];
	}
	a = c;
}

template <class T>
void merge_sort(vector<T>& a, int low, int high) {
	if (low < high) {
		int mid = (low + high) / 2;
		merge_sort(a, low, mid);
		merge_sort(a, mid + 1, high);
		merge(a, low, high, mid);
	}
	return;
}

template <class T>
void merge_sort(vector<T>& a) {
	int n = a.size();
	merge_sort(a, 0, n - 1);
}

/*
* Reservoir Sampling:
* randomly choosing k samples from a list of n items,
* where n is either a very large or unknown number.
* 1. creat a vector of size k, tmp, from a[0..k-1]
* 2. one by one consider all items from [k..n-1]
*  2a. generate a random number j from 0 to i
*  2b. if j is in range 0 to k-1, replace tmp[j] with a[i]
* Complexity: O(n), Space: O(k)
*/
template<class T>
vector<T> reservoir_sampling(const vector<T>& a, int k = 1) {
	vector<T> tmp(a.begin(), a.begin() + k);
	int n = a.size();
	srand((unsigned int)time(NULL));
	for (int i = 0; i < n; ++i) {
		int j = rand() % (i + 1);
		if (j < k) tmp[j] = a[i];
	}
	return tmp;
}




// Section: My Math
// calculate the derivative via f'(x) = (f(x + h/2) - f(x - h/2))/h

class Differentiation {
public:
	// error analysis, see url: math.umd.edu/~dlevy/classes/amsc466/lecture-notes/differentiation-chap.pdf	
	static double leftDiff(const std::function<double(double)> &f, double x0 = 0., double eps = 0.001);
	static double rightDiff(const std::function<double(double)> &f, double x0 = 0., double eps = 0.001);
	static double centDiff(const std::function<double(double)> &f, double x0 = 0., double eps = 0.001);
	static double centDiff2(const std::function<double(double)> &f, double x0 = 0., double eps = 0.001);
};

class Integral {
public:
	/*
	dx=(b-a)/N, Ki = max value of nth prime of function f over [a, b]
	Computation: O(N) for all
	Left and Right Rect: K1*o(dx^2)
	Mid Rect, Trapezoid: K2*o(dx^3)
	Simpson: K4*o(dx^5)
	see url: kenyon.edu/Depts/Math/Paquin/Error.pdf
	*/		
	static double leftRect(const std::function<double(double)> &f, double a = 0.0, double b = 1.0, int n = 100);
	static double rightRect(const std::function<double(double)> &f, double a = 0.0, double b = 1.0, int n = 100);
	static double midRect(const std::function<double(double)> &f, double a = 0.0, double b = 1.0, int n = 100);
	static double trapezium(const std::function<double(double)> &f, double a = 0.0, double b = 1.0, int n = 100);
	static double simpson(const std::function<double(double)> &f, double a = 0.0, double b = 1.0, int n = 100);	
};

class Fibonacci {
public:
	static int statArray(int n);
};

unsigned int fact(unsigned int n);

namespace mymath {
	double f1(double x);

	double f2(double x);

	class Math {
	public:
		/*
		virtual Math& operator*(Math& rv) = 0;
		virtual Math& multiply(Matrix*) = 0;
		virtual Math& multiply(Vector*) = 0;
		virtual Math& multiply(Scalar*) = 0;
		*/
	};

	class Matrix : public Math {
		int a;
	public:
		Matrix(int dat);
		int val() const;
		/*
		Math& operator* (Math& rv);
		Math& multiply(Matrix*);
		Math& multiply(Vector*);
		Math& multiply(Scalar*);
		*/
	};

	class Vector : public Math {
	public:
		/*
		Math& operator* (Math& rv);
		Math& multiply(Matrix*);
		Math& multiply(Vector*);
		Math& multiply(Scalar*);
		*/
	};

	class Scalar : public Math {
	public:
		/*
		Math& operator* (Math& rv);
		Math& multiply(Matrix*);
		Math& multiply(Vector*);
		Math& multiply(Scalar*);
		*/
	};
}

// Section: Test
void hello();
void goodbye();
typedef void(*FunctionPointer0d)();

double add(double a, double b);
double multiply(double a, double b);
double substract(double a, double b);
double dividedby(double a, double b);
typedef double(*FunctionPointer2d)(double, double);

double testFunctionPointer(FunctionPointer0d& fpv, FunctionPointer2d& fp2, double a = 10., double b = 5.);
void test_upperbound(const vector<int>& vec, int num);

enum ShapeType { circle, square, rectangle };
enum BeverageType { water = 10, coca = 25, pepsi, juice = 50 };

void testEnum();

class Employee {
public:
	Employee(string theName, float thePayRate);
	string getName() const;
	void setName(string theName);
	float getPayRate() const;
	void setPayRate(float thePayRate);
	float pay(float hoursWorked) const;
protected:
	string name;
	float payRate;
};

class Manager : public Employee {
public:
	Manager(string theName, float thePayRate, bool isSalaried);
	bool getSalaried() const;
	void setSalaried(bool isSalaried);
	float pay(float hoursWorked) const;
protected:
	bool salaried;
};

class Supervisor : public Manager {
public:
	Supervisor(string theName, float thePayRate, string theDept);
	string getDept() const;
	void setDept(string theDept);
protected:
	string dept;
};

// calculate dynamic card game, red +1, black -1.
// how much to pay for a game? assume risk-neutral
double deck_card();

/*
world series: Green Book, Dynamic Programming, Page 123
*/
double world_series(int n, double payoff);

// Section: Design Patterns
class PayOff {
public:
	PayOff() {}
	virtual ~PayOff() {}
	virtual PayOff* clone() const = 0;
	virtual double operator()(double Spot) const = 0;
};

class PayOffCall : public PayOff {
	double Strike;
public:
	PayOffCall(double Strike_);
	virtual ~PayOffCall() {}
	virtual PayOff* clone() const;
	virtual double operator()(double Spot) const;
};

class PayOffPut : public PayOff {
	double Strike;
public:
	PayOffPut(double Strike_);
	virtual ~PayOffPut() {}
	virtual PayOff* clone() const;
	virtual double operator()(double Spot) const;
};

class PayOffFactory {
public:
	typedef PayOff* (*CreatePayOffFunction)(double);
	// refer to pointer to functions which take in a double and returns PayOff*
	static PayOffFactory& Instance();
	~PayOffFactory() {}
	void RegisterPayOff(string, CreatePayOffFunction);
	PayOff* CreatePayOff(string PayOffId, double Strike);
private:
	map<string, CreatePayOffFunction> TheCreatorFunctions;
	// STL map class to associate objects with string identifiers
	PayOffFactory() {}
	PayOffFactory(const PayOffFactory&) {}
	PayOffFactory& operator=(const PayOffFactory&) { return *this; }
};

template<class T>
class PayOffHelper{
public:
	static PayOff* Create(double);
	PayOffHelper(string);
};

template<class T>
PayOff* PayOffHelper<T>::Create(double Strike) { 
	return new T(Strike);
}

template<class T>
PayOffHelper<T>::PayOffHelper(string id) {
	PayOffFactory& thePayOffFactory = PayOffFactory::Instance();
	thePayOffFactory.RegisterPayOff(id, PayOffHelper<T>::Create);
}

template<class T>
class Wrapper {
	T* DataPtr;
public:
	Wrapper() { DataPtr = 0; }
	Wrapper(T* inner) { DataPtr = inner; }
	Wrapper(const T& inner) { DataPtr = inner.clone(); }
	Wrapper(const Wrapper<T>& wp);
	Wrapper& operator=(const Wrapper<T>& wp);
	~Wrapper() { if (DataPtr != 0) delete DataPtr; SEP; }
	T& operator*() { return *DataPtr; }
	const T& operator*() const { return *DataPtr; }
	T* operator->() { return DataPtr; }
	const T* operator->() const { return DataPtr; }
};

template<class T>
Wrapper<T>::Wrapper(const Wrapper<T>& wp) {	
	DataPtr = (wp.DataPtr != 0) ? wp.DataPtr->clone() : 0;
}

template<class T>
Wrapper<T>& Wrapper<T>::operator=(const Wrapper<T>& wp) {
	if (this == &wp) return *this;	
	if (DataPtr != 0) delete DataPtr;
	DataPtr = (wp.DataPtr != 0) ? wp.DataPtr->clone() : 0;
	return *this;
}

class PayOffBridge {
	PayOff* ThePayOffPtr;
public:
	PayOffBridge(const PayOffBridge& original);
	PayOffBridge(const PayOff& innerPayOff);
	PayOffBridge& operator=(const PayOffBridge& original);
	~PayOffBridge();
	inline double operator()(double Spot) const;
};

inline double PayOffBridge::operator()(double Spot) const {
	//return ThePayOffPtr->operator()(Spot);
	return (*ThePayOffPtr)(Spot);
}

class VanillaOption {
	double Expiry;
	PayOffBridge ThePayOff;
public:
	VanillaOption(const PayOffBridge& ThePayOff_, double Expiry_);
	double GetExpiry() const;
	double OptionPayOff(double Spot) const;
};

class ParametersInner {
public:
	ParametersInner() {}
	virtual ~ParametersInner() {}
	virtual ParametersInner* clone() const = 0;
	virtual double Integral(double time1, double time2) const = 0;
	virtual double IntegralSquare(double time1, double time2) const = 0;
};

class ParametersConstant : public ParametersInner {
	double Constant;
	double ConstantSquare;
public:
	ParametersConstant(double constant);
	virtual ParametersInner* clone() const;
	virtual double Integral(double time1, double time2) const;
	virtual double IntegralSquare(double time1, double time2) const;	
};

class Parameters{
	ParametersInner* InnerObjectPtr;
public:
	Parameters(const ParametersInner& innerObject);
	Parameters(const Parameters& original);
	Parameters& operator= (const Parameters& original);
	virtual ~Parameters();
	inline double Integral(double time1, double time2) const;
	inline double IntegralSquare(double time1, double time2) const;
	double Mean(double time1, double time2) const;
	double RootMeanSquare(double time1, double time2) const;	
};

inline double Parameters::Integral(double time1, double time2) const {
	return InnerObjectPtr->Integral(time1, time2);
}

inline double Parameters::IntegralSquare(double time1, double time2) const {
	return InnerObjectPtr->IntegralSquare(time1, time2);
}

class StatisticsMC {
public:
	StatisticsMC() {}
	virtual ~StatisticsMC() {}
	virtual StatisticsMC* clone() const = 0;
	virtual void DumpOneResult(double result) = 0;
	virtual vector<vector<double> > GetResultsSoFar() const = 0;
};

class StatisticsMean : public StatisticsMC {
public:
	StatisticsMean();
	virtual StatisticsMC* clone() const;
	virtual void DumpOneResult(double result);
	virtual vector<vector<double> > GetResultsSoFar() const;
private:
	double RunningSum;
	unsigned long PathsDone;
};

class ConvergenceTable : public StatisticsMC {
public:
	ConvergenceTable(const Wrapper<StatisticsMC>& Inner_);
	virtual StatisticsMC* clone() const;
	virtual void DumpOneResult(double result);
	virtual vector<vector<double> > GetResultsSoFar() const;
private:
	Wrapper<StatisticsMC> Inner;
	vector<vector<double> > ResultsSoFar;
	unsigned long StoppingPoint;
	unsigned long PathsDone;
};

void SimpleMonteCarlo(
	const VanillaOption &TheOption,
	double Spot,
	const Parameters& Vol,
	const Parameters& r,
	unsigned long NumberOfPaths,
	StatisticsMC& gatherer);

#endif
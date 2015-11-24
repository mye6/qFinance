#ifndef SOLVER_H
#define SOLVER_H

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <functional>
#include <cassert>
#include <map>
#include <cmath>
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

// Data Structure and Algorithm Analysis
// dsaa
namespace dsaa{
	class IntCell {
		int storedValue;
	public:
		explicit IntCell(int dat = 0);
		explicit IntCell(const IntCell& rhs);
		IntCell& operator=(const IntCell& rhs);
		int read() const;
		void write(int x);
	};

	class IntPtCell {
		int* storedValue;
	public:
		explicit IntPtCell(int dat = 0);
		explicit IntPtCell(const IntPtCell& rhs);
		IntPtCell& operator=(const IntPtCell& rhs);
		~IntPtCell();
		int read() const;
		void write(int x);
	};

	template<typename T>
	const T& findMax(const vector<T>& vec) {
		cout << "const" << endl;
		int m = 0;
		for (size_t i = 0; i < vec.size(); ++i) {
			if (vec[m] < vec[i]) m = i;
		}
		return vec[m];
	}

	template<typename T>
	T& findMax(vector<T>& vec) {
		cout << "non const" << endl;
		int m = 0;
		for (size_t i = 0; i < vec.size(); ++i) {
			if (vec[m] < vec[i]) m = i;
		}
		return vec[m];
	}

	class Employee{
		string name;
		double salary;
	public:
		void setValue(const string& n, double s);
		const string& getName() const;
		void print(ostream& out) const;
		bool operator<(const Employee& rhs) const;		
	};
	// define an output operator for Employee
	ostream& operator<<(ostream& out, const Employee& rhs);

	// Vector
	template<class T>
	class Vector {
		enum {SPARE_CAPACITY = 16};
		int size_, capacity_;
		T* objs;		
	public:
		explicit Vector(int sz = 0) : size_(sz), capacity(sz + SPARE_CAPACITY), objs(new T[capacity_]) {
			cout << "explicit Vector(int sz = 0)" << endl;
		}
		explicit Vector(int sz, const T& t) : size_(sz), capacity_(sz + SPARE_CAPACITY), objs(new T[capacity_]) {
			cout << "explicit Vector(int sz, const T& t)" << endl;
			for (int i = 0; i < size_; ++i)
				objs[i] = t;
		}
		~Vector() { 
			cout << "~Vector()" << endl;
			delete[] objs;
		}
		Vector(const Vector& rhs) {
			cout << "Vector(const Vector& rhs)" << endl;
			operator=(rhs);
		}
		const Vector& operator=(const Vector& rhs) {
			cout << "const Vector& operator=(const Vector& rhs)" << endl;
			if (this == &rhs) return *this;
			delete[] objs;
			size_ = rhs.size_;
			capacity_ = rhs.capacity_;
			objs = new T[capacity_];
			for (int i = 0; i < size_; ++i) objs[i] = rhs.objs[i];
			return *this;
		}
		void reserve(int newCapacity) {
			cout << "reserve(the newCapacity)" << endl;
			if (newCapacity < size_) return;
			T* oldObjs = objs;
			objs = new T[newCapacity];
			for (int i = 0; i < size_; ++i)
				objs[i] = oldObjs[i];
			capacity_ = newCapacity;
			delete[] oldObjs;			
		}
		void resize(int newSize) {
			cout << "resize(int newSize)" << endl;
			if (newSize > capacity_) reserve(newSize * 2 + 1);
			size_ = newSize;
		}
		T& operator[](int index) { return objs[index]; }
		const T& operator[](int index) const { return objs[index]; }
		int size() const { return size_; }
		int capacity() const { return capacity_; }
		
		bool empty() const { return size_ == 0; }
		void push_back(const T& x) {
			if (size_ == capacity_) reserve(2 * capacity_ + 1);
			objs[size_++] = x;
		}
		void pop_back() { size_--; }
		const T& back() const { return objs[size_ - 1]; }
		typedef T* iterator;
		typedef const T* const_iterator;
		iterator begin() { return &objs[0]; }
		const_iterator begin() const { return &objs[0]; }
		iterator end() { return &objs[size_]; }
		const_iterator end() const { return &objs[size_]; }		
		void print(ostream& os) const;
		/*
		void print(ostream& os) const {
			for (Vector<T>::const_iterator it = begin(); it < end(); it++)
				os << *it << " ";
			os << endl;
		}
		*/
		friend ostream& operator<<(ostream& os, const Vector& vec) {
			vec.print(os);
			return os;
		}
	};
	template<class T>
	void Vector<T>::print(ostream& os) const {
		for (Vector<T>::const_iterator it = begin(); it < end(); it++)
			os << *it << " ";
		os << endl;
	}

	template<class Object>
	class List {
		struct Node{
			Object data;
			Node* prev;
			Node* next;
			Node(const Object& d = Object(), Node* p = NULL, Node* n = NULL) : data(d), prev(p), next(n) {}
		};
	public:
		
		class const_iterator {
		public:
			const_iterator() : current(NULL) {}
			const Object& operator*() const { return retrieve(); }
			const_iterator& operator++() {
				current = current->next;
				return *this;
			}
			const_iterator operator++(int) {
				const_iterator old = *this; ++(*this); return old;
			}
			const_iterator& operator--() {
				current = current->prev;
				return *this;
			}
			const_iterator operator--(int) {
				const_iterator old = *this; --(*this); return old;
			}

			bool operator==(const const_iterator& rhs) const {
				return current == rhs.current;
			}
			bool operator!=(const const_iterator& rhs) const {
				return !(*this == rhs);
			}
		protected:
			Node* current;
			Object& retrieve() const { return current->data; }
			const_iterator(Node* p) : current(p) {}
			friend class List<Object> ;
		};

		class iterator : public const_iterator {
			iterator() {}
			Object& operator*() { return retrieve(); }
			const Object& operator*() const { return const_iterator::operator*(); }
			iterator& operator++() {
				current = current->next; return *this;
			}
			iterator operator++(int) {
				iterator old = *this; ++(*this); return old;
			}
			iterator& operator--() {
				current = current->prev; return *this;
			}
			iterator operator--(int) {
				iterator old = *this; --(*this); return old;
			}

		protected:
			iterator(Node* p) : const_iterator(p) {}
			friend class List<Object>;
		};
		
	public:
		List() { init(); }
		List(const List& rhs) { init(); *this = rhs; }

		const List& operator=(const List& rhs) {
			if (this == &rhs) return *this;
			clear();
			for (const_iterator itr = rhs.begin(); itr != rhs.end(); ++itr)
				push_back(*itr);
			return *this;
		}

		~List() {			
			clear();
			delete head;
			delete tail;
		}

		void push_front(const Object& x) { insert(begin(), x); }
		void push_back(const Object& x) { insert(end(), x); }
		iterator begin() {
			return iterator(head->next);
		}
		const_iterator begin() const {
			return const_iterator(head->next);
		}
		iterator end() {
			return iterator(tail);
		}
		const_iterator end() const {
			return const_iterator(tail);
		}
		iterator insert(iterator itr, const Object& x) {
			Node* p = itr.current;
			theSize++;
			return iterator(p->prev = p->prev->next = new Node(x, p->prev, p));
		}
		int size() const { return theSize; }
		bool empty() const { return size() == 0; }
		Object& front() { return *begin(); }
		const Object& front() const { return *begin(); }
		Object& back() { return *--end(); }
		const Object& back() const { return *--end(); }
		void pop_front() { erase(begin()); }
		void pop_back() { erase(--end()); }

		void clear() {
			while (!empty())
				pop_front();
		}

		iterator erase(iterator itr) {			
			Node* p = itr.current;
			iterator retVal(p->next);
			p->prev->next = p->next;
			p->next->prev = p->prev;
			delete p;
			theSize--;
			return retVal;
		}

		iterator erase(iterator start, iterator end) {
			for (iterator itr = start; itr != end;)
				itr = erase(itr);
			return end;
		}

		void print(ostream& os) const {
			for (List::const_iterator it = begin(); it != end(); it++)
				os << *it << " ";			
		}
		friend ostream& operator<<(ostream& os, const List& ls) {
			ls.print(os);
			return os;
		}
		
	private:
		int theSize;
		Node* head;
		Node* tail;
		void init() {			
			theSize = 0;
			head = new Node;
			tail = new Node;
			head->next = tail;
			tail->prev = head;
		}
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

enum ShapeType { circle, square, rectangle };
enum BeverageType { water = 10, coca = 25, pepsi, juice = 50 };

void testEnum();

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
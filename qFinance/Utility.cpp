#include "Solver.h"

int sgn(double x) {
	if (x >= 0.0) return 1;
	return -1;
}

/*
vec1d: 1 2 3 4 5 6
vec2d = reshape_vec1d(vec1d, 2);
output vec2d:
1 2 3
4 5 6
*/
vector<vector<double> > reshape_vec1d(const vector<double>& vec, int nrow) {
	int ncol = (int)vec.size() / nrow;
	vector<vector<double> > res;
	for (int i = 0; i < nrow; ++i) {
		res.push_back(sub_vec(vec, i*ncol, ncol));
	}
	return res;
}

/*
vec2d:
1 2 3
4 5 6
7 8 9
vec1d = flatten_vec2d(vec2d)
output vec1d:
1 2 3 4 5 6 7 8 9
*/
vector<double> flatten_vec2d(const vector<vector<double> >& dat) {
	int m = dat.size();
	vector<double> res(begin(dat[0]), end(dat[0]));
	for (int i = 1; i < m; ++i)
		res.insert(end(res), begin(dat[i]), end(dat[i]));
	return res;
}

void print_map(const map<string, vector<double>>& mp, size_t sz) {
	// print a map: string -> vector<double>
	if (sz > mp.size()) sz = mp.size();
	typedef map<string, vector<double>>::const_iterator MapIterator;
	for (MapIterator iter = mp.begin(); iter != mp.end(); iter++) {
		cout << iter->first << "\t|\t";
		print_vec<double>(iter->second, ", ");
	}
}

vector<double> sub_vec(const vector<double>& vec, int first, int len) {
	// obtain the sub vector with length len starting from the first element
	vector<double>::const_iterator beg = vec.begin() + first;
	vector<double>::const_iterator end = beg + len;
	vector<double> tmp_vec(beg, end);
	return tmp_vec;
}

double power_rec(double x, int n) {
	// obtain the power function through recursion
	// use memorization to cache intermediate results (static vector) to achieve higher efficiency
	// static vector<double> cache(n, -1.); but base value x will not be changed for other computation
	if (n < 0) return 1.0 / power_rec(x, -n);
	if (n == 0)  return 1;
	double v = power_rec(x, n / 2);
	if (n % 2 == 0) return v * v;
	else return v * v * x;	
}

double exp_taylor(double x) {
	// obtain the exp function by multiplication/addition through Taylor expansion
	if (x < 0.) return 1.0 / exp_taylor(-x);
	double x_over_n = x;
	double n = 1.;
	if (log2(x) > 1.) {
		n = power_rec(2., (int)floor(log2(x)));
		x_over_n = x / n;
	}
	double res = 1., taylor_term = x_over_n, denom = 1.;
	while (taylor_term > numeric_limits<double>::min()) {
		res += taylor_term;
		taylor_term *= x_over_n / (++denom);
	}
	return power_rec(res, (int)n);
}

/*
* Prime number:
* 1. construct status[1..N] to record whether each element is a prime
* 2. for i in 2 to sqrt(N), get rid of all i*j<=N (j starts with 2)
* 3. output all numbers with status[i] as true, i.e. prime numbers
* Complexity: O(n)
*/
vector<int> prime_vec(int N) {
	vector<bool> status(N + 1, true);
	// status to record whether each element is a prime number
	status[1] = false;
	for (int i = 2; i <= sqrt(N); ++i) {
		if (status[i]) {
			for (int j = 2; i*j <= N; ++j) status[i*j] = false;
		}
	}
	vector<int> res;
	for (int i = 2; i <= N; ++i) {
		if (status[i]) res.push_back(i);
	}
	return res;
}

/*
convert a 0-255 unsigned char to 8-digit-binary string
val: 255
output: 11111111
*/
string toBinary(const unsigned char val) {
	stringstream s;
	for (int i = 7; i >= 0; --i) {
		if (val & (1 << i)) s << "1";
		else s << "0";
	}
	return s.str();
}

// print out the converted 8-digit-binary string
void printBinary(const unsigned char val) {	
	PRINT(toBinary(val));
}

/*
* Fisher–Yates shuffle Algorithm
* O(n) time complexity
* given a function rand()
* The idea is to start from the last element,
* swap it with a randomly selected element from the whole array (including last)
*/
void swap(int& x, int& y) {
	// pass by reference
	int tmp = x;
	x = y;
	y = tmp;
}

void randomize(vector<int>& a) {
	if (a.size() == 0 || a.size() == 1) return;
	int n = a.size();
	srand((unsigned int)time(NULL));
	// Use a different seed value so that we don't get same
	// result each time we run this program
	for (int i = n - 1; i > 0; --i) {
		int j = rand() % (i + 1); // Pick a random index from 0 to i
		swap(a[i], a[j]);
	}
}

/*
* XOR swap uses XOR bitwise operation
* A^B = B^A
* (A^B)^C = A^(B^C)
* A^0 = A
* A^A = 0
*/
void swap_xor(int& x, int& y) {
	if (x != y) {
		x ^= y;
		y ^= x;
		x ^= y;
	}
}

/* generate a random number within [0..n] */
int randi(int n) {
	//srand((unsigned int)time(NULL));
	return rand() % (n + 1);
}

/*
* For example for 4 ( 100) and 16(10000), we get following after subtracting 1
* 3 –> 011
* 15 –> 01111
* !(x&(x - 1))
* x&(x-1) will get an integer, whose digits are determined by x
*
*/
bool isPowerOfTwo(int x) {
	// return x&(x - 1) == 0; // this does not work as == will be called first
	// return (x&(x - 1)) == 0; // works for positive integers (>0)
	return x && (!(x&(x - 1))); // works for non-negative integers (>=0)
}

/*
 count how many lines in a file
 input: filename
 output: number of lines
*/
int count_lines(const string& filename) {
	int res = 0;
	ifstream in(filename);
	string line;
	while (getline(in, line)) ++res;
	in.close();
	return res;
}

/*
 count how many lines in a number of files
 input: vector of filenames
 output: number of lines
*/
int count_lines(const vector<string>& files) {
	int total = 0;
	for (size_t i = 0; i < files.size(); ++i) {
		total += count_lines(files[i]);
	}
	return total;
}

/*
count how many lines in current directory
input: no, may require to change inside the function body
output: number of lines
*/
int count_lines() {
	vector<string> files{
		"Main.cpp", "Code_Backup.cpp", "Leetcode.h", "Leetcode.cpp",
		"Solver.h", "Puzzle.h", "Puzzle.cpp", "Random.cpp", "QuadraticProbing.h",
		"SeparateChaining.h", "Sort.h", "LinkedList.h", "BinarySearchTree.h",
		"QueueAr.h", "StackLi.h", "StackAr.h", "Matrix.h", "String.h", "String.cpp",
		"Vector.h", "dsexceptions.h", "Test.cpp", "Utility.cpp", "DesignPatterns.cpp",
		"MyMath.cpp", "EigenIO.cpp", "Finance.cpp", "Finance.h", "Processor.cpp",
		"Data.cpp"
	};
	return count_lines(files);
}
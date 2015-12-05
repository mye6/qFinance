#include "Solver.h"

// args: void; return: void
// typedef void(*FunctionPointer0d)(); // has been defined in Solver.h
void hello() { cout << "hello" << endl; }
void goodbye() { cout << "goodbye" << endl; }

// args: double, double; return: double
// typedef double(*FunctionPointer2d)(double, double); // has been defined in Solver.h
double add(double a, double b) { return a + b; }
double multiply(double a, double b) { return a * b; }
double substract(double a, double b) { return a - b; }
double dividedby(double a, double b) { return a / b; }

double testFunctionPointer(FunctionPointer0d& fpv, FunctionPointer2d& fp2, double a, double b) {
	(*fpv)();
	return (*fp2)(a, b);
}

void test_upperbound(const vector<int>& vec, int num) {
	vector<int> v(vec);
	sort(v.begin(), v.end());
	vector<int>::iterator up;
	up = std::upper_bound(v.begin(), v.end(), 20);
	cout << "upper_bound at position " << (up - v.begin()) << '\n';
}

void testEnum() {
	ShapeType shape = square;
	PRINT(shape);
	BeverageType bev = pepsi;
	PRINT(bev);
	PRINT(shape / 2);
	PRINT((double)shape / 2);
	PRINT(bev / 2);
	PRINT((double)bev / 2);


	switch (square) {
	case circle: PRINT("pi r square"); break;
	case square: PRINT("a square"); break;
	case rectangle: PRINT("a b"); break;
	default: break;
	}

	switch (bev) {
	case water: PRINT("healthy"); break;
	case coca: PRINT("with piza"); break;
	case pepsi: PRINT("better than coca?"); break;
	case juice: PRINT("lots of sugar"); break;
	default: break;
	}
}

Employee::Employee(string theName, float thePayRate)
	: name(theName), payRate(thePayRate) {}
string Employee::getName() const { return name; }
void Employee::setName(string theName) { name = theName; }
float Employee::getPayRate() const { return payRate; }
void Employee::setPayRate(float thePayRate) { payRate = thePayRate; }
float Employee::pay(float hoursWorked) const { return hoursWorked * payRate; }

Manager::Manager(string theName, float thePayRate, bool isSalaried)
	: Employee(theName, thePayRate), salaried(isSalaried) {}
bool Manager::getSalaried() const { return salaried; }
void Manager::setSalaried(bool isSalaried) { salaried = isSalaried; }
float Manager::pay(float hoursWorked) const {
	if (salaried) return payRate;
	return Employee::pay(hoursWorked);
}

Supervisor::Supervisor(string theName, float thePayRate, string theDept)
	: Manager(theName, thePayRate, true), dept(theDept) {}
string Supervisor::getDept() const { return dept; }
void Supervisor::setDept(string theDept) { dept = theDept; }

double deck_card() {
	const int N = 26;
	vector<vector<double> > D(N + 1, vector<double>(N + 1, (double)(INT_MIN)));
	// row: # red cards, column: # black cards
	D[0][0] = 0.;
	for (int r = 1; r <= N; ++r) D[r][0] = 0.;
	for (int b = 1; b <= N; ++b) D[0][b] = (double)b;

	for (int r = 1; r <= N; ++r)
		for (int b = 1; b <= N; ++b)
			D[r][b] = max((double)(b - r), D[r - 1][b] * r / (r + b) + D[r][b - 1] * b / (r + b));
	return D[N][N];
}

/*
world series: Green Book, Dynamic Programming, Page 123
*/
double world_series(int n, double payoff) {
	struct Node {
		double money;
		double bet;
	};
	vector<vector<Node> > mat(n + 1, vector<Node>(n + 1));
	for (int j = 0; j <= n - 1; ++j) {
		mat[n][j].money = payoff;
		mat[n][j].bet = 0;
	}
	for (int i = 0; i <= n - 1; ++i) {
		mat[i][n].money = -payoff;
		mat[i][n].bet = 0;
	}
	for (int i = n - 1; i >= 0; --i) {
		for (int j = n - 1; j >= 0; --j) {
			mat[i][j].money = 0.5 * (mat[i + 1][j].money + mat[i][j + 1].money);
			mat[i][j].bet = 0.5 * (mat[i + 1][j].money - mat[i][j + 1].money);
		}
	}
	return mat[0][0].bet;
}
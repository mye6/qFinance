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
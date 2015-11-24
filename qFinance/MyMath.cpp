#include "Solver.h"

double Differentiation::leftDiff(const std::function<double(double)> &f, double x0, double eps) {
	double lo = x0 - eps, hi = x0;
	return (f(hi) - f(lo)) / eps;
}

double Differentiation::rightDiff(const std::function<double(double)> &f, double x0, double eps) {
	double lo = x0, hi = x0 + eps;
	return (f(hi) - f(lo)) / eps;
}

double Differentiation::centDiff(const std::function<double(double)> &f, double x0, double eps) {
	double eps2 = eps / 2, lo = x0 - eps2, hi = x0 + eps2;
	return (f(hi) - f(lo)) / eps;
}

double Differentiation::centDiff2(const std::function<double(double)> &f, double x0, double eps) {
	double lo = x0 - eps, hi = x0 + eps;
	return (f(hi) + f(lo) - 2*f(x0)) / eps / eps;
}

double Integral::leftRect(const std::function<double(double)> &f, double a, double b, int n) {
	// leftRect method, see url: rosettacode.org/wiki/Numerical_integration
	double dx = (b - a) / n, sum = 0.;
	for (int i = 0; i < n; ++i, a+=dx)
		sum += f(a);	
	return dx*sum;
}

double Integral::rightRect(const std::function<double(double)> &f, double a, double b, int n) {
	// rightRect method, see url: rosettacode.org/wiki/Numerical_integration
	double dx = (b - a) / n, sum = 0.;
	a += dx;
	for (int i = 0; i < n; ++i, a += dx)
		sum += f(a);	
	return dx*sum;
}

double Integral::midRect(const std::function<double(double)> &f, double a, double b, int n) {
	// midRect method, see url: rosettacode.org/wiki/Numerical_integration
	double dx = (b - a) / n, sum = 0.;
	a += .5*dx;
	for (int i = 0; i < n; ++i, a+=dx)		
		sum += f(a);	
	return dx*sum;
}

double Integral::trapezium(const std::function<double(double)> &f, double a, double b, int n) {
	// trapezium method, see url: rosettacode.org/wiki/Numerical_integration
	double dx = (b - a) / n, sum = f(a);
	a += dx;
	for (int i = 1; i < n; ++i, a+=dx)
		sum += f(a)*2.;
	sum += f(b);
	return 0.5*dx*sum;
}

double Integral::simpson(const std::function<double(double)> &f, double a, double b, int n) {
	// simpson method, see url: rosettacode.org/wiki/Numerical_integration	
	double dx = (b - a) / n, sum1 = f(a+dx/2), sum2 = 0.0;
	for (int i = 1; i < n; ++i) {
		sum1 += f(a + dx*i + dx / 2);
		sum2 += f(a + dx*i);
	}	
	return (dx / 6) * (f(a) + f(b) + 4 * sum1 + 2 * sum2);
}

// fibonacci
int Fibonacci::statArray(int n) {
	const int sz = 100;
	static int f[sz];
	f[0] = f[1] = 1;
	int i;
	for (i = 0; i < sz; i++) {
		if (f[i] == 0) break;
	}
	while (i <= n) {
		f[i] = f[i - 1] + f[i - 2];
		i++;
	}
	return f[n];
}

namespace mymath {
	double f1(double x) {
		// return (2 * x + 1.);
		return (x * x + 1.);
	}

	double f2(double x) {
		return (x * x * x / 3 + x + 2.);
	}

	Matrix::Matrix(int dat) : a(dat) {}
	int Matrix::val() const { return a; }
	/*
	class Matrix : public Math {
	int a;
	public:
	Matrix(int a);
	int val() const;
	Math& operator* (Math& rv);
	Math& multiply(Matrix*);
	Math& multiply(Vector*);
	Math& multiply(Scalar*);
	};

	class Matrix : public Math {
	public:
	Math& operator* (Math& rv);
	Math& multiply(Matrix*);
	Math& multiply(Vector*);
	Math& multiply(Scalar*);
	};

	class Vector : public Math {
	public:
	Math& operator* (Math& rv);
	Math& multiply(Matrix*);
	Math& multiply(Vector*);
	Math& multiply(Scalar*);
	};

	class Scalar : public Math {
	public:
	Math& operator* (Math& rv);
	Math& multiply(Matrix*);
	Math& multiply(Vector*);
	Math& multiply(Scalar*);
	};
	*/
}
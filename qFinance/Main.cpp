#include "Solver.h"
#include "Finance.h"

int main() {
	double S = 100., K = 100., r = 0.1, sigma = 0.25, time = 1.0;
	int N = 400;
	PRINT(option_price_call_european_binomial(S, K, r, sigma, time, N));
	PRINT(option_price_put_european_binomial(S, K, r, sigma, time, N));
	PRINT(option_price_call_american_binomial(S, K, r, sigma, time, N));
	PRINT(option_price_put_american_binomial(S, K, r, sigma, time, N));
	PRINT(option_price_call_american_binomial(S, K, r, sigma, time, N));
	PRINT(option_price_put_american_binomial(S, K, r, sigma, time, N));
	PRINT(option_price_put_european_finite_diff_explicit(S, K, r, sigma, time, N, N));


	system("pause");
	return 0;
}
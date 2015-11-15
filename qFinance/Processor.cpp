#include "Solver.h"

bool avg_lower(const vector<float>& dat, int M, float P) {
	// compare average value of the M smallest elements in vector dat is <= P
	// if avg <= P, return true; otherwise return false
	vector<float> prices(dat);
	make_heap(prices.begin(), prices.end(), greater<float>());	
	float sum = 0.;
	vector<float> vec;
	for (int i = 0; i < M; ++i) {
		pop_heap(prices.begin(), prices.end(), greater<float>());
		sum += prices.back();
		vec.push_back(prices.back());
		prices.pop_back();
	}
	if (sum <= M*P) {		
		return true;
	}
	else return false;
}
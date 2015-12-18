#include "Solver.h"
#include "Puzzle.h"

double angle_hour(int hour, int minute) {
	return 30. * ((hour % 12) + minute / 60.);
}

double angle_minute(int minute) {
	return 6. * minute;
}



void print(ostream& os, const HM& hm) {
	string s = to_string(hm.hour) + "/" + to_string(hm.minute);
	if (hm.hour < 10) s.insert(0, 1, '0');
	if (hm.minute < 10) s.insert(3, 1, '0');
	os << s << endl;
	cout << angle_hour(hm.hour, hm.minute) << " | " << angle_minute(hm.minute) << endl;
}

void hm_cross(const string& filename, int hour_end) {
	ofstream out(filename);
	vector<HM> res;

	for (int hour = 0; hour <= hour_end; ++hour) {
		for (int minute = 0; minute <= 59; ++minute) {
			double cur = angle_minute(minute) - angle_hour(hour, minute);
			double next = angle_minute(minute + 1) - angle_hour(hour, minute + 1);
			if (cur <= 0. && next > 0.) {
				res.push_back(HM(hour, minute));
			}
		}
	}

	for (size_t i = 0; i < res.size(); ++i) {
		print(out, res[i]);
	}

}

int birthday() {
	int n = 1;
	double res = 1.0;
	while (res >= 0.5) {
		++n;
		res *= (366. - n) / 365.;
	}
	return n;
}

/*
input: number of points
output: 95% confidence interval, i.e. Lower and Upper bound
Xi = 1 (if u1^2 + u2^2 < 1); = 0 otherwise
E(X) = pi/4, pi_hat = 4*X_bar
X_bar ~ N(mu, sigma^2/n)
p_hat = X_bar
mu = p_hat (probability, should be pi/4)
sigma^2 = p_hat(1-p_hat)
pi_bar ~ N(4*p_hat, 16*p_hat*(1-p_hat/n)
*/
vector<double> pi_estimator(int n) {
	double xbar = 0., u1, u2;
	for (int i = 0; i < n; ++i) {
		u1 = unif(); u2 = unif();
		if (u1*u1 + u2*u2 < 1.) ++xbar;
	}
	xbar /= (double)n;
	double lower = 4 * xbar - 1.96 * 4 * sqrt(xbar*(1 - xbar) / n);
	double upper = 4 * xbar + 1.96 * 4 * sqrt(xbar*(1 - xbar) / n);
	return vector < double > {lower, xbar*4., upper};
}

/*
QuickSelect(A, k): select the kth smallest element out of an array of n element
1. given an array, choose a pivot element p of median of n/5 medians
2. partition A around p, let A1, A2, A3 be the subarray of elements <, ==, > p
3. if k <= len(A1), return QuickSelect(A1, k)
4. else if k > len(A1)+len(A2), return QuickSelect(A3, k-len(A1)-len(A2)
5. Else: return p
T(n) <= T(7n/10) + O(n)
Time complexity: O(n); Space complexity: O(n)
*/
int find_kth(int* v, int n, int k) {
	//int n = v.size();
	if (n == 1 && k == 0) return v[0];
	int m = (n + 4) / 5; // (n+4)/5 groups
	int* medians = new int[m]; // record the m medians
	for (int i = 0; i<m; i++) {
		if (5 * i + 4 < n) { // Group 0,1,..,m-2
			int* w = v + 5 * i;
			// selection sort performed on the 5-element subarray starting with w
			for (int j0 = 0; j0 < 3; ++j0) {
				int jmin = j0;
				for (int j = j0 + 1; j<5; ++j) {
					if (w[j] < w[jmin]) jmin = j;
				}
				swap(w[j0], w[jmin]);
			}
			// after sorting, obtain the median for the subgroup
			medians[i] = w[2];
		}
		else { // Group m-1
			medians[i] = v[5 * i];
			// just use the first element for the m-1 group as median
		}
	}

	int pivot = find_kth(medians, m, m / 2); // obtain median by (m/2)th element
	delete[] medians;

	// put the pivot to the end position of the array
	for (int i = 0; i < n; ++i) {
		if (v[i] == pivot) {
			swap(v[i], v[n - 1]);
			break;
		}
	}

	// partition the array using auxiliary variable store (count)
	int store = 0;
	for (int i = 0; i < n - 1; i++) {
		if (v[i] < pivot) {
			swap(v[i], v[store++]);
		}
	}
	swap(v[store], v[n - 1]);

	// if store == k, already found the median (pivot)
	if (store == k) {
		return pivot;
	}
	// if store > k, pivot is too big for the kth statistic, search from 0 (length=store), kth statistic
	else if (store > k) {
		return find_kth(v, store, k);
	}
	// if store < k, pivot is too small, search from store+1 (length=n-1-store), 
	else {
		return find_kth(v + store + 1, n - store - 1, k - store - 1);
	}
}
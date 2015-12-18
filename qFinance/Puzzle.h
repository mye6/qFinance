#ifndef PUZZLE_H
#define PUZZLE_H

// input: hour, minute
// output: angle of the hour hand
double angle_hour(int hour, int minute);

// input: minute
// output: angle of the minute hand
double angle_minute(int minute);

// struct: hour, minute, to make the calculation convenient
struct HM {
	int hour;
	int minute;
	HM(int hour_, int minute_) : hour(hour_), minute(minute_) { }
};

// easy to output struct HM
void print(ostream& os, const HM& hm);

// used newton's method for calculation
// minute hand <= hour hange at minute && minute hand > hour hand at minute + 1 for each hour
void hm_cross(const string& filename, int hour_end);


// green book ,birthday problem in probability theory
//smallest number of people with same birthday has probability > 0.5
int birthday();


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
vector<double> pi_estimator(int n = 10);

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
int find_kth(int* v, int n, int k);

#endif
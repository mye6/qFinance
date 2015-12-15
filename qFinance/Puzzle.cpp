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
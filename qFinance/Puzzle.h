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

#endif
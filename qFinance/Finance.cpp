#include "Finance.h"

// calculate the pv of cahs flows for discrete dates and annual compounding rates
double cash_flow_pv_discrete(
	const vector<double>& cflow_times,
	const vector<double>& cflow_amounts,
	const double& r) {

	if (cflow_times.size() != cflow_amounts.size()) return ERROR;

	double PV = 0.;
	for (size_t i = 0; i < cflow_times.size(); ++i)
		PV += cflow_amounts[i] / pow(1. + r, cflow_times[i]);
	return PV;
}

double cash_flow_irr_discrete(
	const vector<double>& cflow_times,
	const vector<double>& cflow_amounts) {

	// default initial guess
	double x1 = 0.0, x2 = 0.2;
	double f1 = cash_flow_pv_discrete(cflow_times, cflow_amounts, x1);
	double f2 = cash_flow_pv_discrete(cflow_times, cflow_amounts, x2);

	// ensure f(x1) and f(x2) have different signs
	for (int i = 0; i < MAX_ITERATIONS; ++i) {
		if (f1*f2 < 0.0) break;
		if (fabs(f1) < fabs(f2))
			f1 = cash_flow_pv_discrete(cflow_times, cflow_amounts, x1 += 1.6*(x1 - x2));
		else
			f2 = cash_flow_pv_discrete(cflow_times, cflow_amounts, x2 += 1.6*(x2 - x1));
	}
	if (f2*f1 > 0.0) return ERROR;

	// refine the initial guess to rtb- root begin, dx - delta x of each step	
	double f = cash_flow_pv_discrete(cflow_times, cflow_amounts, x1);
	double rtb, dx;
	if (f < 0.0) { rtb = x1; dx = x2 - x1; }
	else { rtb = x2; dx = x1 - x2; }

	for (int i = 0; i < MAX_ITERATIONS; ++i) {
		dx *= 0.5;
		double x_mid = rtb + dx;
		double f_mid = cash_flow_pv_discrete(cflow_times, cflow_amounts, x_mid);
		if (f_mid <= 0.0) rtb = x_mid;
		if ((fabs(f_mid) < ACCURACY) || (fabs(dx) < ACCURACY)) return x_mid;
	}
	return ERROR;
}

bool cash_flow_unique_irr(
	const vector<double>& cflow_times,
	const vector<double>& cflow_amounts) {

	int sign_changes = 0;
	for (size_t i = 1; i < cflow_times.size(); ++i) {
		if (sgn(cflow_amounts[i - 1]) != sgn(cflow_amounts[i])) sign_changes++;
	}

	if (sign_changes == 0)  return false;
	if (sign_changes == 1)  return true;

	double A = cflow_amounts[0];
	sign_changes = 0;
	for (size_t i = 1; i <= cflow_times.size(); ++i) {
		if (sgn(A) != sgn(A += cflow_amounts[i])) sign_changes++;
	}
	if (sign_changes <= 1) return true;
	return false;
}

// cash flow for continuously compounded cash flows
double cash_flow_pv(
	const vector<double>& cflow_times,
	const vector<double>& cflow_amounts,
	const double& r) {

	if (cflow_times.size() != cflow_amounts.size()) return ERROR;
	double PV = 0.;
	for (size_t i = 0; i < cflow_times.size(); ++i)
		PV += cflow_amounts[i] * exp(-r*cflow_times[i]);
	return PV;
}

// calculate the bond price for discrete dates and annual compounding rates
double bonds_price_discrete(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& r) {

	return cash_flow_pv_discrete(times, cashflows, r);
}

double bonds_yield_to_maturity_discrete(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& bondprice) {

	// finding a bonds yield to maturity using simple bisection
	double bot = 0., top = 1.0;
	while (bonds_price_discrete(times, cashflows, top) > bondprice)
		top *= 2.;

	double r = 0.5*(top + bot);
	for (int i = 0; i < MAX_ITERATIONS; ++i) {
		double diff = bonds_price_discrete(times, cashflows, r) - bondprice;
		if (fabs(diff) < ACCURACY) return r;
		if (diff > 0.0) bot = r;
		else top = r;
		r = 0.5*(top + bot);
	}
	return r;
}

double bonds_duration_discrete(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& r) {

	double B = 0., D = 0.;
	for (size_t i = 0; i < times.size(); ++i) {
		D += times[i] * cashflows[i] / pow(1 + r, times[i]);
		B += cashflows[i] / pow(1 + r, times[i]);
	}
	return D / B;
}

double bonds_duration_modified_discrete(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& bondprice) {

	double y = bonds_yield_to_maturity_discrete(times, cashflows, bondprice);
	double D = bonds_duration_discrete(times, cashflows, y);
	return D / (1 + y);
}

double bonds_convexity_discrete(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& r) {

	double Cx = 0.0;
	for (size_t i = 0; i < times.size(); ++i) {
		Cx += cashflows[i] * times[i] * (times[i] + 1) / pow(1 + r, times[i]);
	}
	double B = bonds_price_discrete(times, cashflows, r);
	return (Cx / pow(1 + r, 2)) / B;
}

double bonds_price(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& r) {

	return cash_flow_pv(times, cashflows, r);
}
// calculate the bond price for discrete dates and continuously compounded rates

double bonds_yield_to_maturity(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& bondprice) {

	// finding a bonds yield to maturity using simple bisection
	double bot = 0., top = 1.0;
	while (bonds_price(times, cashflows, top) > bondprice)
		top *= 2.;

	double r = 0.5*(top + bot);
	for (int i = 0; i < MAX_ITERATIONS; ++i) {
		double diff = bonds_price(times, cashflows, r) - bondprice;
		if (fabs(diff) < ACCURACY) return r;
		if (diff > 0.0) bot = r;
		else top = r;
		r = 0.5*(top + bot);
	}
	return r;
}
// calculate the YTM of bond for discrete dates and annual compounding rates

double bonds_duration(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& r) {

	double B = 0., D = 0.;
	for (size_t i = 0; i < times.size(); ++i) {
		D += times[i] * cashflows[i] * exp(-r*times[i]);
		B += cashflows[i] * exp(-r*times[i]);
	}
	return D / B;
}

double bonds_duration_macaulay(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& bondprice) {

	double y = bonds_yield_to_maturity(times, cashflows, bondprice);
	return bonds_duration(times, cashflows, y);
}

double bonds_convexity(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& r) {

	double Cx = 0.0;
	for (size_t i = 0; i < times.size(); ++i) {
		Cx += cashflows[i] * times[i] * times[i] * exp(-r*times[i]);
	}
	double B = bonds_price(times, cashflows, r);
	return Cx / B;
}

// Futures
double futures_price(
	const double& S,
	const double& r,
	const double& time_to_maturity) {

	return exp(r*time_to_maturity)*S;
}

// call price in Black Scholes, no dividend yield
double option_price_call_black_scholes(
	double S, double K, double r, double sigma, double time) {
	
	double vol_time_sqrt = sigma*sqrt(time);
	double d1 = (log(S / K) + r*time) / vol_time_sqrt + 0.5*vol_time_sqrt;
	double d2 = d1 - vol_time_sqrt;
	return S * N(d1) - K*exp(-r*time)*N(d2);
}

// put price in Black Scholes, no dividend yield
double option_price_put_black_scholes(
	double S, double K, double r, double sigma, double time) {
	
	double vol_time_sqrt = sigma*sqrt(time);
	double d1 = (log(S / K) + r*time) / vol_time_sqrt + 0.5*vol_time_sqrt;
	double d2 = d1 - vol_time_sqrt;
	return K*exp(-r*time)*N(-d2) - S * N(-d1);
}

// calculate the greeks for call option
void option_price_partials_call_black_scholes(
	double S, double K, double r, double sigma, double time,
	double& Delta, double& Gamma, double& Theta, double& Vega, double& Rho) {

	double time_sqrt = sqrt(time);
	double d1 = (log(S / K) + r*time) / (sigma*time_sqrt) + 0.5*sigma*time_sqrt;
	double d2 = d1 - sigma*time_sqrt;
	Delta = N(d1);
	Gamma = n(d1) / (S*sigma*time_sqrt);
	Theta = -(S*sigma*n(d1)) / (2 * time_sqrt) - r*K*exp(-r*time)*N(d2); // over t, not (T-t)
	Vega = S*time_sqrt*n(d1);
	Rho = K*time*exp(-r*time)*N(d2);
}

// calculate the greeks for put option
void option_price_partials_put_black_scholes(
	double S, double K, double r, double sigma, double time,
	double& Delta, double& Gamma, double& Theta, double& Vega, double& Rho) {

	double time_sqrt = sqrt(time);
	double d1 = (log(S / K) + r*time) / (sigma*time_sqrt) + 0.5*sigma*time_sqrt;
	double d2 = d1 - sigma*time_sqrt;
	Delta = N(d1) - 1.;
	Gamma = n(d1) / (S*sigma*time_sqrt);
	Theta = -(S*sigma*n(d1)) / (2 * time_sqrt) + r*K*exp(-r*time)*N(-d2); // over t, not (T-t)
	Vega = S*time_sqrt*n(d1);
	Rho = -K*time*exp(-r*time)*N(-d2);
}

// calculate the implied BS volatility from call option using bisections 
double option_price_implied_volatility_call_black_scholes_bisections(
	double S, double K, double r, double time, double option_price) {

	if (option_price < 0.99*(S - K*exp(-time*r))) return 0.0;
	double low = 1.e-5, high = 0.3;
	double price = option_price_call_black_scholes(S, K, r, high, time);
	while (price < option_price) {
		high *= 2.0;
		price = option_price_call_black_scholes(S, K, r, high, time);
		if (high > HIGH_VALUE) return ERROR;
	}
	for (int i = 0; i < MAX_ITERATIONS; ++i) {
		double mid = 0.5*(low + high);
		price = option_price_call_black_scholes(S, K, r, mid, time);
		double test = price - option_price;
		if (fabs(test) < ACCURACY) return mid;
		if (test < 0.0) low = mid;
		else high = mid;
	}
	return ERROR;
}

// calculate the implied BS volatility from put option using bisections
double option_price_implied_volatility_put_black_scholes_bisections(
	double S, double K, double r, double time, double option_price) {

	if (option_price < 0.99*(K*exp(-time*r) - S)) return 0.0;

	double low = 1.e-5, high = 0.3;
	double price = option_price_put_black_scholes(S, K, r, high, time);
	while (price < option_price) {
		high *= 2.0;
		price = option_price_put_black_scholes(S, K, r, high, time);
		if (high > HIGH_VALUE) return ERROR;
	}
	for (int i = 0; i < MAX_ITERATIONS; ++i) {
		double mid = 0.5*(low + high);
		price = option_price_put_black_scholes(S, K, r, mid, time);
		double test = price - option_price;
		if (fabs(test) < ACCURACY) return mid;
		if (test < 0.0) low = mid;
		else high = mid;
	}
	return ERROR;
}

// calculate the implied BS volatility from call option using Newton's method
double option_price_implied_volatility_call_black_scholes_newton(
	double S, double K, double r, double time, double option_price) {

	if (option_price < 0.99*(S - K*exp(-time*r))) return 0.0;
	double time_sqrt = sqrt(time);
	double sigma = (option_price / S) / (0.398*time_sqrt);

	for (int i = 0; i < MAX_ITERATIONS; ++i) {
		double price = option_price_call_black_scholes(S, K, r, sigma, time);
		double diff = option_price - price;
		if (fabs(diff) < ACCURACY) return sigma;
		double d1 = (log(S / K) + r*time) / (sigma*time_sqrt) + 0.5*sigma*time_sqrt;
		double Vega = S*time_sqrt*n(d1);
		sigma += diff / Vega;
	}
	return ERROR;
}

// calculate the implied BS volatility from put option using Newton's method
double option_price_implied_volatility_put_black_scholes_newton(
	double S, double K, double r, double time, double option_price) {

	if (option_price < 0.99*(K*exp(-time*r) - S)) return 0.0;
	double time_sqrt = sqrt(time);
	// double sigma = (option_price / S) / (0.398*time_sqrt);
	double sigma = 0.2;

	for (int i = 0; i < MAX_ITERATIONS; ++i) {
		double price = option_price_put_black_scholes(S, K, r, sigma, time);
		double diff = option_price - price;
		if (fabs(diff) < ACCURACY) return sigma;
		double d1 = (log(S / K) + r*time) / (sigma*time_sqrt) + 0.5*sigma*time_sqrt;
		double Vega = S*time_sqrt*n(d1);
		sigma += diff / Vega;
	}
	return ERROR;
}

// call price in Black Scholes, with dividend yield payout
double option_price_european_call_payout(
	double S, double K, double r, double q, double sigma, double time) {
	
	double Sadj = S * exp(-q*time);
	return option_price_call_black_scholes(Sadj, K, r, sigma, time);
}

// put price in Black Scholes, with dividend yield payout
double option_price_european_put_payout(
	double S, double K, double r, double q, double sigma, double time) {

	double Sadj = S * exp(-q*time);	
	return option_price_put_black_scholes(Sadj, K, r, sigma, time);
}

// call price in Black Scholes, with dividend payments
double option_price_european_call_dividends(
	double S, double K, double r, double sigma, double time,
	const vector<double>& dividend_times, const vector<double>& dividend_amounts) {

	double Sadj = S - cash_flow_pv(dividend_times, dividend_amounts, r);
	return option_price_call_black_scholes(Sadj, K, r, sigma, time);
}

// put price in Black Scholes, with dividend payments
double option_price_european_put_dividends(
	double S, double K, double r, double q, double sigma, double time,
	const vector<double>& dividend_times, const vector<double>& dividend_amounts) {

	double Sadj = S - cash_flow_pv(dividend_times, dividend_amounts, r);
	return option_price_put_black_scholes(Sadj, K, r, sigma, time);
}

// american call with one dividend payment, Roll-Geske-Whaley formula
double option_price_american_call_one_dividend(
	double S, double K, double r, double sigma, double tau, double D1, double tau1) {

	if (D1 <= K*(1.0 - exp(-r*(tau - tau1)) ) )
		return option_price_call_black_scholes(S - D1*exp(-r*tau1), K, r, sigma, tau);
	
	// find S_bar
	// 1. locate S_high
	double S_high = S;
	double c = option_price_call_black_scholes(S_high,K,r,sigma,tau-tau1);
	double test = c - S_high - D1 + K;
	while ((test > 0.0) && (S_high <= HIGH_VALUE)) {
		S_high *= 2.0;
		c = option_price_call_black_scholes(S_high, K, r, sigma, tau - tau1);
		test = c - S_high - D1 + K;
	}
	if (S_high > HIGH_VALUE) {
		return option_price_call_black_scholes(S - D1*exp(-r*tau1), K, r, sigma, tau);
	}

	// 2. find S_bar that solves c = S_bar-D+K based on S_high
	double S_low = 0.0;
	double S_bar = 0.5*(S_low + S_high);
	c = option_price_call_black_scholes(S_bar, K, r, sigma, tau - tau1);
	test = c - S_bar - D1 + K;
	while ((fabs(test)>ACCURACY) && ((S_high-S_low)>ACCURACY)) {
		if (test < 0.0) S_high = S_bar;
		else S_low = S_bar;
		S_bar = 0.5*(S_high + S_low);
		c = option_price_call_black_scholes(S_bar, K, r, sigma, tau-tau1);
		test = c - S_bar - D1 + K;
	}

	// use S_bar to calculate the option price
	double tau_sqrt = sqrt(tau), tau1_sqrt = sqrt(tau1);
	double rho = -sqrt(tau1 / tau);
	double a1 = (log((S - D1*exp(-r*tau1))/K) + (r+0.5*sigma*sigma)*tau) / (sigma * tau_sqrt);
	double a2 = a1 - sigma*tau_sqrt;
	double b1 = (log((S - D1*exp(-r*tau1)) / S_bar) + (r + 0.5*sigma*sigma)*tau1) / (sigma * tau1_sqrt);
	double b2 = b1 - sigma * tau1_sqrt;
	double C = (S - D1*exp(-r*tau1))*N(b1) + (S - D1*exp(-r*tau1))*N(a1, -b1, rho)
		- K*exp(-r*tau)*N(a2, -b2, rho) - (K - D1)*exp(-r*tau1)*N(b2);
	return C;
}

// american call with perpetual dividend yield
double option_price_american_perpetual_call(
	double S, double K, double r, double q, double sigma) {

	double sigma_sqr = sigma*sigma;
	double h1 = 0.5 - (r - q) / sigma_sqr;
	h1 += sqrt(pow(((r-q)/sigma_sqr - 0.5), 2) + 2*r/sigma_sqr);
	double cp = (K / (h1 - 1.0))*pow((h1 - 1.)*S / (h1*K), h1);
	return cp;
}

// European call options writen on a Futures
double futures_option_price_call_european_black(
	double F, double K, double r, double sigma, double time) {
	
	double sigma_sqr = sigma*sigma;
	double time_sqrt = sqrt(time);
	double d1 = (log(F/K) + 0.5*sigma_sqr*time) / (sigma*time_sqrt);
	double d2 = d1 - sigma * time_sqrt;	
	return exp(-r*time)*(F*N(d1) - K*N(d2) );
	
}

// European put options writen on a Futures
double futures_option_price_put_european_black(
	double F, double K, double r, double sigma, double time) {

	double sigma_sqr = sigma*sigma;
	double time_sqrt = sqrt(time);
	double d1 = (log(F / K) + 0.5*sigma_sqr*time) / (sigma*time_sqrt);
	double d2 = d1 - sigma * time_sqrt;	
	return exp(-r*time)*(K*N(-d2)-F*N(-d1));	
}

// European call options writen on foreign currency
double currency_option_price_call_european_black(
	double S, double K, double r, double rf, double sigma, double time) {
	return option_price_european_call_payout(S, K, r, rf, sigma, time);
}

// European put options writen on foreign currency
double currency_option_price_put_european_black(
	double S, double K, double r, double rf, double sigma, double time) {
	return option_price_european_put_payout(S, K, r, rf, sigma, time);
}

// european call option using binomial tree approximation
double option_price_call_european_binomial(
	double S, double K, double r, double sigma, double time, int steps) {
	
	double R = exp(r*(time/steps)), u = exp(sigma*sqrt(time/steps));
	double uu = u*u, d = 1./u;

	vector<double> prices(steps + 1);
	prices[0] = S*pow(d, steps);	
	for (int i = 1; i <= steps; ++i)
		prices[i] = uu*prices[i-1];

	vector<double> values(steps + 1);
	for (int i = 0; i <= steps; ++i)
		values[i] = max(0.0, (prices[i] - K));
	
	double Rinv = 1. / R, p_up = (R - d) / (u - d), p_down = 1. - p_up;
	for (int step = steps - 1; step >= 0; --step) {
		for (int i = 0; i <= step; ++i) {
			values[i] = Rinv*(p_up*values[i + 1] + p_down*values[i]);		
		}
	}
	return values[0];
}

// european put option using binomial tree approximation
double option_price_put_european_binomial(
	double S, double K, double r, double sigma, double time, int steps) {
	
	double R = exp(r*(time / steps)), u = exp(sigma*sqrt(time / steps));
	double uu = u*u, d = 1. / u;

	vector<double> prices(steps + 1);
	prices[0] = S*pow(d, steps);
	for (int i = 1; i <= steps; ++i)
		prices[i] = uu*prices[i - 1];

	vector<double> values(steps + 1);
	for (int i = 0; i <= steps; ++i)
		values[i] = max(0.0, (K - prices[i]));

	double Rinv = 1. / R, p_up = (R - d) / (u - d), p_down = 1. - p_up;
	for (int step = steps - 1; step >= 0; --step) {
		for (int i = 0; i <= step; ++i) {
			values[i] = Rinv*(p_up*values[i + 1] + p_down*values[i]);
		}
	}
	return values[0];
}


// american call option using binomial tree approximation
double option_price_call_american_binomial(
	double S, double K, double r, double sigma, double time, int steps) {

	double R = exp(r*(time / steps)), u = exp(sigma*sqrt(time / steps));
	double uu = u*u, d = 1. / u;

	vector<double> prices(steps + 1);
	prices[0] = S*pow(d, steps);
	for (int i = 1; i <= steps; ++i)
		prices[i] = uu*prices[i - 1];

	vector<double> values(steps + 1);
	for (int i = 0; i <= steps; ++i)
		values[i] = max(0.0, (prices[i] - K));

	double Rinv = 1. / R, p_up = (R - d) / (u - d), p_down = 1. - p_up;
	for (int step = steps - 1; step >= 0; --step) {
		for (int i = 0; i <= step; ++i) {
			values[i] = Rinv*(p_up*values[i + 1] + p_down*values[i]);
			prices[i] = d*prices[i + 1];
			values[i] = max(values[i], prices[i] - K);
		}
	}
	return values[0];
}

// american put option using binomial tree approximation
double option_price_put_american_binomial(
	double S, double K, double r, double sigma, double time, int steps) {

	double R = exp(r*(time / steps)), u = exp(sigma*sqrt(time / steps));
	double uu = u*u, d = 1. / u;

	vector<double> prices(steps + 1);
	prices[0] = S*pow(d, steps);
	for (int i = 1; i <= steps; ++i)
		prices[i] = uu*prices[i - 1];

	vector<double> values(steps + 1);
	for (int i = 0; i <= steps; ++i)
		values[i] = max(0.0, (K - prices[i]));

	double Rinv = 1. / R, p_up = (R - d) / (u - d), p_down = 1. - p_up;
	for (int step = steps - 1; step >= 0; --step) {
		for (int i = 0; i <= step; ++i) {
			values[i] = Rinv*(p_up*values[i + 1] + p_down*values[i]);
			prices[i] = d*prices[i + 1];
			values[i] = max(values[i], K - prices[i]);
		}
	}
	return values[0];
}

// american call option partials using binomial tree approximation
void option_price_partials_american_call_binomial(
	double S, double K, double r, double sigma, double time,
	double& delta, double& gamma, double& theta, double& vega, double& rho, int steps) {

	double R = exp(r*(time / steps)), u = exp(sigma*sqrt(time / steps));
	double uu = u*u, d = 1. / u;

	vector<double> prices(steps + 1);
	prices[0] = S*pow(d, steps);
	for (int i = 1; i <= steps; ++i)
		prices[i] = uu*prices[i - 1];

	vector<double> values(steps + 1);
	for (int i = 0; i <= steps; ++i)
		values[i] = max(0.0, (prices[i] - K));

	double Rinv = 1. / R, p_up = (R - d) / (u - d), p_down = 1. - p_up;
	for (int step = steps - 1; step >= 2; --step) {
		for (int i = 0; i <= step; ++i) {
			values[i] = Rinv*(p_up*values[i + 1] + p_down*values[i]);
			prices[i] = d*prices[i + 1];
			values[i] = max(values[i], prices[i] - K);
		}
	}
	double f22 = values[2], f21 = values[1], f20 = values[0];
	
	for (int i = 0; i <= 1; ++i) {
		values[i] = Rinv*(p_up*values[i + 1] + p_down*values[i]);
		prices[i] = d*prices[i + 1];
		values[i] = max(values[i], prices[i] - K);
	}
	double f11 = values[1], f10 = values[0];

	values[0] = Rinv*(p_up*values[1] + p_down*values[0]);
	prices[0] = d*prices[1];
	values[0] = max(values[0], prices[0] - K);
	double f00 = values[0];

	delta = (f11 - f10) / (S*u - S*d);
	double dt = time / (double)steps;
	theta = (f21 - f00) / (2.*dt);
	double h = 0.5 * S * (uu - d*d);
	gamma = ((f22-f21)/(S*(uu-1.)) - (f21-f20)/(S*(1-d*d)))/ h;
	
	double diff = 0.02;
	double tmp_sigma = sigma + diff;
	double tmp_price = option_price_call_american_binomial(S,K,r,tmp_sigma, time,steps);
	vega = (tmp_price - f00) / diff;

	diff = 0.05;
	double tmp_r = r + diff;
	tmp_price = option_price_call_american_binomial(S, K, tmp_r, sigma, time, steps);
	rho = (tmp_price - f00) / diff;
}

// american put option greeks using binomial tree approximation
void option_price_partials_american_put_binomial(
	double S, double K, double r, double sigma, double time,
	double& delta, double& gamma, double& theta, double& vega, double& rho, int steps) {

	double R = exp(r*(time / steps)), u = exp(sigma*sqrt(time / steps));
	double uu = u*u, d = 1. / u;

	vector<double> prices(steps + 1);
	prices[0] = S*pow(d, steps);
	for (int i = 1; i <= steps; ++i)
		prices[i] = uu*prices[i - 1];

	vector<double> values(steps + 1);
	for (int i = 0; i <= steps; ++i)
		values[i] = max(0.0, (K - prices[i]));

	double Rinv = 1. / R, p_up = (R - d) / (u - d), p_down = 1. - p_up;
	for (int step = steps - 1; step >= 2; --step) {
		for (int i = 0; i <= step; ++i) {
			values[i] = Rinv*(p_up*values[i + 1] + p_down*values[i]);
			prices[i] = d*prices[i + 1];
			values[i] = max(values[i], K - prices[i]);
		}
	}
	double f22 = values[2], f21 = values[1], f20 = values[0];

	for (int i = 0; i <= 1; ++i) {
		values[i] = Rinv*(p_up*values[i + 1] + p_down*values[i]);
		prices[i] = d*prices[i + 1];
		values[i] = max(values[i], K - prices[i]);
	}
	double f11 = values[1], f10 = values[0];

	values[0] = Rinv*(p_up*values[1] + p_down*values[0]);
	prices[0] = d*prices[1];
	values[0] = max(values[0], K - prices[0]);
	double f00 = values[0];

	delta = (f11 - f10) / (S*u - S*d);
	double dt = time / (double)steps;
	theta = (f21 - f00) / (2.*dt);
	double h = 0.5 * S * (uu - d*d);
	gamma = ((f22 - f21) / (S*(uu - 1.)) - (f21 - f20) / (S*(1 - d*d))) / h;

	double diff = 0.02;
	double tmp_sigma = sigma + diff;
	double tmp_price = option_price_put_american_binomial(S, K, r, tmp_sigma, time, steps);
	vega = (tmp_price - f00) / diff;

	diff = 0.05;
	double tmp_r = r + diff;
	tmp_price = option_price_put_american_binomial(S, K, tmp_r, sigma, time, steps);
	rho = (tmp_price - f00) / diff;
}

// european call option using finite difference, explicit method
double option_price_call_european_finite_diff_explicit(
	double S, double K, double r, double sigma, double time, int Ns, int Nt) {
	return 0.0;
}

// european put option using finite difference, explicit method
double option_price_put_european_finite_diff_explicit(
	double S, double K, double r, double sigma, double time, int Ns, int Nt) {

	double sigma_sqr = sigma * sigma;
	int M = Ns; if ((M % 2) == 1) ++M;
	double delta_S = 2.0*S/(double)M;
	PRINT(delta_S);

	int N = Nt; double delta_t = time/(double)N;
	PRINT(delta_t);

	cout << "ratio: " << delta_t / (delta_S * delta_S) << endl;

	vector<double> S_values(M + 1);
	for (int m = 0; m <= M; ++m) S_values[m] = m*delta_S;
	print_vec<double>(S_values);
	
	vector<double> a(M), b(M), c(M), d(M);
	double r1 = 1. / (1. + r*delta_t);
	double r2 = delta_t * r1;
	for (int j = 1; j < M; ++j) {
		a[j] = r2 * 0.5 * j * (sigma_sqr*j - r);
		b[j] = r1 * (1. - sigma_sqr*j*j*delta_t);
		c[j] = r2 * 0.5 * j * (sigma_sqr*j + r);
		d[j] = a[j] + b[j] + c[j];
	}
	cout << "a" << endl;
	print_vec<double>(a, " ", a.size());
	cout << "b" << endl;
	print_vec<double>(b, " ", b.size());
	cout << "c" << endl;
	print_vec<double>(c, " ", c.size());
	cout << "d" << endl;
	print_vec<double>(d, " ", d.size());

	vector<double> f_next(M + 1);
	for (int m = 0; m <= M; ++m) f_next[m] = max(0.0, K - S_values[m]);
	cout << "f_next" << endl;
	print_vec<double>(f_next);


	vector<double> f(M + 1);

	SEP;
	for (int t = N - 1; t >= 0; --t) {
		f[0] = K;
		for (int m = 1; m <= M - 1; ++m)
			f[m] = a[m] * f_next[m - 1] + b[m] * f_next[m] + c[m] * f_next[m + 1];
		f[M] = 0.;

		for (int m = 0; m <= M; ++m) f_next[m] = f[m];
		
		cout << f[M/2] << " ";
		//print_vec<double>(f);
	}
	

	return f[M / 2];
}


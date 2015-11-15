#ifndef FINANCE_H
#define FINANCE_H

#include "Solver.h"

const double ERROR = -1e30;
const int MAX_ITERATIONS = 100;
const double ACCURACY = 1.0e-5;
const double HIGH_VALUE = 1.e10;


// Section: cash flow and bond pricing
double cash_flow_pv_discrete(
	const vector<double>& cflow_times,
	const vector<double>& cflow_amounts,
	const double& r);
// calculate the pv of cahs flows for discrete dates and annual compounding rates

double cash_flow_pv(
	const vector<double>& cflow_times,
	const vector<double>& cflow_amounts,
	const double& r);
// calculate the pv of cahs flows for discrete dates and continuously compounding

double cash_flow_irr_discrete(
	const vector<double>& cflow_times,
	const vector<double>& cflow_amounts);
// find the internal intereat rate of returns, discrete dates + annual compounding

bool cash_flow_unique_irr(
	const vector<double>& cflow_times,
	const vector<double>& cflow_amounts);
// simple check to see if a unique IRR exists based on Descartes rule

double bonds_price_discrete(
	const vector<double>& cflow_times,
	const vector<double>& cflow_amounts,
	const double& r);
// calculate the bond price for discrete dates and annual compounding rates

double bonds_yield_to_maturity_discrete(
	const vector<double>& cflow_times,
	const vector<double>& cflow_amounts,
	const double& bondprice);
// calculate the YTM of bond for discrete dates and annual compounding rates

double bonds_duration_discrete(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& r);

double bonds_duration_modified_discrete(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& bondprice);

double bonds_convexity_discrete(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& r);

double bonds_price(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& r);
// calculate the bond price for discrete dates and continuously compounded rates

double bonds_yield_to_maturity(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& bondprice);
// calculate the YTM of bond for discrete dates and annual compounding rates

double bonds_duration(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& r);

double bonds_duration_macaulay(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& bondprice);

double bonds_convexity(
	const vector<double>& times,
	const vector<double>& cashflows,
	const double& r);

// Section: Futures
double futures_price(
	const double& S,
	const double& r,
	const double& time_to_maturity);

// Section: Option Pricer
// Option Pricer via Black Scholes
// call price in Black Scholes, no dividend yield
double option_price_call_black_scholes(double S, double K, double r, double sigma, double time);

// put price in Black Scholes, no dividend yield
double option_price_put_black_scholes(double S, double K, double r, double sigma, double time);

// calculate the greeks for call option
void option_price_partials_call_black_scholes(
	double S, double K, double r, double sigma, double time,
	double& Delta, double& Gamma, double& Theta, double& Vega, double& Rho);

// calculate the greeks for put option
void option_price_partials_put_black_scholes(
	double S, double K, double r, double sigma, double time,
	double& Delta, double& Gamma, double& Theta, double& Vega, double& Rho);

// call price in Black Scholes, with dividend yield payout
double option_price_european_call_payout(double S, double K, double r, double q, double sigma, double time);

// put price in Black Scholes, with dividend yield payout
double option_price_european_put_payout(double S, double K, double r, double q, double sigma, double time);

// call price in Black Scholes, with dividend payments
double option_price_european_call_dividends(
	double S, double K, double r, double sigma, double time,
	const vector<double>& dividend_times, const vector<double>& dividend_amounts);

// put price in Black Scholes, with dividend payments
double option_price_european_put_dividends(
	double S, double K, double r, double q, double sigma, double time,
	const vector<double>& dividend_times, const vector<double>& dividend_amounts);

// american call with one dividend payment, Roll-Geske-Whaley formula
double option_price_american_call_one_dividend(
	double S, double K, double r, double sigma, double tau, double D1, double tau1);

// american call with perpetual dividend yield
double option_price_american_perpetual_call(
	double S, double K, double r, double q, double sigma);

// European call options writen on a Futures
double futures_option_price_call_european_black(
	double F, double K, double r, double sigma, double time);

// European put options writen on a Futures
double futures_option_price_put_european_black(
	double F, double K, double r, double sigma, double time);

// European call options writen on foreign currency
double currency_option_price_call_european_black(
	double S, double K, double r, double rf, double sigma, double time);

// European put options writen on foreign currency
double currency_option_price_put_european_black(
	double S, double K, double r, double rf, double sigma, double time);

// Section: Implied Volatility
// calculate the implied BS volatility from call option using bisections 
double option_price_implied_volatility_call_black_scholes_bisections(
	double S, double K, double r, double time, double option_price);

// calculate the implied BS volatility from put option using bisections
double option_price_implied_volatility_put_black_scholes_bisections(
	double S, double K, double r, double time, double option_price);

// calculate the implied BS volatility from call option using Newton's method
double option_price_implied_volatility_call_black_scholes_newton(
	double S, double K, double r, double time, double option_price);

// calculate the implied BS volatility from put option using Newton's method
double option_price_implied_volatility_put_black_scholes_newton(
	double S, double K, double r, double time, double option_price);

// european call option using binomial tree approximation
double option_price_call_european_binomial(
	double S, double K, double r, double sigma, double time, int steps = 1000);

// european put option using binomial tree approximation
double option_price_put_european_binomial(
	double S, double K, double r, double sigma, double time, int steps = 1000);

// american call option using binomial tree approximation
double option_price_call_american_binomial(
	double S, double K, double r, double sigma, double time, int steps = 1000);

// american put option using binomial tree approximation
double option_price_put_american_binomial(
	double S, double K, double r, double sigma, double time, int steps = 1000);

// american call option partials using binomial tree approximation
void option_price_partials_american_call_binomial(
	double S, double K, double r, double sigma, double time, 
	double& delta, double& gamma, double& theta, double& vega, double& rho, int steps = 1000);

// american put option greeks using binomial tree approximation
void option_price_partials_american_put_binomial(
	double S, double K, double r, double sigma, double time, 
	double& delta, double& gamma, double& theta, double& vega, double& rho, int steps = 1000);

// european call option using finite difference, explicit method
double option_price_call_european_finite_diff_explicit(
	double S, double K, double r, double sigma, double time, int Ns = 200, int Nt = 200);

// european put option using finite difference, explicit method
double option_price_put_european_finite_diff_explicit(
	double S, double K, double r, double sigma, double time, int Ns = 200, int Nt = 200);

#endif
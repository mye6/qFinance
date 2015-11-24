#include "Solver.h"

PayOffCall::PayOffCall(double Strike_) : Strike(Strike_) {}
double PayOffCall::operator()(double Spot) const { return max(Spot - Strike, 0.0); }
PayOff* PayOffCall::clone() const { return new PayOffCall(*this); }

PayOffPut::PayOffPut(double Strike_) : Strike(Strike_) {}
double PayOffPut::operator()(double Spot) const { return max(Strike - Spot, 0.0); }
PayOff* PayOffPut::clone() const { return new PayOffPut(*this); }

PayOffFactory& PayOffFactory::Instance() {
	static PayOffFactory theFactory;
	return theFactory;
}

void PayOffFactory::RegisterPayOff(string PayOffId, CreatePayOffFunction CreatorFunction) {
	TheCreatorFunctions.insert(pair<string, CreatePayOffFunction>(PayOffId, CreatorFunction));
}

PayOff* PayOffFactory::CreatePayOff(string PayOffId, double Strike) {
	map<string, CreatePayOffFunction>::const_iterator i = TheCreatorFunctions.find(PayOffId);
	if (i == TheCreatorFunctions.end()) {
		cout << PayOffId << " is an unknown payoff" << endl;
		return NULL;
	}
	return (i->second)(Strike);
}

struct BeforeMain {
	BeforeMain() {
		cout << "before main" << endl;
		// global variables, RegisterCall and RegisterPut
		// will call the constructor of PayOffHelper which
		// calls the RegisterPayOff function of PayOffFactory class
		PayOffHelper<PayOffCall> RegisterCall("call");		
		PayOffHelper<PayOffPut> RegisterPut("put");
	}
};

#ifdef BEFOREMAIN
namespace {
	BeforeMain theBeforeMain;	
}
#endif

PayOffBridge::PayOffBridge(const PayOffBridge& original) { ThePayOffPtr = original.ThePayOffPtr->clone(); }
PayOffBridge::PayOffBridge(const PayOff& innerPayOff) { ThePayOffPtr = innerPayOff.clone(); }
PayOffBridge& PayOffBridge::operator=(const PayOffBridge& original) {
	if (this == &original) return *this;	
	delete ThePayOffPtr;
	ThePayOffPtr = original.ThePayOffPtr->clone();
	return *this;
}
PayOffBridge::~PayOffBridge(){ delete ThePayOffPtr; }

VanillaOption::VanillaOption(const PayOffBridge &ThePayOff_, double Expiry_) 
	: ThePayOff(ThePayOff_), Expiry(Expiry_) {}
double VanillaOption::GetExpiry() const { return Expiry; }
double VanillaOption::OptionPayOff(double Spot) const { return ThePayOff(Spot); }

ParametersConstant::ParametersConstant(double constant) 
	: Constant(constant), ConstantSquare(Constant * Constant) {}
ParametersInner* ParametersConstant::clone() const { return new ParametersConstant(*this); }
double ParametersConstant::Integral(double time1, double time2) const { return (time2 - time1) * Constant; }
double ParametersConstant::IntegralSquare(double time1, double time2) const { 
	return (time2 - time1) * ConstantSquare;
}

Parameters::Parameters(const ParametersInner& innerObject) { InnerObjectPtr = innerObject.clone(); }
Parameters::Parameters(const Parameters& original) { InnerObjectPtr = original.InnerObjectPtr->clone(); }
Parameters& Parameters::operator= (const Parameters& original) { 
	if (this == &original) return *this;
	delete InnerObjectPtr;
	InnerObjectPtr = original.InnerObjectPtr->clone();	
	return *this;
}
Parameters::~Parameters() { delete InnerObjectPtr; }
double Parameters::Mean(double time1, double time2) const {
	double total = Integral(time1, time2);
	return total / (time2 - time1);
}

double Parameters::RootMeanSquare(double time1, double time2) const {
	double total = IntegralSquare(time1, time2);
	return total / (time2 - time1);
}

StatisticsMean::StatisticsMean() : RunningSum(0.0), PathsDone(0UL) {}
StatisticsMC* StatisticsMean::clone() const { return new StatisticsMean(*this); }
void StatisticsMean::DumpOneResult(double result) { PathsDone++; RunningSum += result; }
vector<vector<double> > StatisticsMean::GetResultsSoFar() const {
	vector<vector<double> > Results(1);
	Results[0].resize(1);
	Results[0][0] = RunningSum / PathsDone;
	return Results;
}

ConvergenceTable::ConvergenceTable(const Wrapper<StatisticsMC>& Inner_) 
	: Inner(Inner_) { StoppingPoint = 2; PathsDone = 0; }
StatisticsMC* ConvergenceTable::clone() const { return new ConvergenceTable(*this); }
void ConvergenceTable::DumpOneResult(double result) {
	Inner->DumpOneResult(result);
	++PathsDone;
	if (PathsDone == StoppingPoint) {
		StoppingPoint *= 2;
		vector<vector<double> > thisResult(Inner->GetResultsSoFar());
		for (unsigned long i = 0; i < thisResult.size(); i++) {
			thisResult[i].push_back(PathsDone);
			ResultsSoFar.push_back(thisResult[i]);
		}
	}
}

vector<vector<double> > ConvergenceTable::GetResultsSoFar() const {
	vector<vector<double> > tmp(ResultsSoFar);
	if (PathsDone * 2 != StoppingPoint) {
		vector<vector<double> > thisResult(Inner->GetResultsSoFar());
		for (unsigned long i = 0; i < thisResult.size(); i++) {
			thisResult[i].push_back(PathsDone);
			tmp.push_back(thisResult[i]);
		}
	}
	return tmp;
}

void SimpleMonteCarlo(
	const VanillaOption &TheOption,
	double Spot,
	const Parameters& Vol,
	const Parameters& r,
	unsigned long NumberOfPaths,
	StatisticsMC& gatherer) {
	double Expiry = TheOption.GetExpiry();
	double variance = Vol.IntegralSquare(0, Expiry);
	double rootVariance = sqrt(variance);
	double itoCorrelation = -0.5*variance;
	double movedSpot = Spot*exp(r.Integral(0, Expiry) + itoCorrelation);
	double thisSpot;
	double discounting = exp(-r.Integral(0.0, Expiry));
	for (unsigned long i = 0; i < NumberOfPaths; i++) {
		double thisGaussian = normal();
		thisSpot = movedSpot*exp(rootVariance*thisGaussian);
		double thisPayOff = TheOption.OptionPayOff(thisSpot);
		gatherer.DumpOneResult(thisPayOff*discounting);
	}
	return;
}

#include "Solver.h"
#include "Finance.h"

int main() {
	double Expiry = 2.0, Strike = 20., Spot = 30., Vol = 0.2, r = 0.05;
	unsigned long NumberOfPaths = 1000;
	PayOffCall thePayOff(Strike);
	VanillaOption theOption(thePayOff, Expiry);
	ParametersConstant VolParam(Vol);
	ParametersConstant rParam(r);
	StatisticsMean gatherer;
	ConvergenceTable gathererTwo(gatherer);
	SimpleMonteCarlo(theOption,
		Spot,
		VolParam,
		rParam,
		NumberOfPaths,
		gathererTwo);
	vector<vector<double> > results = gathererTwo.GetResultsSoFar();
	cout << "\nFor the call price the results are \n";
	for (unsigned long i = 0; i < results.size(); i++) {
		for (unsigned long j = 0; j < results[i].size(); j++)
			cout << results[i][j] << " ";
		cout << "\n";
	}

	system("pause");
	return 0;
}
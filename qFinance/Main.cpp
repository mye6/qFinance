#include "Solver.h"
#include "Finance.h"
#include "Leetcode.h"


void test(PayOff* po) {
	PayOff* po2 = po->clone();
	PRINT(po2->operator()(40.));
	delete po2; // can delete po2 as it is on heap
}



int main() {
	
	

	system("pause");
	return 0;
}
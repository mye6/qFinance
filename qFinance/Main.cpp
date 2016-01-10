#include "Solver.h"
#include "Leetcode.h"
/*
#include "Finance.h"
#include "Puzzle.h"

#include "ThinkCPP.h"
#include "FaqCPP.h"
*/

// 80. Remove Duplicates from Sorted Array II
int removeDuplicatesII(vector<int>& nums) {
	int i = 0;
	for (int n : nums) 
		if (i < 2 || n > nums[i - 2]) nums[i++] = n;
	return i;
}









int main(){
	vector<int> nums{ 1, 1, 1, 2, 2, 3 };
	PRINT(removeDuplicatesII(nums));
	PRINT(nums);
	
	

	system("pause");
	return 0;
}  
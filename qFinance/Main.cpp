#include "Solver.h"
#include "Finance.h"
#include "Puzzle.h"
#include "Leetcode.h"

int removeElement(vector<int>& nums, int val) {
	int i = 0;
	for (size_t j = 0; j < nums.size(); ++j) {
		if (nums[j] != val) nums[i++] = nums[j];
	}
	return i;
}

int main() {
	
	vector<int> nums{ 1, 2, 3, 4, 5, 6, 7 };
	PRINT(removeElement(nums,4));
	PRINT(nums);
	PRINT(removeElement(nums, 5));
	PRINT(nums);
	PRINT(removeElement(nums, 2));
	PRINT(nums);

	system("pause");
	return 0;
}
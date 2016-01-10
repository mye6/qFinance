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

// 27. Remove Element
int removeElement(vector<int>& nums, int val) {
	int i = 0;
	for (int n : nums)
		if (n != val) nums[i++] = n;
	return i;
}

// 118. Pascal's Triangle
vector<vector<int> > generatePascal(int numRows) {
	vector<vector<int>> r(numRows);
	for (int i = 0; i < numRows; i++) {
		r[i].resize(i + 1);
		r[i][0] = r[i][i] = 1;
		for (int j = 1; j < i; j++)
			r[i][j] = r[i - 1][j - 1] + r[i - 1][j];
	}
	return r;
}

// 119. Pascal's Triangle II
vector<int> getRow(int rowIndex) {
	vector<int> A(rowIndex + 1, 0);
	A[0] = 1;
	for (int i = 1; i<rowIndex + 1; i++)
		for (int j = i; j >= 1; j--)
			A[j] += A[j - 1];
	return A;
}

// 283. Move Zeroes
void moveZeroes(vector<int>& nums) {
	int n = nums.size(), k = 0;
	for (int i = 0; i < n; ++i)	
		if (nums[i] != 0) nums[k++] = nums[i];
	for (int i = k; i < n; ++i)
		nums[i] = 0;
}

// 88. Merge Sorted Array
void mergeSortedArray(vector<int>& nums1, int m, vector<int>& nums2, int n) {
	int i = m - 1, j = n - 1, k = m + n - 1;
	while (j >= 0) {
		if (i >= 0 && nums1[i] > nums2[j]) {
			nums1[k--] = nums1[i--]; // only move nums1 here.
		}
		else {
			nums1[k--] = nums2[j--];
		}
	}
}

// 169.1 Majority Element, hash table : value->count.O(N)
int majorityElement1(vector<int>& nums) {
	unordered_map<int, int> counts;
	int n = nums.size();
	for (int i = 0; i < n; i++)
		if (++counts[nums[i]] > n / 2) return nums[i];
	return -1; // dummy
}
// 169.2 Majority Element, Moore Voting Algorithm, O(N)
int majorityElement2(vector<int>& nums) {
	int major, counts = 0, n = nums.size();
	for (int i = 0; i < n; i++) {
		if (counts == 0) { major = nums[i]; counts = 1; }
		else counts += (nums[i] == major ? 1 : -1);
	}
	return major;
}
// 169.3 Majority Element, sort and return the mid element : O(NlogN)
int majorityElement3(vector<int>& nums) {
	//sort(nums.begin(), nums.end());
	nth_element(nums.begin(), nums.begin() + nums.size() / 2, nums.end());
	return nums[nums.size() / 2];
}
// 169.3 Majority Element, sort and count length.O(NlogN)
int majorityElement4(vector<int>& nums) {
	sort(nums.begin(), nums.end());
	int res = nums[0];
	for (int i = 1, n = nums.size(), k = 1; i < n && k < (n + 1) / 2; ++i) {
		(nums[i] == nums[i - 1]) ? ++k : (k = 1);
		if (k >= (n + 1) / 2) res = nums[i];
	}
	return res;
}

// 229. Majority Element II
vector<int> majorityElementII(vector<int>& nums) {
	int candidate1 = 0, candidate2 = 0, count1 = 0, count2 = 0;
	for (int num : nums) {
		if (num == candidate1) count1++;
		else if (num == candidate2) count2++;
		else {
			if (count1 == 0) { candidate1 = num; count1++; }
			else if (count2 == 0) { candidate2 = num; count2++; }
			else { --count1; --count2; }
		}
	}
	count1 = count2 = 0;
	for (int num : nums) {
		if (num == candidate1) ++count1;
		else if (num == candidate2) ++count2;
	}
	int n = nums.size();
	vector<int> majority;
	if (count1 > n / 3) majority.push_back(candidate1);
	if (count2 > n / 3) majority.push_back(candidate2);
	return majority;
}

// 62.1. Unique Paths
int uniquePaths(int m, int n) {
	vector<vector<int> > path(m, vector<int>(n, 1));
	for (int i = 1; i < m; i++)
		for (int j = 1; j < n; j++)
			path[i][j] = path[i - 1][j] + path[i][j - 1];
	return path[m - 1][n - 1];
}

// 62.2. Unique Paths
int combination(int m, int n){
	if (n > m / 2) return combination(m, m - n);
	double result = 1;
	for (int i = 1; i <= n; ++i){
		result *= m - n + i;
		result /= i;
	}
	return (int)result;	
}
int uniquePaths2(int m, int n) {
	return combination(m + n - 2, n - 1);
}

// 63. Unique Paths II
int uniquePathsWithObstacles(vector<vector<int>>& a) {
	if (a.empty() || a[0].empty()) return 0;
	int m = a.size(), n = a[0].size();
	bool isBlocked; // denote if the row is blocked
	for (int i = 0; i<m; ++i) {
		isBlocked = true;
		for (int j = 0; j < n; ++j) {
			int left = (j == 0 ? 0 : a[i][j - 1]);
			int top = (i == 0 ? 0 : a[i - 1][j]);
			if (i == 0 && j == 0 && a[i][j] == 0) a[i][j] = 1; // first box 1			
			else a[i][j] = (a[i][j] == 1 ? 0 : left + top); 
			if (a[i][j] > 0) isBlocked = false;
		}
		if (isBlocked) return 0;
	}
	return a[m - 1][n - 1];
}

// 73. Set Matrix Zeroes
void setZeroes(vector<vector<int> > &matrix) {
	int col0 = 1, rows = matrix.size(), cols = matrix[0].size();
	for (int i = 0; i < rows; ++i) {
		if (matrix[i][0] == 0) col0 = 0;
		for (int j = 1; j < cols; ++j)
			if (matrix[i][j] == 0) matrix[i][0] = matrix[0][j] = 0;
	}

	for (int i = rows - 1; i >= 0; --i) {
		for (int j = cols - 1; j >= 1; --j)
			if (matrix[i][0] == 0 || matrix[0][j] == 0) matrix[i][j] = 0;
		if (col0 == 0) matrix[i][0] = 0;
	}
}

// 280. Wiggle Sort
void wiggleSort(vector<int>& nums) {
	int n = nums.size();
	for (int i = 1; i < n; i++) // swap if: odd (i-1)>(i), even (i-1)<(i)
		if (((i & 1) && nums[i - 1]>nums[i]) || (!(i & 1) && nums[i - 1]<nums[i]))
			swap(nums[i], nums[i - 1]);
}



int main(){
	vector<int> nums{ 3, 5, 2, 1, 6, 4 };
	PRINT(nums);
	wiggleSort(nums);
	PRINT(nums);
	
	PRINT(count_lines());
	
	

	system("pause");
	return 0;
}  
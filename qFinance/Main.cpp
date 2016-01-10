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

// 35. Search Insertion Position
int searchInsert(vector<int>& nums, int target) {
	int low = 0, high = nums.size() - 1;
	while (low <= high) {
		int mid = low + (high - low) / 2;
		if (nums[mid] < target) low = mid + 1;
		else high = mid - 1;
	}
	return low;
}

// 33. Search in Rotated Sorted Array
int searchRotatedArray(vector<int>& nums, int target) {
	int start = 0, end = nums.size() - 1;
	while (start < end) {
		int mid = start + (end - start) / 2;
		if (nums[mid] == target) return mid;
		if (nums[mid] > nums[end]) {  // eg. 3,4,5,6,|1,2
			if (nums[start] <= target && target < nums[mid]) end = mid - 1;
			else start = mid + 1;
		}
		else {  // eg. 5,6,|1,2,3,4
			if (nums[mid] < target && target <= nums[end]) start = mid + 1;
			else end = mid - 1;
		}
	}
	return (nums[start] == target ? start : -1);
}

// 81. Search in Rotated Sorted Array II
bool searchRotatedArrayII(vector<int>& nums, int target) {
	int start = 0, end = nums.size() - 1;
	while (start < end) {
		int mid = start + (end - start) / 2;
		if (nums[mid] == target) return true;
		if (nums[mid] > nums[end]) {  // eg. 3,4,5,6,|1,2
			if (nums[start] <= target && target < nums[mid]) end = mid - 1;
			else start = mid + 1;
		}
		else if (nums[mid] < nums[end]) {  // eg. 5,6,|1,2,3,4
			if (nums[mid] < target && target <= nums[end]) start = mid + 1;
			else end = mid - 1;
		}
		else {
			--end; // need to make sure nums[mid]!=nums[end]
		}
	}
	return (nums[start] == target ? true : false);
}

// 34. Search for a Range
vector<int> searchRange(vector<int>& nums, int target) {
	vector<int> res(2, -1);
	int i = 0, j = nums.size() - 1;	
	// Search for the left one, while won't stop even when found
	while (i < j) {
		int mid = i + (j - i) / 2; // mid biased to the left
		if (nums[mid] < target) i = mid + 1;
		else j = mid;
	}
	if (nums[i] != target) return res;
	else res[0] = i;
	// Search for the right one, while won't stop even when found
	j = nums.size() - 1; // use the current i
	while (i < j) {
		int mid = i + (j - i) / 2 + 1; // mid biased to the right
		if (nums[mid] > target) j = mid - 1;
		else i = mid;
	}
	res[1] = j; // if the same as i, still use it
	return res;
}

// 74. Search a 2D Matrix
bool searchMatrix(vector<vector<int>>& matrix, int target) {
	int n = matrix.size(), m = matrix[0].size();
	int l = 0, r = m * n - 1;
	while (l < r){
		int mid = (l + r - 1) / 2;
		if (matrix[mid / m][mid % m] < target) l = mid + 1;
		else r = mid;
	}
	return matrix[r / m][r % m] == target;
}

// 240. Search a 2D Matrix II
bool searchMatrixII(vector<vector<int>>& matrix, int target) {
	if (matrix.empty() || matrix[0].empty()) return false;
	for (int i = 0, j = matrix[0].size() - 1; i < (int)matrix.size() && j >= 0; ) {
		if (matrix[i][j] == target) return true;
		if (matrix[i][j] < target) ++i;
		else --j;
	}
	return false;
}

// 48.1. Rotate Image
void rotate(vector<vector<int>>& matrix) {
	vector<vector<int>> result = matrix;
	for (size_t i = 0; i < matrix.size(); i++) {
		for (size_t j = 0; j < matrix.size(); j++) {
			result[j][matrix.size() - i - 1] = matrix[i][j];
		}
	}
	matrix = result;
}

// 48.2. Rotate Image
void rotate2(vector<vector<int>>& matrix) {
	reverse(matrix.begin(), matrix.end());
	for (size_t i = 0; i < matrix.size(); ++i)
		for (size_t j = i + 1; j < matrix[i].size(); ++j)
			swap(matrix[i][j], matrix[j][i]);
}

// 238. Product of Array Except Self
vector<int> productExceptSelf(vector<int>& nums) {
	int N = nums.size();
	vector<int> res(N, 1);	
	int left = 1, right = 1;
	for (int i = 0; i < N; ++i) {
		res[i] *= left;
		res[N - 1 - i] *= right;
		left *= nums[i];
		right *= nums[N - 1 - i];
	}
	return res;
}

// 39. Combination Sum
void combinationSum(vector<int>& candidates, int target,
	vector<vector<int> >& res, vector<int>& combination, int begin) {
	if (target == 0) { res.push_back(combination);	return; }
	for (int i = begin; i < (int)candidates.size() && candidates[i] <= target; ++i) {
		combination.push_back(candidates[i]);
		combinationSum(candidates, target - candidates[i], res, combination, i);
		combination.pop_back();
	}
}
vector<vector<int> > combinationSum(vector<int>& candidates, int target) {
	sort(candidates.begin(), candidates.end());
	vector<vector<int> > res;
	vector<int> combination;
	combinationSum(candidates, target, res, combination, 0);
	return res;
}

// 40. Combination Sum II
void combinationSum2(vector<int>& candidates, int target,
	vector<vector<int> >& res, vector<int>& combination, int begin) {
	if (target == 0) { res.push_back(combination); return; }
	for (int i = begin; i<(int)candidates.size() && candidates[i] <= target; ++i) {
		if (i == begin || candidates[i] != candidates[i - 1]) {
			combination.push_back(candidates[i]);
			combinationSum2(candidates, target - candidates[i],
				res, combination, i + 1);
			combination.pop_back();
		}
	}
}
vector<vector<int> > combinationSum2(vector<int>& candidates, int target) {
	sort(candidates.begin(), candidates.end());
	vector<std::vector<int> > res;
	vector<int> combination;
	combinationSum2(candidates, target, res, combination, 0);
	return res;
}

// 216. Combination Sum III
void combinationSum3(int need, int target, vector<vector<int>>& res,
	vector<int>& combination, int begin) {
	if (need == 0 && target == 0){ res.push_back(combination); return; }
	for (int i = begin; i < 10 && i <= target; i++){
		combination.push_back(i);
		combinationSum3(need - 1, target - i, res, combination, i + 1);
		combination.pop_back();
	}
}
vector<vector<int>> combinationSum3(int k, int n) {
	vector<vector<int>> res;
	vector<int> combination;
	combinationSum3(k, n, res, combination, 1);
	return res;
}









int main(){
	vector<int> nums{ 2, 3, 6, 7 };
	PRINT(combinationSum2(nums, 7));
	

	system("pause");
	return 0;
}  
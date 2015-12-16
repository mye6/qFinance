#include "Leetcode.h"


/* Section: Dynamic Programming */
int Solution::rob(vector<int>& nums) {
	int n = nums.size(), pre = 0, cur = 0;
	for (int i = 0; i < n; ++i) {
		int temp = max(pre + nums[i], cur);
		pre = cur;
		cur = temp;
	}
	return cur;
}

int Solution::rob_line(vector<int>& nums, int start, int end) {
	// the same as rob I
	int pre = 0, cur = 0;
	for (int i = start; i < end; ++i) {
		int temp = max(pre + nums[i], cur);
		pre = cur;
		cur = temp;
	}
	return cur;
}

int Solution::rob2(vector<int>& nums) {
	if (nums.size() == 0) return 0;
	else if (nums.size() == 1) return nums[0];
	// case 1: rob 0th, then cannot rob (n-1)th, the same as 0 to n-2
	// case 2: no rob 0th, then can rob (n-1)th, the same as 1 to n-1
	else return max(rob_line(nums, 0, nums.size() - 1), rob_line(nums, 1, nums.size()));
}

int Solution::maxSubArray(vector<int>& nums) {
	if (nums.size() == 0) return 0;
	int n = nums.size(), ans = nums[0], sum = 0;
	// ans[i] = max(sum[i-1]+nums[i], ans[i-1])
	// sum[i] = max(sum[i-1] + nums[i], 0)
	for (int i = 0; i < n; ++i) {
		sum += nums[i];
		ans = max(sum, ans);
		sum = max(sum, 0);
	}
	return ans;
}

int Solution::numWays(int n, int k) {
	if (n <= 1 || k == 0) return n*k;
	int s = k, d1 = k, d2 = k*(k - 1);
	for (int i = 2; i < n; ++i) {
		s = d2;
		d2 = (k - 1)*(d1 + d2);
		d1 = s;
	}
	return s + d2;
}

int Solution::climbStairs(int n) {
	if (n == 1 || n == 2) return n;
	// d2: #steps for i-2, d1: #steps for i-1
	// D[i] = D[i-2] + D[i-1]
	int d2 = 1, d1 = 2;
	for (int i = 3; i <= n; ++i) {
		int tmp = d2 + d1;
		d2 = d1; 
		d1 = tmp;
	}
	return d1;
}

int Solution::maxProfit(vector<int> &prices) {
	int minPrice = INT_MAX, maxPro = 0;
	for (size_t i = 0; i < prices.size(); ++i) {
		minPrice = min(prices[i], minPrice);
		// minPrice: lowest price in prices[0..i]
		maxPro = max(maxPro, prices[i] - minPrice);
		// maxPro: maximum price in prices[0..i]
	}
	return maxPro;
}

int Solution::numDecodings(string s) {
	if (s.size() == 0 || s.front() == '0') return 0;
	// r2: decode ways of s[i-2] , r1: decode ways of s[i-1] 
	int r1 = 1, r2 = 1;
	for (size_t i = 1; i < s.size(); ++i) {
		// zero voids ways of the last because zero cannot be used separately
		if (s[i] == '0') r1 = 0;
		// possible two-digit letter, so new r1 is sum of both while new r2 is the old r1
		if (s[i - 1] == '1' || s[i - 1] == '2' && s[i] <= '6') {
			r1 = r2 + r1;
			r2 = r1 - r2;
		}
		// one-digit letter, no new way added
		else {
			r2 = r1;
		}
	}
	return r1;
}

int Solution::uniquePathsWithObstacles(vector<vector<int>>& a) {
	int m = a.size();
	if (m == 0) return 0;
	int n = a[0].size();
	bool isBlocked; // denote if the row is blocked

	for (int i = 0; i<m; i++) {
		isBlocked = true;
		for (int j = 0; j < n; j++) {
			int left = (j == 0 ? 0 : a[i][j - 1]);
			int top = (i == 0 ? 0 : a[i - 1][j]);
			if (i == 0 && j == 0 && a[i][j] == 0) a[i][j] = 1;  // to make the first box  1
			else a[i][j] = (a[i][j] == 1 ? 0 : left + top); //update a[i][j] to the no of paths to a[i][j]

			if (a[i][j]>0) isBlocked = false;
		}
		if (isBlocked) return 0;
	}
	return a[m - 1][n - 1];
}

int Solution::numTrees(int n) {
	vector<int> d(n + 1);
	d[0] = d[1] = 1;
	for (int i = 2; i <= n; ++i) {
		d[i] = 0;
		for (int j = 1; j <= i; ++j) {
			d[i] += d[j - 1] * d[i - j];
		}
	}
	return d[n];
}

vector<TreeNode *> Solution::generateTree(int from, int to) {
	vector<TreeNode *> ret;
	if (to - from < 0) ret.push_back(0);
	if (to - from == 0) ret.push_back(new TreeNode(from));
	if (to - from > 0) {
		for (int i = from; i <= to; i++) {
			vector<TreeNode *> l = Solution::generateTree(from, i - 1);
			vector<TreeNode *> r = Solution::generateTree(i + 1, to);
			for (size_t j = 0; j < l.size(); j++) {
				for (size_t k = 0; k < r.size(); k++) {
					TreeNode * h = new TreeNode(i);
					h->left = l[j];
					h->right = r[k];
					ret.push_back(h);
				}
			}
		}
	}
	return ret;
}

void Solution::printTree(TreeNode* p, int indent) {
	if (p != NULL) {
		if (p->right) {
			printTree(p->right, indent + 4);
		}
		if (indent) {
			std::cout << std::setw(indent) << ' ';			
		}
		if (p->right) std::cout << " /\n" << std::setw(indent) << ' ';		
		cout << p->val << "\n ";
		if (p->left) {
			std::cout << std::setw(indent) << ' ' << " \\\n";			
			printTree(p->left, indent + 4);
		}
	}
}

vector<TreeNode *> Solution::generateTrees(int n) {
	if (n == 0) return vector<TreeNode*>(0);
	return generateTree(1, n);
}


/*
* Algorithm for the recursion:
* 1. If one of the node is NULL then return the equality result of p an q.
* This boils down to if both are NULL then return true,
* but if one of them is NULL but not the other one then return false
* 
* 2. At this point both root nodes represent valid pointers.
* Return true if three conditions are satisfied
* a. root nodes have same value, 
* b. the left tree of the roots are same (recursion)
* c. the right tree of the roots are same (recursion).
* Otherwise return f
alse.
*/
bool Solution::isSameTree(TreeNode* p, TreeNode* q) {
	if (p == NULL || q == NULL) return (p == q);
	return (p->val == q->val && isSameTree(p->left, q->left) && isSameTree(p->right, q->right));
}

NumArray::NumArray(vector<int> &nums) : psum(nums.size() + 1, 0){
	partial_sum(nums.begin(), nums.end(), psum.begin() + 1);
}
int NumArray::sumRange(int i, int j) { return psum[j + 1] - psum[i]; }

NumMatrix::NumMatrix(vector<vector<int>> &matrix) : accum(matrix) {
	for (size_t i = 0; i < matrix.size(); ++i)
		for (size_t j = 0; j < matrix[0].size(); ++j)
			accum[i][j] += a(i - 1, j) + a(i, j - 1) - a(i - 1, j - 1);
}

int NumMatrix::sumRegion(int row1, int col1, int row2, int col2) {
	return a(row2, col2) - a(row1 - 1, col2) - a(row2, col1 - 1) + a(row1 - 1, col1 - 1);
}

int NumMatrix::a(int i, int j) { return (i >= 0 && j >= 0) ? accum[i][j] : 0; }


/* Section: Binomial Tree */
void Solution::deepestLeftLeafUtil(TreeNode *root, int lvl, int& maxlvl, bool isLeft, TreeNode **resPtr) {
	// Base case
	if (root == NULL) return;
	// Update result if this node is left leaf and its level is more
	// than the maxl level of the current result
	if (isLeft && !root->left && !root->right && lvl > maxlvl) {
		*resPtr = root;
		maxlvl = lvl;
		return;
	}
	// Recur for left and right subtrees
	deepestLeftLeafUtil(root->left, lvl + 1, maxlvl, true, resPtr);
	deepestLeftLeafUtil(root->right, lvl + 1, maxlvl, false, resPtr);
}

// A wrapper over deepestLeftLeafUtil().
TreeNode* Solution::deepestLeftLeaf(TreeNode *root) {
	int maxlevel = 0;
	TreeNode *result = NULL;
	deepestLeftLeafUtil(root, 0, maxlevel, false, &result);
	return result;
}


/* Section: Math */
int Solution::mySqrt(int x) {
	if (x<2) return x; // to avoid mid = 0
	int low = 0, high = x, mid;
	while (low<high) {
		mid = (low + high) / 2;
		if (x / mid >= mid) low = mid + 1;
		else high = mid;
	}
	return high - 1;
};

/*Section: Array */

/*
0. boundary check: if size() is 0 or 1, return false;
1. sort the array, note it is pass by reference here; if pass by const reference, make a copy
2. check nums[i-1] with nums[i] for i = 1...N-1; if same, return true
*/
bool Solution::containsDuplicate(vector<int>& nums) {
	if (nums.size() == 0 || nums.size() == 1) return false;
	sort(nums.begin(), nums.end());
	for (size_t i = 1; i < nums.size(); ++i) {
		if (nums[i - 1] == nums[i]) return true;
	}
	return false;
}

bool Solution::containsDuplicate2(vector<int>& nums) {
	return set<int>(nums.begin(), nums.end()).size() < nums.size();
}

/*
1. define a boolean, carry = true
2. if (++digits[i] % 10) == 0 evaluates to be true, continue (i>=0&&carry)
3. if carry is true after the loop, insert 1 at the beginning
*/
vector<int> Solution::plusOne(vector<int>& digits) {
	int n = digits.size();
	bool carry = true;
	for (int i = n - 1; i >= 0 && carry; --i) {
		carry = ((++digits[i] % 10) == 0);
		if (carry) digits[i] = 0; // very important, convert from 10 to 0
	}
	if (carry) digits.insert(digits.begin(), 1); // insert at the beginning of a vector
	return digits;
}

/*
1. define idx1, idx2, dist to record location of word1, word2, and distance
2. update during the loop
*/
int Solution::shortestDistance(vector<string>& words, string word1, string word2) {
	int n(words.size()), idx1(-1), idx2(-1), dist(INT_MAX); // idx1, idx2 to record the location of word1 and word2
	for (int i = 0; i < n; ++i) {
		if (words[i] == word1) idx1 = i; // evaluate the equivalence of two strings, s1 == s2
		else if (words[i] == word2) idx2 = i;
		if (idx1 != -1 && idx2 != -1) dist = min(dist, abs(idx1 - idx2)); // when both index were detected, record dist
	}
	return dist;
}

// rotate by reverse. time complexity: O(N); space complexity: O(1)
void Solution::rotate(vector<int>& nums, int k) {
	int n = nums.size();
	k = k%n;
	reverse(nums.begin(), nums.begin() + n); // reverse all numbers
	reverse(nums.begin(), nums.begin() + k); // reverse 0..k-1 numbers
	reverse(nums.begin() + k, nums.begin() + n); // reverse k..n-1 numbers
}

// rotate by copying. time complexity: O(N); space complexity: O(1) 
void Solution::rotate2(vector<int>& nums, int k) {
	int n = nums.size();
	if (n == 0 || k <= 0) return; // boundary check
	vector<int> tmp(nums); // use copy constructor to copy the vector
	for (int i = 0; i < n; ++i) nums[(i + k) % n] = tmp[i]; // 
}

// remove val elements using a tracker index, i; fill ith element only by non-val element
int Solution::removeElement(vector<int>& nums, int val) {
	int i = 0; // track the non-val elements
	for (size_t j = 0; j < nums.size(); ++j) {
		if (nums[j] != val) nums[i++] = nums[j];
	}
	return i;
}
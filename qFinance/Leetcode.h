#ifndef LEETCODE_H
#define LEETCODE_H
#include "Solver.h"

// Definition for a binary tree node.
struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
	/* Section: Dynamic Programming */
	static int rob(vector<int>& nums); // House Robber
	static int rob_line(vector<int>& nums, int start, int end); // House Robber 2 Helper
	static int rob2(vector<int>& nums); // House Robber 2
	static int numWays(int n, int k); // Paint Fence
	static int climbStairs(int n); // Climbing Stairs
	static int maxProfit(vector<int> &prices); // Best Time to Buy and Sell Stock
	static int numDecodings(string s); // Decode Ways
	static int uniquePathsWithObstacles(vector<vector<int>>& a); // Unique Paths
	static int numTrees(int n); // Unique Binary Search Trees
	static vector<TreeNode *> generateTree(int from, int to);
	static vector<TreeNode *> generateTrees(int n);
	static int maxSubArray(vector<int>& nums);
	static bool isSameTree(TreeNode* p, TreeNode* q);

	/* Section: Binomial Tree */
	static void deepestLeftLeafUtil(TreeNode *root,
		int lvl, int& maxlvl, bool isLeft, TreeNode **resPtr);
	static TreeNode* deepestLeftLeaf(TreeNode *root);
	
	static void printTree(TreeNode* p, int indent = 3);

	/* Section: Math*/
	static int mySqrt(int x);

	/*Section: Array */
	static bool containsDuplicate(vector<int>& nums);
	static vector<int> plusOne(vector<int>& digits);
	static int shortestDistance(vector<string>& words, string word1, string word2);
	static void rotate(vector<int>& nums, int k);
	static void rotate2(vector<int>& nums, int k);
};

class NumArray {
public:
	NumArray(vector<int> &nums);
	int sumRange(int i, int j);
private:
	vector<int> psum;
	// psum[i] records the sum of nums[0..i-1]
};

class NumMatrix {
public:
	NumMatrix(vector<vector<int>> &matrix);
	// a(i-1, j) + a(i, j-1) - a(i-1, j-1) computes the sum of matrix[0..i][0..j] except matrix[i][j]
	int sumRegion(int row1, int col1, int row2, int col2);
private:	
	vector<vector<int> > accum;
	// accum[i][j] is the sum of matrix[0..i][0..j]
	
	int a(int i, int j);
	// a(i, j) helps with edge cases
};



#endif


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

struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}
	~ListNode() {
		cout << "[" << val << "] removed" << endl;
	}
};

struct TreeLinkNode {
	int val;
	TreeLinkNode *left, *right, *next;
	TreeLinkNode(int x) : val(x), left(NULL), right(NULL), next(NULL) {}
};


struct RandomListNode {
	int label;
	RandomListNode *next, *random;
	RandomListNode(int x) : label(x), next(NULL), random(NULL) {}
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
	static bool containsDuplicate(vector<int>& nums); // sorting
	static bool containsDuplicate2(vector<int>& nums); // set, or hash map
	static vector<int> plusOne(vector<int>& digits);
	static int shortestDistance(vector<string>& words, string word1, string word2);
	static void rotate(vector<int>& nums, int k);
	static void rotate2(vector<int>& nums, int k);
	static int removeElement(vector<int>& nums, int val);

	/*Section: Hash Table*/
	static bool wordPattern(string pattern, string str);
	
	/*Section: Stack */
	static string removeDuplicateLetters(string s); // 316
	static int evalRPN(vector<string>& tokens); // 150
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

// uses an un_ordered map to record the abbreviations
class ValidWordAbbr {
public:
	ValidWordAbbr(vector<string> &dictionary);
	bool isUnique(string word);
private:
	unordered_map<string, unordered_set<string>> mp; // performance, unordered_map, O(1) complexity
};

/* Section: Stack */
class Queue {
	stack<int> input, output;
public:
	void push(int x);
	void pop(void);
	int peek(void);
	bool empty(void);
};

class MinStack {
private:
	stack<int> s1, s2;
public:
	void push(int x);
	void pop();
	int top();
	int getMin();
};

class Stack {
private:
	queue<int> q;
public:
	void push(int x);
	void pop();
	int top();
	bool empty();
};

/* Section: Linked List*/
ListNode* genList(const vector<int>& nums);

ostream& operator<<(ostream& os, ListNode* head);

void clear(ListNode* head);

ListNode* reverseList(ListNode* head);

ListNode* removeNthFromEnd(ListNode* head, int n);

ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);

int countNodes(ListNode* head);

bool isPalindrome(ListNode* head);

ListNode* deleteDuplicates(ListNode* head);

ListNode* deleteDuplicates2(ListNode* head);

ListNode* removeElements(ListNode* head, int val);

ListNode* findTail(ListNode* head);

ListNode *getIntersectionNode(ListNode *headA, ListNode *headB);

ListNode* insertionSortList(ListNode* head);

ListNode* sortList(ListNode* head);

ListNode* swapPairs(ListNode* head);

ListNode* partition(ListNode* head, int x);

// 141. Linked List Cycle
bool hasCycle(ListNode* head);

// 142. Linked List Cycle II
ListNode* detectCycle(ListNode* head);

/*Section: Array*/
// 217
bool containsDuplicate(vector<int>& nums);

// 217
bool containsDuplicate2(vector<int>& nums);

// 219
bool containsNearbyDuplicate(vector<int>& nums, int k);

// 287
int findDuplicate3(vector<int>& nums);

// 26 remove duplicates from sorted array
int removeDuplicates(vector<int>& nums);

// 66 plus one
vector<int> plusOne(vector<int>& digits);

// 243 shortest distance
int shortestDistance(vector<string>& words, string word1, string word2);

// 245 shortest distance III
int shortestDistance3(vector<string>& words, string word1, string word2);

// 244 shortest word distance II
class wordDistance {
private:
	unordered_map<string, vector<int>> wmap;
public:
	wordDistance(vector<string>& words);
	int shortest(string word1, string word2);
};

// 189
void rotate(vector<int>& nums, int k);

// 189.2
void rotate2(vector<int>& nums, int k);

// 153
int findMin(vector<int>& nums);

// 325. Maximum Size Subarray Sum Equals k
int maxSubArrayLen(vector<int>& nums, int k);

/*Section: Tree*/
// 297. Serialize and Deserialize Binary Tree, preorder approach
class Codec {
private:
	void serialize(TreeNode* root, ostringstream& out);
	TreeNode* deserialize(istringstream& in);
public:
	string serialize(TreeNode* root);
	TreeNode* deserialize(string data);
};

// 108 Convert Sorted Array to Binary Search Tree
TreeNode* sortedArrayToBST(int start, int end, vector<int>& nums);
TreeNode* sortedArrayToBST(vector<int>& nums);

// 226. Invert Binary Tree
TreeNode* invertTree(TreeNode* root);

// 114. Flatten Binary Tree to Linked List
void flatten(TreeNode* root);

// 156. Binary Tree Upside Down
TreeNode* upsideDownBinaryTree(TreeNode* root);

// 144. Binary Tree Preorder Traversal, recursive
void preorderTraversal(TreeNode* root, vector<int>& nodes);
vector<int> preorderTraversal(TreeNode* root);

// 144. Binary Tree Preorder Traversal, iterattive
vector<int> preorderTraversal2(TreeNode* root);

// 94 Binary Tree Inorder Traversal, recursive
void inorderTraversal(TreeNode* root, vector<int>& nodes);
vector<int> inorderTraversal(TreeNode* root);

// 94. Binary Tree Inorder Traversal, iterative
vector<int> inorderTraversal2(TreeNode* root);

// 173. Binary Search Tree Iterator
class BSTIterator {
private:
	stack<TreeNode *> myStack;
	void pushAll(TreeNode *node);
public:
	BSTIterator(TreeNode *root);
	
	/** @return whether we have a next smallest number */
	bool hasNext();

	/** @return the next smallest number */
	int next();
};

// 94 Binary Tree Postorder Traversal
void postorderTraversal(TreeNode* root, vector<int>& nodes);
vector<int> postorderTraversal(TreeNode* root);

// 102 Binary Tree Level Order Traversal
void levelOrder(TreeNode* root, int level, vector<vector<int> >& res);
vector<vector<int> > levelOrder(TreeNode(*root));

// 107 Binary Tree Level Order Traversal II
void levelOrderBottom(TreeNode* root, int level, vector<vector<int> >& res);
vector<vector<int> > levelOrderBottom(TreeNode(*root));

// 103 Binary Tree Zigzag Level Order Traversal II
void zigzagLevelOrder(TreeNode* root, int level, vector<vector<int> >& res);
vector<vector<int> > zigzagLevelOrder(TreeNode(*root));

// 199 Binary Tree Right Side View
void rightSideView(TreeNode* root, int level, vector<int>& res);
vector<int> rightSideView(TreeNode* root);

// Self. cout treeNode
void levelPrint(TreeNode* root, int level, vector<vector<string> >& res);
ostream& operator<<(ostream& os, TreeNode* root);

// 104. Maximum Depth of Binary Tree, recursive
int maxDepth(TreeNode* root);

// 104. Maximum Depth of Binary Tree, iterative
int maxDepth2(TreeNode* root);

// 111. Minimum Depth of Binary Tree, recursive
int minDepth(TreeNode* root);

// 111. Minimum Depth of Binary Tree, iterative
int minDepth2(TreeNode* root);

// 110. Balanced Binary Tree
int height(TreeNode *root);
bool isBalanced(TreeNode* root);

// 100. Same Tree
bool isSameTree(TreeNode* p, TreeNode* q);

// 222. Count Complete Tree Nodes
int countNodes(TreeNode* root);

// 270. Closest Binary Search Tree Value, iterative
int closestValue2(TreeNode* root, double target);

// 270. Closest Binary Search Tree Value, recursive
void closestValue(TreeNode* node, double target, double &result);
int closestValue(TreeNode* root, double target);

// 112. Path Sum
bool hasPathSum(TreeNode* root, int sum);

// 113. Path Sum II
void pathSum(TreeNode* node, int sum, vector<int>& path, vector<vector<int> >& paths);
vector<vector<int> > pathSum(TreeNode* root, int sum);

// 257. Binary Tree Paths
void binaryTreePaths(TreeNode* root, string s, vector<string>& res);
vector<string> binaryTreePaths(TreeNode* root);

// 129. Sum Root to Leaf Numbers
void sumNumbers(TreeNode* node, int csum, int& sum);
int sumNumbers(TreeNode* root);

// 101. Symmetric Tree
bool isSymmetric(TreeNode* left, TreeNode* right);
bool isSymmetric(TreeNode* root);

// 101. Symmetric Tree
bool isSymmetric2(TreeNode* root);

// 98. Validate Binary Search Trees
bool isValidBST(TreeNode* node, long long min, long long max);
bool isValidBST(TreeNode* root);

// 230. Kth Smallest Element in a BST
void kthSmallest(TreeNode* root, int& k, int& res);
int kthSmallest(TreeNode* root, int k);

// 230. Inorder Successor in BST
TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p);

// 235. Lowest Common Ancestor of a Binary Search Tree
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);

// 235. Lowest Common Ancestor of a Binary Search Tree
TreeNode* lowestCommonAncestor2(TreeNode* root, TreeNode* p, TreeNode* q);

// 236. Lowest Common Ancestor of a Binary Tree
TreeNode* lowestCommonAncestorBT(TreeNode* root, TreeNode* p, TreeNode* q);

// 95. Unique Binary Search Trees I
int numTrees(int n);

// 96. Unique Binary Search Trees II
vector<TreeNode*> generateTrees(int left, int right);
vector<TreeNode*> generateTrees(int n);

/*Section: Hash Table*/
// 1. Two Sum
vector<int> twoSum(vector<int>& nums, int target);

// 167. Two Sum II - Input Array is Sorted
vector<int> twoSumII(vector<int>& nums, int target);

// 15. 3 Sum
vector<vector<int> > threeSum(vector<int> &nums);

// 18. 4 Sum
vector<vector<int>> fourSum(vector<int>& nums, int target);

// 16. 3Sum Closest
int threeSumClosest(vector<int>& nums, int target);

// 170. Two Sum III - Data Structure Design
class TwoSum {
	unordered_map<int, int> map;
public:
	void add(int number);
	bool find(int value);
};

// 299. Bulls and Cows
string getHint(string secret, string guess);

// 202. Happy Number
int digitSquareSum(int n);
bool isHappy(int n);

// 242. Valid Anagram
bool isAnagram(string s, string t);

// 242. Valid Anagram
bool isAnagram2(string s, string t);

// 249. Group Shifted Strings
string shift(string s);
vector<vector<string>> groupStrings(vector<string>& strings);

// 49. Group Anagram
vector<vector<string>> groupAnagrams(vector<string>& strs);

// 205. Isomorphic Strings
bool isIsomorphic(string s, string t);

// 3. Longest Substring Without Repeating Characters
int lengthOfLongestSubstring(string s);

// 36. Valid Sudoku
bool isValidSudoku(vector<vector<char> >& board);

// 266. Palindrome Permutation
bool canPermutePalindrome(string s);

// 246. Strobogrammatic Number
bool isStrobogrammatic(string num);

// 247. Strobogrammatic Number II
vector<string> findStrobogrammatic(int n, int m);
vector<string> findStrobogrammatic(int n);

// 314. Binary Tree Vertical Order
vector<vector<int>> verticalOrder(TreeNode* root);

// 274. H-Index
int hIndex(vector<int>& citations);

// 275. H-Index II
int hIndexII(vector<int>& citations);

// 204. Count Primes
int countPrimes(int n);

// 311. Sparse Matrix Multiplication
vector<vector<int>> multiply(vector<vector<int>>& A, vector<vector<int>>& B);

// 187. Repeated DNA Sequences
int str2int(string s);
vector<string> findRepeatedDnaSequences(string s);

// 166. Fraction to Recurring Decimal
string fractionToDecimal(int numerator, int denominator);

// 138. Copy List with Random Pointers
RandomListNode *copyRandomList(RandomListNode *head);

/*Section: Math*/
// 231. Power of Two, bit
bool isPowerOfTwo0(int n);

// 231. Power of Two, map
bool isPowerOfTwo1(int n);

// 231. Power of Two, module
bool isPowerOfTwo2(int n);

// 231. Power of Two, log
bool isPowerOfTwo3(int n);

// 231. Power of Two, iterative
bool isPowerOfTwo4(int n);

// 231. Power of Two, recursive
bool isPowerOfTwo5(int n);

// 326. Power of Three, module
bool isPowerOfThree1(int n);

// 326. Power of Three, log
bool isPowerOfThree2(int n);

// 326. Power of Three, iterative O(h), n=3^h
bool isPowerOfThree3(int n);

// 326. Power of Three, recursive
bool isPowerOfThree4(int n);

// 67. Add Binary
string addBinary0(string a, string b);

// 8. String to Integer (atoi)
int atoi0(const char *str);

// 223. Rectangle Area
int computeArea(int A, int B, int C, int D, int E, int F, int G, int H);

// 263. Ugly Number
bool isUgly(int num);

// 264. Ugly Number II
int nthUglyNumber(int n);

// 313. Super Ugly Number
int nthSuperUglyNumber(int n, vector<int>& primes);

// 2. Add Two Numbers
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);

// 43. Multiply Strings
string multiply(string num1, string num2);

// 258. Add Digits, conditions
int addDigits(int num);

// 258. Add Digits, simplication
int addDigits2(int num);

// 172. Factorial Trailing Zeroes, O(log2(N)), n = #factor-5's
// 100--> 24, 5^2. 2 loops
int trailingZeroes(int n);

// 168. Excel Sheet Column Title
string convertToTitle(int n);

// 171. Excel Sheet Column Number
int titleToNumber(string s);

// 9. Palindrome Number
bool isPalindrome(int x);

// 7. Reverse Integer
int reverse(int x);

// 13. Roman to Integer
int romanToInt(string s);

// 22. Integer to Roman
string intToRoman(int num);

// 273. Integer to English Words
string int_string(int n);
string numberToWords(int n);

// 279. Perfect Squares
int numSquares(int n);

// 268. Missing Number
int missingNumber(vector<int>& nums);

// 233. Number of Digit One
long long currDigitOne(long long left, long long right, long long digit);
int countDigitOne(int n);

// 224. Basic Calculator
int calculate(string s);

// 150. Evaluate Reverse Polish Notation
int evalRPN(vector<string>& tokens);

// 69. Sqrt(x)
int mySqrt(int x);

// 319. Bulb Switcher
int bulbSwitch(int n);

// 29. Divide Two Integers
int divide(int dividend, int divisor);

// 50. Pow(x, n), recursive
double mypow(double x, int n);

// 50, Pow(x, n), iterative
double myPow2(double x, int n);

// calculate e^x using taylor expansion
double exp_taylor2(double x);

// Fibonacci sequence, hard coded, O(1)
long long Fib1(int n);

// Fibonacci sequence, matrix, O(logN)
void multiply(long long F[2][2], long long M[2][2]);
void power(long long F[2][2], int n);
long long Fib2(int n);

// Fibonacci sequence, static array, O(N)
long long Fib3(int n);

// Fibonacci sequence, formula, O(logN)
long long Fib4(int n);

// Fibonacci sequence, recursion, O(2^N)
long long Fib5(int n);

// Factorial sequence, static array, O(1)
long long Fac1(int n);

// Factorial sequence, static array, O(N)
long long Fac2(int n);

/*Section: Array*/
// 80. Remove Duplicates from Sorted Array II
int removeDuplicatesII(vector<int>& nums);

// 27. Remove Element
int removeElement(vector<int>& nums, int val);

// 118. Pascal's Triangle
vector<vector<int> > generatePascal(int numRows);

// 119. Pascal's Triangle II
vector<int> getRow(int rowIndex);

// 283. Move Zeroes
void moveZeroes(vector<int>& nums);

// 88. Merge Sorted Array
void mergeSortedArray(vector<int>& nums1, int m, vector<int>& nums2, int n);

// 169.1 Majority Element, hash table : value->count.O(N)
int majorityElement1(vector<int>& nums);

// 169.2 Majority Element, Moore Voting Algorithm, O(N)
int majorityElement2(vector<int>& nums);

// 169.3 Majority Element, sort and return the mid element : O(NlogN)
int majorityElement3(vector<int>& nums);

// 169.3 Majority Element, sort and count length.O(NlogN)
int majorityElement4(vector<int>& nums);

// 229. Majority Element II
vector<int> majorityElementII(vector<int>& nums);

// 229. Majority Element II
vector<int> majorityElement(vector<int>& nums);

// 62.1. Unique Paths
int uniquePaths(int m, int n);

// 62.2. Unique Paths
int combination(int m, int n);
int uniquePaths2(int m, int n);

// 63. Unique Paths II
int uniquePathsWithObstacles(vector<vector<int>>& a);

// 73. Set Matrix Zeroes
void setZeroes(vector<vector<int> > &matrix);

// 280. Wiggle Sort
void wiggleSort(vector<int>& nums);

// 35. Search Insertion Position
int searchInsert(vector<int>& nums, int target);

// 33. Search in Rotated Sorted Array
int searchRotatedArray(vector<int>& nums, int target);

// 81. Search in Rotated Sorted Array II
bool searchRotatedArrayII(vector<int>& nums, int target);

// 34. Search for a Range
vector<int> searchRange(vector<int>& nums, int target);

// 74. Search a 2D Matrix
bool searchMatrix(vector<vector<int>>& matrix, int target);

// 240. Search a 2D Matrix II
bool searchMatrixII(vector<vector<int>>& matrix, int target);

// 48.1. Rotate Image
void rotate(vector<vector<int>>& matrix);

// 48.2. Rotate Image
void rotate2(vector<vector<int>>& matrix);

// 238. Product of Array Except Self
vector<int> productExceptSelf(vector<int>& nums);

// 39. Combination Sum
void combinationSum(vector<int>& candidates, int target,
	vector<vector<int> >& res, vector<int>& combination, int begin);
vector<vector<int> > combinationSum(vector<int>& candidates, int target);

// 40. Combination Sum II
void combinationSum2(vector<int>& candidates, int target,
	vector<vector<int> >& res, vector<int>& combination, int begin);
vector<vector<int> > combinationSum2(vector<int>& candidates, int target);

// 216. Combination Sum III
void combinationSum3(int need, int target, vector<vector<int>>& res,
	vector<int>& combination, int begin);
vector<vector<int>> combinationSum3(int k, int n);

// 78.1. Subsets, Recursive
void genSubsets(vector<int>& nums, int start, vector<int>& sub, vector<vector<int>>& subs);
vector<vector<int>> subsets(vector<int>& nums);

// 78.2. Subsets, Iterative
vector<vector<int>> subsets2(vector<int>& nums);

// 90.1. Subsets II, recursive
void subsetsWithDup(vector<int>& nums, int begin, vector<int> &sub, vector<vector<int> >& subs);
vector<vector<int> > subsetsWithDup(vector<int>& nums);

// 90.2. Subsets II, iterative
vector<vector<int>> subsetsWithDup2(vector<int>& nums);

// 31. Next Permutation
void nextPermutation(vector<int>& nums);

// 163. Missing Ranges
string get_range(int start, int end);
vector<string> findMissingRanges(vector<int>& nums, int lower, int upper);

// 259. 3Sum Smaller
int threeSumSmaller(vector<int>& nums, int target);

// 209. Minimum Size Subarray Sum
int minSubArrayLen(int s, vector<int>& nums);

// 64. Minimum Path Sum
int minPathSum(vector<vector<int>>& grid);

// 79. Word Search
bool wordSearch(vector<vector<char>>& board, string& word,
	int row, int col, int begin, int M, int N, int len);
bool wordSearch(vector<vector<char>>& board, string word);

// 55. Jump Game
bool canJump(vector<int>& nums);

// 45. Jump Game II
int jump(vector<int>& nums);

// 289. Game of Life
void gameOfLife(vector<vector<int>>& board);

// 54. Spiral Matrix
vector<int> spiralOrder(vector<vector<int>>& matrix);

// 59. Spiral Matrix II
vector<vector<int> > generateMatrix(int n);

// 162. Find Peak Element
int findPeakElement(vector<int>& nums);

// 75. Sort Colors
void sortColors(vector<int>& nums);

// 11. Container with Most Water
int maxArea(vector<int>& height);


/*Section: String*/
// 14. Longest Common Prefix
string longestCommonPrefix(vector<string>& strs);

// 20. Valid Parentheses
bool isValidParentheses(string s);

// 165. Compare Version Numbers
int compareVersion(string version1, string version2);

// 38. Count and Say
string countSay(string s);
string countAndSay(int n);

// 58. Length of Last Word
int lengthOfLastWord(string s);

// 6. ZigZag Conversion
string convertZigZag(string s, int numRows);

// 125. Valid Palindrome
bool isPalindrome(string s);

// 293. Flip Game
vector<string> generatePossibleNextMoves(string s);

// 294. Flip Game II
bool canWin(string s);

// 28. Implement strStr()
int strStr(string haystack, string needle);

// 93. Restore IP Address
bool isValidIPAddress(string s);
vector<string> restoreIpAddresses(string s);

// 5. Longest Palindrome Substring
string longestPalindrome(string s);

// 17. Letter Combinations of a Phone Number
void letterCombinations(string& digits, int i, string s, vector<string>& res);
vector<string> letterCombinations(string digits);

// 151. Reverse Words in a String
void reverseWords(string &s);

// 186. Reverse Words in a String II
void reverseWordsII(string &s);

// 161. One Edit Distance
bool isOneEditDistance(string s, string t);

// 22. Generate Parentheses
void generateParenthesis(int n, int m, string s, vector<string>& res);
vector<string> generateParenthesis(int n);

// 271. Encode and Decode a String
string encodeToString(vector<string>& strs);
vector<string> decodeToVec(string s);

// 71. Simplify Path
string simplifyPath(string path);

// 91. Deconde ways
int numDecodings(string s);

// 227. Basic Calculator II
int calculateII(string s);


/*Section: Backtracking*/
// 77. Combinations
void combine(int begin, int n, int pos, int k,
	vector<int>& com, vector<vector<int> >& coms);
vector<vector<int> > combine(int n, int k);

// 254. Factor Combinations
vector<vector<int>> getFactors(int n, int k);
vector<vector<int>> getFactors(int n);

// 320. Generalized Abbreviation
void generateAbbreviations(bool prevNum, int i, string abbr, vector<string>& res, string& word);
vector<string> generateAbbreviations(string word);

// 267. Palindrome Permutations II
vector<string> permutations(string& s);
vector<string> generatePalindromes(string s);

// 46. Permutations
void permute(int pos, vector<int>& nums, vector<vector<int>>& res);
vector<vector<int>> permute(vector<int>& nums);

// 47. Permutations II
void permuteUnique(int pos, vector<int> nums, vector<vector<int>>& res);
vector<vector<int>> permuteUnique(vector<int>& nums);

// 60. Permutation Sequence
string getPermutation(int n, int k);

// 131. Palindrome Partitioning
void partitionPalindrome(int pos, vector<string>& t, vector<vector<string>>& res, string& s);
vector<vector<string> > partitionPalindrome(string s);

// 89. Gray Code
vector<int> grayCode(int n);

/*Section: Dynamic Programming*/
//276. Paint Fence
int numWays(int n, int k);

// 256. Paint House
int minCost(vector<vector<int>>& costs);

// 198. House Robber
int rob(vector<int>& nums);

// 213. House Robber II
int rob_line(vector<int>& nums);
int rob2(vector<int>& nums);

// 70. Climbing Stairs
int climbStairs(int n);

// 121. Best Time to Buy and Sell Stock
int maxProfit(vector<int>& prices);

// 122. Best Time to Buy and Sell Stock II
int maxProfitII(vector<int> &prices);

// 123. Best Time to Buy and Sell Stock III
int maxProfitIII(vector<int>& prices);

// 309. Best Time to Buy and Sell Stock with Cooldown
int maxProfitIV(vector<int>& prices);

// 120. Triangle
int minimumTotal(vector<vector<int>>& triangle);

// 53. Maximum Subarray
int maxSubArray(vector<int>& nums);

// 152. Maximum Product Subarray
int maxProduct(vector<int>& A);

// 300.1. Longest Increasing Subsequence, DP, O(N^2)
int lengthOfLIS1(vector<int>& nums);

// 300.2. Longest Increasing Subsequence, Online, O(NlogN)
int lengthOfLIS_2(vector<int>& nums);

//221. Maximal Square
int maximalSquare(vector<vector<char>>& matrix);

// 322. Coin Change
int coinChange(vector<int>& coins, int amount);

/*Section: Depth First Search and Breadth First Search*/
// 200.1. Number of Islands, DFS
void numIslandsDFS(vector<vector<char>>& grid, int i, int j);
int numIslands(vector<vector<char>>& grid);

// 200.2. Number of Islands, BFS
void numIslandsBFS(vector<vector<char>>& grid, int x, int y);
int numIslands2(vector<vector<char>>& grid);

// 130. Surrounded Regions, BFS
void bfs(int i, int j, vector<vector<char>>& board);
void solve(vector<vector<char>>& board);


#endif


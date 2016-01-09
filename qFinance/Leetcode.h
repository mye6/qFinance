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




#endif


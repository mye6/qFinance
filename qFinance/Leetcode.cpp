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

/*Section: Hash Table*/
bool Solution::wordPattern(string pattern, string str) {
	map<char, int> p2i; // map char to int
	map<string, int> w2i; // map string to int
	istringstream in(str); // parse the word strings
	int i = 0, n = pattern.size();
	for (string word; in >> word; ++i) {
		if (p2i[pattern[i]] != w2i[word] || i == n)
			return false; // if str is longer, or no match, return with false, before recording
		p2i[pattern[i]] = w2i[word] = i + 1; // record each char/string mapping
	}
	return i == n;
}


ValidWordAbbr::ValidWordAbbr(vector<string> &dictionary) {
	for (string& d : dictionary) {
		int n = d.length();
		string abbr = d[0] + to_string(n) + d[n - 1];
		mp[abbr].insert(d);
	}
}

bool ValidWordAbbr::isUnique(string word) {
	int n = word.length();
	string abbr = word[0] + to_string(n) + word[n - 1];
	return mp[abbr].count(word) == mp[abbr].size();
}

/* Section: Stack*/
void Queue::push(int x) { input.push(x); }
void Queue::pop(void) { peek(); output.pop(); }
int Queue::peek(void) {
	if (output.empty()){
		while (!input.empty()) {
			output.push(input.top());
			input.pop();
		}
	}
	return output.top();
}
bool Queue::empty() {
	return input.empty() && output.empty();
}

void MinStack::push(int x) {
	s1.push(x);
	if (s2.empty() || x <= getMin()) s2.push(x);
	// s2 empty or x<=getMin() need to add to s2
}
void MinStack::pop() {
	if (s1.top() == getMin()) s2.pop();
	s1.pop();
}
int MinStack::top() {
	return s1.top();
}
int MinStack::getMin() {
	return s2.top();
}

void Stack::push(int x) {
	q.push(x);
	for (int i = 1; i < (int)q.size(); ++i) {
		q.push(q.front()); q.pop();
	}
}
void Stack::pop() { q.pop(); }
int Stack::top() { return q.front(); }
bool Stack::empty() { return q.empty(); }

string Solution::removeDuplicateLetters(string s) {
	vector<int> count(256, 0);
	vector<bool> visited(256, false);
	for (char c : s) ++count[c];
	string result = "0";
	for (char c : s) {
		--count[c];
		if (visited[c]) continue;
		while (c < result.back() && count[result.back()] > 0) {
			visited[result.back()] = false;
			result.pop_back();
		}
		result += c;
		visited[c] = true;
	}
	return result.substr(1);
}

int Solution::evalRPN(vector<string>& tokens) {
	stack<int> stk;
	for (string s : tokens) {
		if (s.size() > 1 || isdigit(s[0])) stk.push(stoi(s));
		else {
			int x2 = stk.top(); stk.pop();
			int x1 = stk.top(); stk.pop();
			switch (s[0]) {
			case '+': x1 += x2; break;
			case '-': x1 -= x2; break;
			case '*': x1 *= x2; break;
			case '/': x1 /= x2; break;
			}
			stk.push(x1);
		}
	}
	return stk.top();
}

/*Section: Linked List*/
ListNode* genList(const vector<int>& nums) {
	ListNode *head = NULL, *tail = NULL;
	for (size_t i = 0; i < nums.size(); ++i) {
		ListNode* node = new ListNode(nums[i]);
		if (i == 0) head = (tail = node);
		else tail = (tail->next = node);
	}
	return head;
}

ostream& operator<<(ostream& os, ListNode* head) {
	for (ListNode* cur = head; cur; cur = cur->next) {
		os << cur->val << "->";
	}
	os << "#";
	return os;
}

void clear(ListNode* head) {
	ListNode* tmp;
	for (ListNode* cur = head; cur; cur = tmp) {
		tmp = cur->next;
		delete cur;
	}
}

ListNode* reverseList(ListNode* head) {
	ListNode dummy(0);
	ListNode* tail = NULL, *tmp;
	for (ListNode* cur = head; cur; cur = tmp) {
		tmp = cur->next;
		cur->next = tail;
		tail = cur;
		dummy.next = cur;
	}
	return dummy.next;
}

ListNode* removeNthFromEnd(ListNode* head, int n) {
	ListNode dummy(0); dummy.next = head;
	ListNode *t1 = &dummy, *t2 = &dummy;
	for (int i = 0; i < n; ++i) t2 = t2->next;
	while (t2->next) { t1 = t1->next; t2 = t2->next; }
	t1->next = t1->next->next;
	return dummy.next;
}

ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
	ListNode dummy(INT_MIN);
	ListNode* tail = &dummy;
	while (l1 && l2) {
		if (l1->val < l2->val) {
			tail->next = l1;
			l1 = l1->next;
		}
		else {
			tail->next = l2;
			l2 = l2->next;
		}
		tail = tail->next;
	}
	tail->next = (l1 ? l1 : l2);
	return dummy.next;
}

int countNodes(ListNode* head) {
	int res = 0;
	for (ListNode* cur = head; cur; cur = cur->next) {
		++res;
	}
	return res;
}

bool isPalindrome(ListNode* head) {
	ListNode *sp = head, *fp = head, *revp = NULL, *tmp;
	while (fp && fp->next){
		fp = fp->next->next;
		tmp = sp->next;
		sp->next = revp;
		revp = sp;
		sp = tmp;
	}
	if (fp) sp = sp->next;
	while (sp && revp){
		if (sp->val != revp->val) return false;
		sp = sp->next;
		revp = revp->next;
	}
	return true;
}

// 83. Remove Duplicate from Sorted List
ListNode* deleteDuplicates(ListNode* head) {
	ListNode* cur = head;
	while (cur) {
		while (cur->next && cur->val == cur->next->val)
			cur->next = cur->next->next;
		cur = cur->next;
	}
	return head;
}

// 82. Remove Duplicate from Sorted List II
ListNode* deleteDuplicates2(ListNode* head) {
	if (head == NULL) return NULL;
	ListNode dummy(head->val - 1);
	ListNode *tail = &dummy, *pre = &dummy;
	while (head) {
		if (head->val != pre->val && (head->next == NULL || head->next->val != head->val)) {
			tail->next = head;
			tail = tail->next;
		}
		pre = head;
		head = head->next;
	}
	tail->next = NULL;
	return dummy.next;
}

ListNode* removeElements(ListNode* head, int val) {
	ListNode dummy(0); dummy.next = head;
	ListNode *cur = &dummy;
	while (cur){
		while (cur->next && cur->next->val == val) cur->next = cur->next->next;
		cur = cur->next;
	}
	return dummy.next;
}

ListNode* findTail(ListNode* head) {
	ListNode* tail = head;
	while (tail && tail->next) {
		tail = tail->next;
	}
	return tail;
}

ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
	ListNode *hA = headA, *hB = headB;
	while (hA && hB) {
		if (hA == hB) return hA;
		hA = hA->next; hB = hB->next;
		if (!hA && !hB) return NULL; // both NULL
		if (!hA) hA = headB; // hA NULL
		if (!hB) hB = headA; // hB NULL
	}
	return NULL;
}

ListNode* insertionSortList(ListNode* head) {
	ListNode dummy(INT_MIN);
	ListNode *prev, *cur, *next;
	for (ListNode* p = head; p; p = next) {
		next = p->next;
		// invariant: list headed by dummy.next is sorted
		for (prev = &dummy, cur = prev->next;
			cur && p->val > cur->val;
			prev = cur, cur = cur->next);
		prev->next = p;
		p->next = cur;
	}
	return dummy.next;
}

ListNode* sortList(ListNode* head) {
	if (!head || !head->next) return head;
	ListNode *slow = head, *fast = head;
	while (fast->next && fast->next->next) {
		slow = slow->next; fast = fast->next->next;
	}
	ListNode *a = head, *b = slow->next;
	slow->next = NULL;
	ListNode *l1 = sortList(a), *l2 = sortList(b);
	head = mergeTwoLists(l1, l2);
	return head;
}

ListNode* swapPairs(ListNode* head) {
	ListNode dummy(0); dummy.next = head;
	ListNode* pre = &dummy, *next;
	while (head && head->next) {
		next = head->next;
		head->next = next->next;
		next->next = head;
		pre->next = next;
		pre = head;
		head = pre->next;
	}
	return dummy.next;
}

ListNode* partition(ListNode* head, int x) {
	ListNode node1(0), node2(0);
	ListNode *tail1 = &node1, *tail2 = &node2;
	while (head) {
		if (head->val < x) tail1 = (tail1->next = head);
		else tail2 = (tail2->next = head);
		head = head->next;
	}
	tail2->next = NULL;
	tail1->next = node2.next;
	return node1.next;
}

/*Section: Array*/
// 217. Contains Duplicate
bool containsDuplicate(vector<int>& nums) {
	return unordered_set<int>(nums.begin(), nums.end()).size() < nums.size();
}

// 217. Contains Duplicate
bool containsDuplicate2(vector<int>& nums) {
	sort(nums.begin(), nums.end());
	for (size_t i = 1; i < nums.size(); ++i)
		if (nums[i] == nums[i - 1]) return true;
	return false;
}

// 219. Contains Duplicate II
bool containsNearbyDuplicate(vector<int>& nums, int k) {
	unordered_map<int, int> map;
	for (int i = 0; i < (int)nums.size(); ++i) {
		if (map.find(nums[i]) != map.end() && i - map[nums[i]] <= k)
			return true;
		map[nums[i]] = i;
	}
	return false;
}

// 287
int findDuplicate3(vector<int>& nums) {
	int n = nums.size();
	if (n == 0 || n == 1) return -1;
	int slow = nums[0], fast = nums[nums[0]];
	while (slow != fast) {
		slow = nums[slow];
		fast = nums[nums[fast]];
	}
	fast = 0;
	while (fast != slow) {
		fast = nums[fast];
		slow = nums[slow];
	}
	return slow;
}

// 26 remove duplicates from sorted array
int removeDuplicates(vector<int>& nums) {
	if (nums.empty()) return 0;
	int i = 1;
	for (int j = 1; j < (int)nums.size(); ++j) {
		if (nums[j] != nums[j - 1]) nums[i++] = nums[j];
	}
	return i;
}

// 66 plus one
vector<int> plusOne(vector<int>& digits) {
	bool carry = true;
	for (int i = digits.size() - 1; i >= 0 && carry; --i) {
		carry = ((++digits[i] %= 10) == 0);
		if (carry) digits[i] = 0;
	}
	if (carry) digits.insert(digits.begin(), 1);
	return digits;
}

// 243 shortest distance
int shortestDistance(vector<string>& words, string word1, string word2) {
	long long dist(INT_MAX), i1(-dist), i2(dist);
	int n(words.size());
	for (int i = 0; i < n; ++i) {
		if (words[i] == word1) i1 = i;
		if (words[i] == word2) i2 = i;
		dist = min(dist, abs(i1 - i2));
	}
	return (int)dist;
}

// 245 shortest distance III
int shortestDistance3(vector<string>& words, string word1, string word2) {
	long long dist(INT_MAX), i1(-dist), i2(dist);
	int n(words.size());
	for (int i = 0; i < n; ++i) {
		if (words[i] == word1) i1 = i;
		if (words[i] == word2) {
			if (word1 == word2)  i1 = i2;
			i2 = i;
		}
		dist = min(dist, abs(i1 - i2));
	}
	return (int)dist;
}

// 244 shortest word distance II
wordDistance::wordDistance(vector<string>& words) {
	for (int i = 0; i < (int)words.size(); ++i)
		wmap[words[i]].push_back(i);
}

int wordDistance::shortest(string word1, string word2) {
	int i = 0, j = 0, dist = INT_MAX;
	while (i < (int)wmap[word1].size() && j < (int)wmap[word2].size()) {
		dist = min(dist, abs(wmap[word1][i] - wmap[word2][j]));
		wmap[word1][i] < wmap[word2][j] ? ++i : ++j;
	}
	return dist;
}

// 189. Rotate Array
void rotate(vector<int>& nums, int k) {
	int n = nums.size();
	reverse(nums.begin(), nums.end());
	reverse(nums.begin(), nums.begin() + k%n);
	reverse(nums.begin() + k%n, nums.end());
}

// 189.2 Rotate Array
void rotate2(vector<int>& nums, int k) {
	int n = nums.size();
	if (n == 0 || k <= 0) return;
	vector<int> tmp(nums);
	for (int i = 0; i < n; ++i) nums[(i + k) % n] = tmp[i];
}

// 153. Find Minimum in Rotated Sorted Array
int findMin(vector<int>& nums) {
	int left = 0, right = nums.size() - 1;
	while (left < right) {
		if (nums[left] < nums[right]) return nums[left];
		int mid = left + (right - left) / 2;
		if (nums[mid] >= nums[left]) left = mid + 1;
		else right = mid;
	}
	return nums[left];
}

// 325. Maximum Size Subarray Sum Equals k
int maxSubArrayLen(vector<int>& nums, int k) {
	unordered_map<int, int> mp; mp[0] = -1; // cumsum->index, 0 at index -1, convenient
	int res = 0;
	for (int i = 0, csum = 0; i < (int)nums.size(); ++i) {
		csum += nums[i];
		if (mp.find(csum - k) != mp.end()) // find ?..i subarray sum to k
			res = max(res, i - mp[csum - k]);
		if (mp.find(csum) == mp.end()) // only the earliest csum
			mp[csum] = i;
	}
	return res;
}

/*Section: Tree */
// 297. Serialize and Deserialize Binary Tree
// preorder approach
void Codec::serialize(TreeNode* root, ostringstream& out) {
	if (root) {
		out << root->val << ' ';
		serialize(root->left, out);
		serialize(root->right, out);
	}
	else {
		out << "# ";
	}
}
TreeNode* Codec::deserialize(istringstream& in) {
	string val;
	in >> val;
	if (val == "#") return NULL;
	TreeNode* root = new TreeNode(stoi(val));
	root->left = deserialize(in);
	root->right = deserialize(in);
	return root;
}
string Codec::serialize(TreeNode* root) {
	ostringstream out;
	serialize(root, out);
	return out.str();
}
TreeNode* Codec::deserialize(string data) {
	istringstream in(data);
	return deserialize(in);
}

// 108 Convert Sorted Array to Binary Search Tree
TreeNode* sortedArrayToBST(int start, int end, vector<int>& nums) {
	if (end <= start) return NULL;
	int mid = (end + start) / 2;
	TreeNode* root = new TreeNode(nums[mid]);
	root->left = sortedArrayToBST(start, mid, nums);
	root->right = sortedArrayToBST(mid + 1, end, nums);
	return root;
}
TreeNode* sortedArrayToBST(vector<int>& nums) {
	return sortedArrayToBST(0, nums.size(), nums);
}

// 226. Invert Binary Tree
TreeNode* invertTree(TreeNode* root) {
	if (root == NULL) return NULL;
	TreeNode* tmp = root->right;
	root->right = invertTree(root->left);
	root->left = invertTree(tmp);
	return root;
}

// 114. Flatten Binary Tree to Linked List
void flatten(TreeNode* root) {
	if (root == NULL) return;
	flatten(root->left); flatten(root->right);
	TreeNode* tmp = root->right;
	root->right = root->left;
	root->left = NULL;
	while (root->right) root = root->right;
	root->right = tmp;
}

// 156. Binary Tree Upside Down
TreeNode* upsideDownBinaryTree(TreeNode* root) {
	TreeNode *curr = root, *prev = NULL, *next = NULL, *temp = NULL;
	while (curr) {
		next = curr->left;
		curr->left = temp;
		temp = curr->right;
		curr->right = prev;
		prev = curr;
		curr = next;
	}
	return prev;
}

// 144. Binary Tree Preorder Traversal
void preorderTraversal(TreeNode* root, vector<int>& nodes) {
	if (root == NULL) return;
	nodes.push_back(root->val);
	preorderTraversal(root->left, nodes);
	preorderTraversal(root->right, nodes);
}
vector<int> preorderTraversal(TreeNode* root) {
	vector<int> nodes;
	preorderTraversal(root, nodes);
	return nodes;
}

// 144. Binary Tree Preorder Traversal
vector<int> preorderTraversal2(TreeNode* root) {
	vector<int> res;
	if (root == NULL) return res;
	stack<TreeNode*> nodes;
	nodes.push(root);
	while (!nodes.empty()) {
		TreeNode* node = nodes.top(); nodes.pop();
		res.push_back(node->val);
		if (node->right) nodes.push(node->right);
		if (node->left) nodes.push(node->left);
	}
	return res;
}

// 94 Binary Tree Inorder Traversal
void inorderTraversal(TreeNode* root, vector<int>& nodes) {
	if (root == NULL) return;
	inorderTraversal(root->left, nodes);
	nodes.push_back(root->val);
	inorderTraversal(root->right, nodes);
}
vector<int> inorderTraversal(TreeNode* root) {
	vector<int> nodes;
	inorderTraversal(root, nodes);
	return nodes;
}
// 94. Binary Tree Inorder Traversal
vector<int> inorderTraversal2(TreeNode* root) {
	vector<int> res;
	stack<TreeNode*> nodes;
	while (true) {
		while (root) { nodes.push(root); root = root->left; }
		if (nodes.empty()) break;
		root = nodes.top(); nodes.pop();
		res.push_back(root->val);
		root = root->right;
	}
	return res;
}

// 173. Binary Search Tree Iterator
void BSTIterator::pushAll(TreeNode *node) {
	while (node) { myStack.push(node); node = node->left; }	
}

BSTIterator::BSTIterator(TreeNode *root) {
	pushAll(root);
}

/** @return whether we have a next smallest number */
bool BSTIterator::hasNext() {
	return !myStack.empty();
}

/** @return the next smallest number */
int BSTIterator::next() {
	TreeNode *tmpNode = myStack.top(); myStack.pop();
	pushAll(tmpNode->right);
	return tmpNode->val;
}

// 94 Binary Tree Postorder Traversal
void postorderTraversal(TreeNode* root, vector<int>& nodes) {
	if (root == NULL) return;
	postorderTraversal(root->left, nodes);
	postorderTraversal(root->right, nodes);
	nodes.push_back(root->val);
}
vector<int> postorderTraversal(TreeNode* root) {
	vector<int> nodes;
	postorderTraversal(root, nodes);
	return nodes;
}

// 102 Binary Tree Level Order Traversal
void levelOrder(TreeNode* root, int level, vector<vector<int> >& res) {
	if (root == NULL) return;
	if ((int)res.size() < level + 1) res.push_back(vector<int>());
	res[level].push_back(root->val);
	levelOrder(root->left, level + 1, res);
	levelOrder(root->right, level + 1, res);
}
vector<vector<int> > levelOrder(TreeNode(*root)) {
	vector<vector<int> > res;
	levelOrder(root, 0, res);
	return res;
}

// 107 Binary Tree Level Order Traversal II
void levelOrderBottom(TreeNode* root, int level, vector<vector<int> >& res) {
	if (root == NULL) return;
	if ((int)res.size() < level + 1) res.push_back(vector<int>());
	res[level].push_back(root->val);
	levelOrderBottom(root->left, level + 1, res);
	levelOrderBottom(root->right, level + 1, res);
}
vector<vector<int> > levelOrderBottom(TreeNode(*root)) {
	vector<vector<int> > res;
	levelOrderBottom(root, 0, res);
	reverse(res.begin(), res.end());
	return res;
}

// 103 Binary Tree Zigzag Level Order Traversal II
void zigzagLevelOrder(TreeNode* root, int level, vector<vector<int> >& res) {
	if (root == NULL) return;
	if ((int)res.size() < level + 1) res.push_back(vector<int>());
	res[level].push_back(root->val);
	zigzagLevelOrder(root->left, level + 1, res);
	zigzagLevelOrder(root->right, level + 1, res);
}
vector<vector<int> > zigzagLevelOrder(TreeNode(*root)) {
	vector<vector<int> > res;
	zigzagLevelOrder(root, 0, res);
	for (int i = 1; i < (int)res.size(); i += 2) {
		reverse(res[i].begin(), res[i].end());
	}
	return res;
}

// 199 Binary Tree Right Side View
void rightSideView(TreeNode* root, int level, vector<int>& res) {
	if (root == NULL) return;
	if ((int)res.size() < level + 1)
		res.push_back(root->val);
	rightSideView(root->right, level + 1, res);
	rightSideView(root->left, level + 1, res);
}
vector<int> rightSideView(TreeNode* root) {
	vector<int> res;
	rightSideView(root, 0, res);
	return res;
}

// Self. cout treeNode
void levelPrint(TreeNode* root, int level, vector<vector<string> >& res) {
	if ((int)res.size() < level + 1) res.push_back(vector<string>());
	if (root == NULL) { res[level].push_back("#");  return; }
	res[level].push_back(to_string(root->val));
	levelPrint(root->left, level + 1, res);
	levelPrint(root->right, level + 1, res);
}
ostream& operator<<(ostream& os, TreeNode* root) {
	vector<vector<string> > res;
	levelPrint(root, 0, res);
	os << "[" << endl;
	for (size_t i = 0; i < res.size(); ++i) {
		os << "[";
		for (size_t j = 0; j < res[i].size(); ++j) {
			os << res[i][j];
			if (j != res[i].size() - 1) os << " ";
		}
		os << "]" << endl;
	}
	os << "]";
	return os;
}

// 104. Maximum Depth of Binary Tree
int maxDepth(TreeNode* root) {
	return root == NULL ? 0 : max(maxDepth(root->left), maxDepth(root->right)) + 1;
}

// 104. Maximum Depth of Binary Tree
int maxDepth2(TreeNode* root) {
	if (root == NULL) return 0;
	int res = 0;
	queue<TreeNode*> q;
	q.push(root);
	while (!q.empty()) {
		++res;
		for (int i = 0, n = q.size(); i < n; ++i) {
			TreeNode* p = q.front(); q.pop();
			if (p->left != NULL) q.push(p->left);
			if (p->right != NULL) q.push(p->right);
		}
	}
	return res;
}

// 111. Minimum Depth of Binary Tree
int minDepth(TreeNode* root) {
	if (root == NULL) return 0;
	if (root->left == NULL) return 1 + minDepth(root->right);
	if (root->right == NULL) return 1 + minDepth(root->left);
	return min(minDepth(root->left), minDepth(root->right)) + 1;
}

// 111. Minimum Depth of Binary Tree
int minDepth2(TreeNode* root) {
	if (root == NULL) return 0;
	int res = 0;
	queue<TreeNode*> q;
	q.push(root);
	while (!q.empty()) {
		++res;
		for (int i = 0, n = q.size(); i < n; ++i) {
			TreeNode* p = q.front(); q.pop();
			if (p->left == NULL && p->right == NULL) return res;
			if (p->left != NULL) q.push(p->left);
			if (p->right != NULL) q.push(p->right);
		}
	}
	return res;
}

// 110. Balanced Binary Tree
int height(TreeNode *root) {
	if (root == NULL) return 0;
	return max(height(root->left), height(root->right)) + 1;
}
bool isBalanced(TreeNode* root) {
	if (root == NULL) return true;
	return isBalanced(root->left) && isBalanced(root->right)
		&& abs(height(root->left) - height(root->right)) <= 1;
}

// 100. Same Tree
bool isSameTree(TreeNode* p, TreeNode* q) {
	if (p == NULL || q == NULL) return (p == q);
	return (p->val == q->val
		&& isSameTree(p->left, q->left)
		&& isSameTree(p->right, q->right));
}

// 222. Count Complete Tree Nodes
int countNodes(TreeNode* root) {
	if (root == NULL) return 0;
	int leftDepth = 0, rightDepth = 0;
	for (TreeNode* p = root; p; p = p->left) ++leftDepth;
	for (TreeNode* p = root; p; p = p->right) ++rightDepth;
	if (leftDepth == rightDepth) return (1 << leftDepth) - 1;
	else return countNodes(root->left) + countNodes(root->right) + 1;
}

// 270. Closest Binary Search Tree Value
int closestValue2(TreeNode* root, double target) {
	int res = root->val;
	while (root) {
		if ((double)root->val == target) return root->val;
		if (abs((double)root->val - target) < abs(res - target)) res = root->val;
		if ((double)root->val > target) root = root->left;
		else root = root->right;
	}
	return (int)res;
}

// 270. Closest Binary Search Tree Value
void closestValue(TreeNode* node, double target, double &result){
	if (node == NULL) return;
	if (abs((double)node->val - target) < abs(target - result)) result = (double)node->val;
	if ((double)node->val > target) closestValue(node->left, target, result);
	if ((double)node->val < target) closestValue(node->right, target, result);
}
int closestValue(TreeNode* root, double target) {
	double result = (double)root->val;
	closestValue(root, target, result);
	return (int)result;
}

// 112. Path Sum
bool hasPathSum(TreeNode* root, int sum) {
	if (root == NULL) return false;
	if (root->left == NULL && root->right == NULL && root->val == sum) return true;
	else return (hasPathSum(root->left, sum - root->val) || hasPathSum(root->right, sum - root->val));
}

// 113. Path Sum II
void pathSum(TreeNode* node, int sum, vector<int>& path, vector<vector<int> >& paths) {
	if (node == NULL) return;
	path.push_back(node->val);

	if (node->left == NULL && node->right == NULL && node->val == sum) paths.push_back(path);
	pathSum(node->left, sum - node->val, path, paths);
	pathSum(node->right, sum - node->val, path, paths);

	path.pop_back();
}
vector<vector<int> > pathSum(TreeNode* root, int sum) {
	vector<vector<int> > paths;
	vector<int> path;
	pathSum(root, sum, path, paths);
	return paths;
}

// 257. Binary Tree Paths
void binaryTreePaths(TreeNode* root, string s, vector<string>& res) {
	if (root->left == NULL && root->right == NULL) { res.push_back(s); return; }
	if (root->left)
		binaryTreePaths(root->left, s + "->" + to_string(root->left->val), res);
	if (root->right)
		binaryTreePaths(root->right, s + "->" + to_string(root->right->val), res);
}
vector<string> binaryTreePaths(TreeNode* root) {
	vector<string> res;
	if (root == NULL) return res;
	binaryTreePaths(root, to_string(root->val), res);
	return res;
}

// 129. Sum Root to Leaf Numbers
void sumNumbers(TreeNode* node, int csum, int& sum) {
	csum = csum * 10 + node->val;
	if (node->left == NULL && node->right == NULL) sum += csum;
	if (node->left) sumNumbers(node->left, csum, sum);
	if (node->right) sumNumbers(node->right, csum, sum);
}
int sumNumbers(TreeNode* root) {
	if (root == NULL) return 0;
	int sum = 0;
	sumNumbers(root, 0, sum);
	return sum;
}

// 101. Symmetric Tree
bool isSymmetric(TreeNode* left, TreeNode* right) {
	if (left == NULL || right == NULL) return left == right;
	if (left->val != right->val) return false;
	return isSymmetric(left->left, right->right)
		&& isSymmetric(left->right, right->left);
}
bool isSymmetric(TreeNode* root) {
	if (root == NULL) return true;
	return isSymmetric(root->left, root->right);
}

// 101. Symmetric Tree
bool isSymmetric2(TreeNode* root) {
	if (root == NULL) return true;
	stack<TreeNode*> stk;
	stk.push(root->left); stk.push(root->right);
	TreeNode* pA, *pB;
	while (!stk.empty()) {
		pA = stk.top(); stk.pop();
		pB = stk.top(); stk.pop();
		if (pA == NULL && pB == NULL) continue;
		if (pA == NULL || pB == NULL) return false;
		if (pA->val != pB->val) return false;
		stk.push(pA->left); stk.push(pB->right);
		stk.push(pA->right); stk.push(pB->left);
	}
	return true;
}

// 98. Validate Binary Search Trees
bool isValidBST(TreeNode* node, long long min, long long max) {
	if (node == NULL) return true;
	if (node->val <= min || node->val >= max) return false;
	return isValidBST(node->left, min, node->val)
		&& isValidBST(node->right, node->val, max);
}
bool isValidBST(TreeNode* root) {
	return isValidBST(root, LLONG_MIN, LLONG_MAX);
}

// 230. Kth Smallest Element in a BST
void kthSmallest(TreeNode* root, int& k, int& res){
	if (root == NULL || k == 0) return;
	kthSmallest(root->left, k, res);
	--k;
	if (k == 0) res = root->val;
	kthSmallest(root->right, k, res);
}
int kthSmallest(TreeNode* root, int k) {
	int res;
	kthSmallest(root, k, res);
	return res;
}

// 230. Inorder Successor in BST
TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p) {
	if (root == NULL || p == NULL) return NULL;
	TreeNode *suc = NULL;
	while (root) {
		if (root->val <= p->val) root = root->right;
		else { suc = root; root = root->left; }
	}
	return suc;
}

// 235. Lowest Common Ancestor of a Binary Search Tree
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
	if (p->val < root->val && q->val < root->val)
		return lowestCommonAncestor(root->left, p, q);
	if (p->val > root->val && q->val > root->val)
		return lowestCommonAncestor(root->right, p, q);
	return root;
}

// 235. Lowest Common Ancestor of a Binary Search Tree
TreeNode* lowestCommonAncestor2(TreeNode* root, TreeNode* p, TreeNode* q) {
	TreeNode* cur = root;
	while (true) {
		if (p->val < cur->val && q->val < cur->val) cur = cur->left;
		else if (p->val > cur->val && q->val > cur->val) cur = cur->right;
		else return cur;
	}
}

// 236. Lowest Common Ancestor of a Binary Tree
TreeNode* lowestCommonAncestorBT(TreeNode* root, TreeNode* p, TreeNode* q) {
	if (root == NULL || root == p || root == q) return root;
	TreeNode* left = lowestCommonAncestorBT(root->left, p, q);
	TreeNode* right = lowestCommonAncestorBT(root->right, p, q);
	if (left == NULL) return right;
	if (right == NULL) return left;
	return root;
}

// 95. Unique Binary Search Trees I
int numTrees(int n) {
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

// 96. Unique Binary Search Trees II
vector<TreeNode*> generateTrees(int left, int right) {
	if (left > right) return vector<TreeNode*>{NULL};
	vector<TreeNode*> trees;
	for (int i = left; i <= right; ++i) {
		vector<TreeNode*> ltrees = generateTrees(left, i - 1);
		vector<TreeNode*> rtrees = generateTrees(i + 1, right);
		for (TreeNode* ltree : ltrees) {
			for (TreeNode* rtree : rtrees) {
				TreeNode* root = new TreeNode(i);
				root->left = ltree;
				root->right = rtree;
				trees.push_back(root);
			}
		}
	}
	return trees;
}
vector<TreeNode*> generateTrees(int n) {
	if (n == 0) return vector<TreeNode*>();
	return generateTrees(1, n);
}

/*Section: Hash Table*/
/*Section: Hash Table*/
// 1. Two Sum
vector<int> twoSum(vector<int>& nums, int target) {
	unordered_map<int, int> imap;
	vector<int> result;
	for (int i = 0; i < (int)nums.size(); i++) {
		int comp = target - nums[i];
		if (imap.find(comp) != imap.end()) {
			result.push_back(imap[comp] + 1);
			result.push_back(i + 1); // not zero based
			return result;
		}
		imap[nums[i]] = i;
	}
	return result;
}

// 167. Two Sum II - Input Array is Sorted
vector<int> twoSumII(vector<int>& nums, int target) {
	int l = 0, r = nums.size() - 1;
	while (l < r) {
		int s = nums[l] + nums[r];
		if (s > target) --r;
		else if (s < target) ++l;
		else return vector<int>{l + 1, r + 1};
	}
	return vector<int>{}; // dummy line
}

// 15. 3 Sum
vector<vector<int> > threeSum(vector<int> &nums) {
	sort(nums.begin(), nums.end());
	vector<vector<int> > res;
	int n = nums.size();
	for (int i = 0; i < n - 2; ++i) {
		int target = -nums[i];
		int front = i + 1, back = n - 1;
		while (front < back) {
			int sum = nums[front] + nums[back];
			if (sum < target) ++front;
			else if (sum > target) --back;
			else {
				vector<int> t{ nums[i], nums[front], nums[back] };
				res.push_back(t);
				while (front<back && nums[front] == t[1]) ++front;
				while (front<back && nums[back] == t[2]) --back;
			}
		}
		while (i < n - 2 && nums[i + 1] == nums[i]) ++i;
	}
	return res;
}

// 18. 4 Sum
vector<vector<int>> fourSum(vector<int>& nums, int target) {
	vector<vector<int> > res;
	if (nums.empty()) return res;
	sort(nums.begin(), nums.end());
	int n = nums.size();
	for (int i = 0; i < n - 3; ++i) {
		int target_3 = target - nums[i];
		for (int j = i + 1; j < n - 2; ++j) {
			int target_2 = target_3 - nums[j];
			int front = j + 1, back = n - 1;
			while (front < back) {
				int two_sum = nums[front] + nums[back];
				if (two_sum < target_2) ++front;
				else if (two_sum > target_2) --back;
				else {
					vector<int> q{ nums[i], nums[j], nums[front], nums[back] };
					res.push_back(q);
					while (front < back && nums[front] == q[2]) ++front;
					while (front < back && nums[back] == q[3]) --back;
				}
			}
			while (j < n - 2 && nums[j + 1] == nums[j]) ++j;
		}
		while (i < n - 3 && nums[i + 1] == nums[i]) ++i;
	}
	return res;
}

// 16. 3Sum Closest
int threeSumClosest(vector<int>& nums, int target) {
	int n = nums.size();
	if (n < 3) return 0;
	sort(nums.begin(), nums.end());
	int res = nums[0] + nums[1] + nums[2];
	for (int i = 0; i < n - 2; ++i) {
		int front = i + 1, back = n - 1;
		while (front < back) {
			int sum = nums[i] + nums[front] + nums[back];
			if (sum == target) return sum;
			if (abs(target - sum)<abs(target - res)) res = sum;
			if (sum > target) --back;
			else ++front;
		}
	}
	return res;
}

// 170. Two Sum III - Data Structure Design
void TwoSum::add(int number) { map[number]++; }
bool TwoSum::find(int value) {
	for (auto it : map) {
		int i = it.first, j = value - i;
		if ((i == j && it.second>1)
			|| (i != j && map.find(j) != map.end()))
			return true;
	}
	return false;
}

// 299. Bulls and Cows
string getHint(string secret, string guess) {
	if (secret.empty() || secret.size() != guess.size()) return "0A0B";
	int na = 0, nb = 0;
	vector<int> svec(10, 0), gvec(10, 0);
	for (int i = 0; i < (int)secret.size(); ++i) {
		char c1 = secret[i], c2 = guess[i];
		if (c1 == c2) { ++na; }
		else { ++svec[c1 - '0']; ++gvec[c2 - '0']; }
	}
	for (int i = 0; i < (int)svec.size(); ++i) nb += min(svec[i], gvec[i]);
	return to_string(na) + "A" + to_string(nb) + "B";
}


// 202. Happy Number
int digitSquareSum(int n) {
	int res = 0;
	while (n > 0) {
		int t = n % 10;
		res += t*t;
		n /= 10;
	}
	return res;
}
bool isHappy(int n) {
	int i1 = n, i2 = digitSquareSum(n);
	while (i2 != i1) {
		i1 = digitSquareSum(i1);
		i2 = digitSquareSum(digitSquareSum(i2));
	}
	return i1 == 1;
}

// 242. Valid Anagram
bool isAnagram(string s, string t) {
	sort(s.begin(), s.end());
	sort(t.begin(), t.end());
	return s == t;
}

// 242. Valid Anagram
bool isAnagram2(string s, string t) {
	if (s.size() != t.size()) return false;
	int n = s.length();
	vector<int> counts(256, 0);
	for (int i = 0; i < n; ++i) {
		++counts[s[i]];
		--counts[t[i]];
	}
	for (int count : counts)
		if (count != 0) return false;
	return true;
}

// 249. Group Shifted Strings
string shift(string s) {
	string t;
	int n = s.length();
	for (int i = 1; i < n; i++) {
		int diff = s[i] - s[i - 1];
		if (diff < 0) diff += 26;
		t += to_string(diff) + ','; // encoding
	}
	return t;
}
vector<vector<string>> groupStrings(vector<string>& strings) {
	unordered_map<string, vector<string> > mp;
	for (string s : strings) mp[shift(s)].push_back(s);
	vector<vector<string> > groups;
	for (auto m : mp) {
		vector<string> group = m.second;
		sort(group.begin(), group.end());
		groups.push_back(group);
	}
	return groups;
}

// 49. Group Anagram
vector<vector<string>> groupAnagrams(vector<string>& strs) {
	unordered_map<string, vector<string> > smap;
	for (string s : strs) {
		string t = s;
		sort(s.begin(), s.end());
		smap[s].push_back(t);
	}
	vector<vector<string> > res;
	for (auto it : smap) {
		sort(it.second.begin(), it.second.end());
		res.push_back(it.second);
	}
	return res;
}

// 205. Isomorphic Strings
bool isIsomorphic(string s, string t) {
	if (s.size() != t.size()) return false;
	vector<int> smap(256, -1), tmap(256, -1);
	int n = s.size();
	for (int i = 0; i < n; ++i) {
		if (smap[s[i]] != tmap[t[i]]) return false;
		smap[s[i]] = tmap[t[i]] = i;
	}
	return true;
}

// 3. Longest Substring Without Repeating Characters
int lengthOfLongestSubstring(string s) {
	vector<int> smap(256, -1);
	int res = 0, pos = -1;
	for (int i = 0; i < (int)s.size(); ++i) {
		pos = max(pos, smap[s[i]]);
		smap[s[i]] = i;
		res = max(res, i - pos);
	}
	return res;
}

// 36. Valid Sudoku
bool isValidSudoku(vector<vector<char> >& board) {
	bool used1[9][9] = { false }, used2[9][9] = { false }, used3[9][9] = { false };
	int m = board.size(), n = board[0].size();
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (board[i][j] != '.') {
				int num = board[i][j] - '0' - 1, k = i / 3 * 3 + j / 3;
				if (used1[i][num] || used2[j][num] || used3[k][num]) return false;
				used1[i][num] = used2[j][num] = used3[k][num] = true;
			}
		}
	}
	return true;
}

// 266. Palindrome Permutation
bool canPermutePalindrome(string s) {
	vector<int> counts(256, 0);
	for (char c : s) ++counts[c];
	int n = 0;
	for (int i : counts) if (i % 2 != 0) ++n;
	return n <= 1;
}

// 246. Strobogrammatic Number
bool isStrobogrammatic(string num) {
	unordered_map<char, char> map{ { '0', '0' }, { '1', '1' }, { '6', '9' }, { '8', '8' }, { '9', '6' } };
	for (int l = 0, r = num.length() - 1; l <= r; ++l, --r)
		if (map.find(num[l]) == map.end() || map[num[l]] != num[r]) return false;
	return true;
}

// 247. Strobogrammatic Number II
vector<string> findStrobogrammatic(int n, int m) {
	// n: current #digit, m: total #digit
	if (n == 0) return vector<string> {""};
	if (n == 1) return vector<string> {"0", "1", "8"};
	vector<string> vec = findStrobogrammatic(n - 2, m);
	vector<string> res;
	for (size_t i = 0; i < vec.size(); ++i) {
		if (n != m) res.push_back("0" + vec[i] + "0");
		res.push_back("1" + vec[i] + "1");
		res.push_back("6" + vec[i] + "9");
		res.push_back("8" + vec[i] + "8");
		res.push_back("9" + vec[i] + "6");
	}
	return res;
}
vector<string> findStrobogrammatic(int n) {
	return findStrobogrammatic(n, n);
}

// 314. Binary Tree Vertical Order
vector<vector<int>> verticalOrder(TreeNode* root) {
	map<int, vector<int>> map;
	queue<pair<int, TreeNode*>> q;
	if (root) q.emplace(0, root);
	while (!q.empty()) {
		pair<int, TreeNode*> cur = q.front(); q.pop();
		map[cur.first].push_back(cur.second->val);
		if (cur.second->left) q.emplace(cur.first - 1, cur.second->left);
		if (cur.second->right) q.emplace(cur.first + 1, cur.second->right);
	}
	vector<vector<int> > res;
	for (auto i : map) res.push_back(i.second);
	return res;
}

// 274. H-Index
int hIndex(vector<int>& citations) {
	if (citations.empty()) return 0;
	int n = citations.size();
	vector<int> counts(n + 1, 0);
	for (int citation : citations) {
		if (citation >= n) ++counts[n];
		else ++counts[citation];
	}
	int paper = 0;
	for (int i = n; i >= 0; --i) {
		paper += counts[i];
		if (paper >= i) return i;
	}
	return -1;
}

// 275. H-Index II
int hIndexII(vector<int>& citations) {
	int n = citations.size(), left = 0, right = n - 1;
	while (left <= right) {
		int mid = left + (right - left) / 2;
		if (citations[mid] < n - mid) left = mid + 1;
		else right = mid - 1;
	}
	return n - left;
}

// 204. Count Primes
int countPrimes(int n) {
	vector<bool> prime(n, true);
	prime[0] = false, prime[1] = false;
	for (int i = 0; i < sqrt(n); ++i) {
		if (prime[i]) {
			for (int j = i*i; j < n; j += i) prime[j] = false;
		}
	}
	return count(prime.begin(), prime.end(), true);
}

// 311. Sparse Matrix Multiplication
vector<vector<int>> multiply(vector<vector<int>>& A, vector<vector<int>>& B) {
	int m = A.size(), p = B.size(), n = B[0].size();
	vector<vector<int>> C(m, vector<int>(n, 0));
	unordered_map<int, unordered_map<int, int>> sA, sB;
	for (int i = 0; i < m; i++)
		for (int k = 0; k < p; k++)
			if (A[i][k] != 0) sA[i][k] = A[i][k]; // A: row->(col->val); i,k=>i,k

	for (int j = 0; j < n; j++)
		for (int k = 0; k < p; k++)
			if (B[k][j] != 0) sB[j][k] = B[k][j]; // B: col->(row->val); j,k=>k,j

	for (auto iA : sA) // each row in A
		for (auto iB : sB) // each col in B
			for (auto iter : iA.second) // every col of each row in A
				if (iB.second.find(iter.first) != iB.second.end())
					// Cij=sum(Aik*Bkj for all k), matching row, col
					C[iA.first][iB.first] += iter.second*iB.second[iter.first];
	return C;
}

// 187. Repeated DNA Sequences
int str2int(string s) {
	int res = 0;
	for (int i = 0; i < (int)s.size(); ++i)
		res = (res << 3) + (s[i] & 7);
	return res;
}
vector<string> findRepeatedDnaSequences(string s) {
	vector<string> res;
	unordered_map<int, int> map;
	for (int i = 0; i + 10 <= (int)s.size(); ++i)
		if (++map[str2int(s.substr(i, 10))] == 2)
			res.push_back(s.substr(i, 10));
	return res;
}

// 166. Fraction to Recurring Decimal
string fractionToDecimal(int numerator, int denominator) {
	if (numerator == 0) return "0";
	string res;
	if ((numerator < 0) ^ (denominator < 0)) res += '-';
	long numer = numerator < 0 ? (long)numerator * (-1) : (long)numerator;
	long denom = denominator < 0 ? (long)denominator * (-1) : (long)denominator;
	long integral = numer / denom;
	res += to_string(integral);
	long rmd = numer % denom;
	if (rmd == 0) return res;
	res += '.';
	rmd *= 10;
	unordered_map<long, long> mp;
	while (rmd > 0) {
		long quotient = rmd / denom;
		if (mp.find(rmd) != mp.end()) {
			res.insert(mp[rmd], 1, '(');
			res += ')';
			break;
		}
		mp[rmd] = res.size();
		res += to_string(quotient);
		rmd = (rmd % denom) * 10;
	}
	return res;
}

// 138. Copy List with Random Pointers
RandomListNode *copyRandomList(RandomListNode *head) {
	unordered_map<RandomListNode*, RandomListNode*> map;
	RandomListNode *newHead = NULL, *newTail = NULL;
	for (RandomListNode* curr = head; curr; curr = curr->next) {
		RandomListNode *node = new RandomListNode(curr->label);
		map[curr] = node;
		if (newHead == NULL) { newHead = node; newTail = node; }
		else { newTail->next = node; newTail = node; }
	}
	for (RandomListNode* curr = head; curr; curr = curr->next)
		if (curr->random != NULL) map[curr]->random = map[curr->random];
	return newHead;
}

/*Section: Math*/
// 231. Power of Two, bit
bool isPowerOfTwo0(int n) {
	if (n <= 0) return false;
	return !(n&(n - 1));
}

// 231. Power of Two, map
bool isPowerOfTwo1(int n) {
	unordered_set<int> set{
		1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
		1024, 2048, 4096, 8192, 16384, 32768,
		65536, 131072, 262144, 524288, 1048576,
		2097152, 4194304, 8388608, 16777216,
		33554432, 67108864, 134217728, 268435456,
		536870912, 1073741824 };
	return (set.find(n) != set.end());
}

// 231. Power of Two, module
bool isPowerOfTwo2(int n) {
	return (n > 0) && (1073741824 % n == 0);
	// 2^30 = 1073741824
}

// 231. Power of Two, log
bool isPowerOfTwo3(int n) {
	if (n <= 0) return false;
	double x = log10(n) / log10(2);
	return x == floor(x); // only x=2^n will give true
}

// 231. Power of Two, iterative
bool isPowerOfTwo4(int n) {
	if (n > 1) {
		while (n % 2 == 0) n /= 2;
	}
	return n == 1;
}

// 231. Power of Two, recursive
bool isPowerOfTwo5(int n) {
	return n>0 && (n == 1 || (n % 2 == 0 && isPowerOfTwo5(n / 2)));
}

// 326. Power of Three, module
bool isPowerOfThree1(int n) {
	return (n > 0) && (1162261467 % n == 0);
	// 3^19 = 1162261467
}

// 326. Power of Three, log
bool isPowerOfThree2(int n) {
	if (n <= 0) return false;
	double x = log10(n) / log10(3);
	return x == floor(x); // only x=3^n will give true
}

// 326. Power of Three, iterative O(h), n=3^h
bool isPowerOfThree3(int n) {
	if (n > 1) {
		while (n % 3 == 0) n /= 3;
	}
	return n == 1;
}

// 326. Power of Three, recursive
bool isPowerOfThree4(int n) {
	return n>0 && (n == 1 || (n % 3 == 0 && isPowerOfThree4(n / 3)));
}

// 67. Add Binary
string addBinary0(string a, string b) {
	string s = "";
	int c = 0, i = a.size() - 1, j = b.size() - 1;
	while (i >= 0 || j >= 0 || c == 1) {
		c += (i >= 0 ? a[i--] - '0' : 0);
		c += (j >= 0 ? b[j--] - '0' : 0);
		s = char(c % 2 + '0') + s;
		c /= 2;
	}
	return s;
}

// 8. String to Integer (atoi)
int atoi0(const char *str) {
	int sign = 1, base = 0, i = 0;
	while (str[i] == ' ') ++i;
	if (str[i] == '-' || str[i] == '+') sign = 1 - 2 * (str[i++] == '-');
	while (str[i] >= '0' && str[i] <= '9') {
		if (base > INT_MAX / 10 || (base == INT_MAX / 10 && str[i] - '0'>7)) {
			if (sign == 1) return INT_MAX;
			else return INT_MIN;
		}
		base = 10 * base + (str[i++] - '0');
	}
	return base * sign;
}

// 223. Rectangle Area
int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
	int left = max(A, E), right = max(min(C, G), left);
	int bottom = max(B, F), top = max(min(D, H), bottom);
	return (C - A)*(D - B) - (right - left)*(top - bottom) + (G - E)*(H - F);
}

// 263. Ugly Number
bool isUgly(int num) {
	for (int i = 2; i < 6 && num>0; ++i) {
		while (num%i == 0) num /= i;
	}
	return num == 1;
}

// 264. Ugly Number II
int nthUglyNumber(int n) {
	if (n <= 0) return 0;
	if (n == 1) return 1;
	int t2 = 0, t3 = 0, t5 = 0;
	vector<int> d(n);
	d[0] = 1;
	for (int i = 1; i < n; ++i) {
		d[i] = min(d[t2] * 2, min(d[t3] * 3, d[t5] * 5));
		if (d[i] == d[t2] * 2) ++t2;
		if (d[i] == d[t3] * 3) ++t3;
		if (d[i] == d[t5] * 5) ++t5;
	}
	return d[n - 1];
}

// 313. Super Ugly Number
int nthSuperUglyNumber(int n, vector<int>& primes) {
	int k = primes.size();
	vector<int> index(k, 0), d(n, INT_MAX);
	d[0] = 1;
	for (int i = 1; i < n; ++i){
		for (int j = 0; j < k; ++j)
			d[i] = min(d[i], d[index[j]] * primes[j]);
		for (int j = 0; j < k; ++j)
			index[j] += (d[i] == d[index[j]] * primes[j]);
	}
	return d[n - 1];
}

// 2. Add Two Numbers
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
	ListNode dummy(-1), *tail = &dummy;
	int carry = 0;
	while (l1 || l2 || carry > 0) {
		int tmp = ((l1 ? l1->val : 0) + (l2 ? l2->val : 0) + carry);
		tail->next = new ListNode(tmp % 10);
		tail = tail->next;
		carry = tmp / 10;
		l1 ? l1 = l1->next : NULL;
		l2 ? l2 = l2->next : NULL;
	}
	return dummy.next;
}

// 43. Multiply Strings
string multiply(string num1, string num2) {
	int n1 = num1.size(), n2 = num2.size();
	string res(n1 + n2, '0');
	for (int i = n1 - 1; i >= 0; --i) {
		int carry = 0;
		for (int j = n2 - 1; j >= 0; --j) {
			int tmp = (res[i + j + 1] - '0') + (num1[i] - '0')*(num2[j] - '0') + carry;
			res[i + j + 1] = tmp % 10 + '0';
			carry = tmp / 10;
		}
		res[i] += carry;
	}
	size_t startpos = res.find_first_not_of('0');
	return (startpos == string::npos ? "0" : res.substr(startpos));
}

// 258. Add Digits, conditions
int addDigits(int num) {
	if (num == 0) return 0;
	if ((num != 0) && (num % 9 == 0)) return 9;
	else return (num % 9); // if ((num != 0) && (num % 9 != 0))
}

// 258. Add Digits, simplication
int addDigits2(int num) {
	return 1 + (num - 1) % 9;
}

// 172. Factorial Trailing Zeroes, O(log2(N)), n = #factor-5's
// 100--> 24, 5^2. 2 loops
int trailingZeroes(int n) {
	int res = 0;
	for (long long i = 5; n / i > 0; i *= 5) res += (int)(n / i);
	return res;
}

// 168. Excel Sheet Column Title
string convertToTitle(int n) {
	string res = "";
	while (n > 0) {
		res = (char)('A' + (n - 1) % 26) + res;
		n = (n - 1) / 26; // only 27 can have two letters
	}
	return res;
}

// 171. Excel Sheet Column Number
int titleToNumber(string s) {
	int res = 0;
	for (char c : s) res = 26 * res + (c - 'A' + 1);
	return res;
}

// 9. Palindrome Number
bool isPalindrome(int x) {
	if (x<0 || (x != 0 && x % 10 == 0)) return false;
	int sum = 0;
	while (x > sum) {
		sum = sum * 10 + x % 10;
		x /= 10;
	}
	return (x == sum) || (x == sum / 10);
}

// 7. Reverse Integer
int reverse(int x) {
	long long res = 0;
	while (x != 0) {
		res = res * 10 + x % 10;
		x /= 10;
	}
	return (res<INT_MIN || res>INT_MAX) ? 0 : (int)res;
}

// 13. Roman to Integer
int romanToInt(string s) {
	unordered_map<char, int> T = {
			{ 'I', 1 }, { 'V', 5 }, { 'X', 10 },
			{ 'L', 50 }, { 'C', 100 },
			{ 'D', 500 }, { 'M', 1000 } };
	int sum = T[s.back()];
	for (int i = s.length() - 2; i >= 0; --i) {
		if (T[s[i]] < T[s[i + 1]]) sum -= T[s[i]];
		else sum += T[s[i]];
	}
	return sum;
}

// 22. Integer to Roman
string intToRoman(int num) {
	string M[] = { "", "M", "MM", "MMM" };
	string C[] = { "", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM" };
	string X[] = { "", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC" };
	string I[] = { "", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX" };
	return M[num / 1000] + C[(num % 1000) / 100] + X[(num % 100) / 10] + I[num % 10];
}

// 273. Integer to English Words
string int_string(int n) {
	static vector<string> below_20 = { "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen" };
	static vector<string> below_100 = { "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety" };
	if (n >= 1000000000)   return int_string(n / 1000000000) + " Billion" + int_string(n - 1000000000 * (n / 1000000000));
	else if (n >= 1000000) return int_string(n / 1000000) + " Million" + int_string(n - 1000000 * (n / 1000000));
	else if (n >= 1000)    return int_string(n / 1000) + " Thousand" + int_string(n - 1000 * (n / 1000));
	else if (n >= 100)     return int_string(n / 100) + " Hundred" + int_string(n - 100 * (n / 100));
	else if (n >= 20)      return string(" ") + below_100[n / 10 - 2] + int_string(n - 10 * (n / 10)); // add " " in the beginning
	else if (n >= 1)       return string(" ") + below_20[n - 1]; // add " " in the beginning
	else return "";
}
string numberToWords(int n) {
	if (n == 0) return "Zero";
	else return int_string(n).substr(1);
}

// 279. Perfect Squares
int numSquares(int n) {
	if (n <= 0) return 0;
	vector<int> D(n + 1, INT_MAX);
	D[0] = 0;
	for (int i = 1; i <= n; ++i) {
		for (int j = 1; j*j <= i; ++j) {
			D[i] = min(D[i], D[i - j*j] + 1);
		}
	}
	return D[n];
}

// 268. Missing Number
int missingNumber(vector<int>& nums) {
	int missing = 0, n = nums.size();
	for (int i = 0; i < n; ++i)
		missing ^= ((i + 1) ^ nums[i]);
	return missing;
}

// 233. Number of Digit One
long long currDigitOne(long long left, long long right, long long digit) {
	long long units = left % 10; // # at 1's
	long long aboveUnits = left / 10; // # at above 1's
	if (units >= 2) return (aboveUnits + 1)*digit;
	else if (units == 1) return aboveUnits*digit + (right + 1);
	else return aboveUnits*digit; // # at 1's == 0	
}
int countDigitOne(int n) {
	long long res = 0;
	for (long long digit = 1; digit <= n; digit *= 10) {
		long long left = n / digit, right = n % digit;
		res += currDigitOne(left, right, digit);
		PRINT(currDigitOne(left, right, digit));
	}
	return (int)res;
}

// 224. Basic Calculator
int calculate(string s) {
	stack<int> stk; stk.push(1); stk.push(1);
	int res = 0, n = s.size();
	for (int i = 0; i < n; ++i) {
		if (s[i] >= '0' && s[i] <= '9') {
			int num = s[i] - '0';
			int j = i + 1;
			while (j < n && (s[j] >= '0' && s[j] <= '9')) {
				num = 10 * num + (s[j] - '0');
				++j;
			}
			res += stk.top() * num;
			i = j - 1;
			stk.pop();
		}
		else if (s[i] == '+' || s[i] == '(') {
			stk.push(stk.top());
		}
		else if (s[i] == '-') {
			stk.push(-1 * stk.top());
		}
		else if (s[i] == ')') {
			stk.pop();
		}
	}
	return res;
}

// 150. Evaluate Reverse Polish Notation
int evalRPN(vector<string>& tokens) {
	stack<int> stn;
	for (auto s : tokens) {
		if (s.size()>1 || isdigit(s[0])) stn.push(stoi(s));
		else {
			auto x2 = stn.top(); stn.pop();
			auto x1 = stn.top(); stn.pop();
			switch (s[0]) {
			case '+': x1 += x2; break;
			case '-': x1 -= x2; break;
			case '*': x1 *= x2; break;
			case '/': x1 /= x2; break;
			}
			stn.push(x1);
		}
	}
	return stn.top();
}

// 69. Sqrt(x)
int mySqrt(int x) {
	if (x == 0) return 0;
	int low = 1, high = x;
	while (true) {
		int mid = low + (high - low) / 2;
		if (mid > x / mid) high = mid - 1;
		else {
			if (mid + 1 > x / (mid + 1)) return mid;
			low = mid + 1;
		}
	}
}

// 319. Bulb Switcher
int bulbSwitch(int n) {
	return (int)sqrt(n);
}

// 29. Divide Two Integers
int divide(int dividend, int divisor) {
	// handling the case of overflow
	if (divisor == 1) return dividend;
	if (dividend == INT_MIN && abs(divisor) == 1) return INT_MAX;
	int sign = ((dividend > 0) ^ (divisor > 0)) ? -1 : 1;

	long res = 0, n = abs((long)dividend), d = abs((long)divisor);
	while (n >= d) {
		long temp = d;
		long power = 1;
		while ((temp << 1) <= n) {
			power <<= 1;
			temp <<= 1;
		}
		res += power;
		n -= temp;
	}
	return sign * res;
}

// 50. Pow(x, n), recursive
double mypow(double x, int n) {
	if (n == 0) return 1;
	if (n < 0){ n = -n; x = 1 / x; }
	return (n % 2 == 0) ? pow(x*x, n / 2) : x*pow(x*x, n / 2);
}

// 50, Pow(x, n), iterative
double myPow2(double x, int n) {
	double ans = 1;
	unsigned long long p;
	if (n < 0) { p = -n; x = 1 / x; }
	else { p = n; }

	while (p != 0) {
		if (p & 1) ans *= x;
		x *= x;
		p >>= 1;
	}
	return ans;
}

// calculate e^x using taylor expansion
double exp_taylor2(double x) { // originally from utility
	// obtain the exp function by multiplication/addition through Taylor expansion
	if (x < 0.) return 1.0 / exp_taylor(-x);
	double x_over_n = x;
	double n = 1.;
	if (log2(x) > 1.) {
		n = power_rec(2., (int)floor(log2(x)));
		x_over_n = x / n;
	}
	double res = 1., taylor_term = x_over_n, denom = 1.;
	while (taylor_term > numeric_limits<double>::min()) {
		res += taylor_term;
		taylor_term *= x_over_n / (++denom);
	}
	return power_rec(res, (int)n);
}

// Fibonacci sequence, hard coded, O(1)
long long Fib1(int n) {
	static vector<long long> F{
		1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
		1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393,
		196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887,
		9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141,
		267914296, 433494437, 701408733, 1134903170, 1836311903, 2971215073,
		4807526976, 7778742049, 12586269025, 20365011074, 32951280099, 53316291173,
		86267571272, 139583862445, 225851433717, 365435296162, 591286729879,
		956722026041, 1548008755920, 2504730781961, 4052739537881, 6557470319842,
		10610209857723, 17167680177565, 27777890035288, 44945570212853, 72723460248141,
		117669030460994, 190392490709135, 308061521170129, 498454011879264, 806515533049393,
		1304969544928657, 2111485077978050, 3416454622906707, 5527939700884757, 8944394323791464,
		14472334024676221, 23416728348467685, 37889062373143906, 61305790721611591,
		99194853094755497, 160500643816367088, 259695496911122585, 420196140727489673,
		679891637638612258, 1100087778366101931, 1779979416004714189, 2880067194370816120
	};
	if (n <= 0) return -1;
	return F[n - 1];
}

// Fibonacci sequence, matrix, O(logN)
void multiply(long long F[2][2], long long M[2][2]) {
	long long x = F[0][0] * M[0][0] + F[0][1] * M[1][0];
	long long y = F[0][0] * M[0][1] + F[0][1] * M[1][1];
	long long z = F[1][0] * M[0][0] + F[1][1] * M[1][0];
	long long w = F[1][0] * M[0][1] + F[1][1] * M[1][1];
	F[0][0] = x; F[0][1] = y;
	F[1][0] = z; F[1][1] = w;
}
void power(long long F[2][2], int n) {
	if (n == 0 || n == 1) return;
	long long M[2][2] = { { 1, 1 }, { 1, 0 } };
	power(F, n / 2);
	multiply(F, F);
	if (n % 2 != 0) multiply(F, M);
}
long long Fib2(int n) {
	long long F[2][2] = { { 1, 1 }, { 1, 0 } };
	if (n <= 0) return 0;
	power(F, n - 1);
	return F[0][0];
}


// Fibonacci sequence, static array, O(N)
long long Fib3(int n) {
	static vector<long long> F{ 1, 1 };
	if (n <= 0) return 0;
	if (n > (int)F.size()) {
		for (int i = F.size(); i < n; ++i)
			F.push_back(F[i - 2] + F[i - 1]);
	}
	return F[n - 1];
}

// Fibonacci sequence, formula, O(logN)
long long Fib4(int n) {
	static long double Phi = (1. + sqrt(5.)) / 2.;
	static long double phi = (1. - sqrt(5.)) / 2.;
	static long double sqrt5 = sqrt(5.);
	double res = (pow(Phi, n) - pow(phi, n)) / sqrt5;
	return (long long)res;
}


// Fibonacci sequence, recursion, O(2^N)
long long Fib5(int n) {
	if (n <= 0) return 0;
	if (n == 1 || n == 2) return 1;
	return Fib4(n - 2) + Fib4(n - 1);
}

// Factorial sequence, static array, O(1)
long long Fac1(int n) {
	if (n < 0) return 0;
	if (n > 20) return -1;
	static vector<long long> F{
		1, 1, 2, 6, 24, 120, 720, 5040, 40320,
		362880, 3628800, 39916800, 479001600,
		6227020800, 87178291200, 1307674368000,
		20922789888000, 355687428096000,
		6402373705728000, 121645100408832000,
		2432902008176640000
	};
	return F[n];
}

// Factorial sequence, static array, O(N)
long long Fac2(int n) {
	static vector<long long> F{ 1, 1 };
	if (n < 0) return 0;
	if (n + 1 >(int)F.size()) {
		for (int i = F.size(); i <= n; ++i) {
			F.push_back(F[i - 1] * i);
		}
	}
	return F[n];
}

/*Section: Array*/
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

// 229. Majority Element II
vector<int> majorityElement(vector<int>& nums) {
	int candidate1 = 0, candidate2 = 0, count1 = 0, count2 = 0;
	for (int num : nums) {
		if (num == candidate1) ++count1;
		else if (num == candidate2) ++count2;
		else {
			if (count1 == 0) { candidate1 = num; ++count1; }
			else if (count2 == 0) { candidate2 = num; ++count2; }
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
	for (int i = 0, j = matrix[0].size() - 1; i < (int)matrix.size() && j >= 0;) {
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

// 78.1. Subsets, Recursive
void genSubsets(vector<int>& nums, int start, vector<int>& sub, vector<vector<int>>& subs) {
	subs.push_back(sub);
	for (int i = start; i < (int)nums.size(); ++i) {
		sub.push_back(nums[i]);
		genSubsets(nums, i + 1, sub, subs);
		sub.pop_back();
	}
}
vector<vector<int>> subsets(vector<int>& nums) {
	sort(nums.begin(), nums.end());
	vector<vector<int>> subs;
	vector<int> sub;
	genSubsets(nums, 0, sub, subs);
	return subs;
}

// 78.2. Subsets, Iterative
vector<vector<int>> subsets2(vector<int>& nums) {
	sort(nums.begin(), nums.end());
	vector<vector<int>> subs(1, vector<int>()), tmp;
	for (int i = 0; i < (int)nums.size(); ++i) {
		tmp = subs;
		for (vector<int>& sub : tmp) sub.push_back(nums[i]);
		subs.insert(subs.end(), tmp.begin(), tmp.end());
	}
	return subs;
}

// 90.1. Subsets II, recursive
void subsetsWithDup(vector<int>& nums, int begin, vector<int> &sub, vector<vector<int> >& subs) {
	subs.push_back(sub);
	for (int i = begin; i < (int)nums.size(); ++i) {
		if (i == begin || nums[i] != nums[i - 1]) {
			sub.push_back(nums[i]);
			subsetsWithDup(nums, i + 1, sub, subs);
			sub.pop_back();
		}
	}
}
vector<vector<int> > subsetsWithDup(vector<int>& nums) {
	sort(nums.begin(), nums.end());
	vector<vector<int> > subs;
	vector<int> sub;
	subsetsWithDup(nums, 0, sub, subs);
	return subs;
}

// 90.2. Subsets II, iterative
vector<vector<int>> subsetsWithDup2(vector<int>& nums) {
	sort(nums.begin(), nums.end());
	vector<vector<int> > subs(1, vector<int>()), tmp;
	for (int i = 0; i < (int)nums.size(); ++i) {
		if (i == 0 || nums[i] != nums[i - 1]) tmp = subs;
		for (vector<int>& sub : tmp) sub.push_back(nums[i]);
		subs.insert(subs.end(), tmp.begin(), tmp.end());
	}
	return subs;
}

// 31. Next Permutation
void nextPermutation(vector<int>& nums) {
	int n = nums.size(), i;
	// reverse find the first number which breaks descending order
	for (i = n - 2; i >= 0 && nums[i] >= nums[i + 1]; --i);
	// if one breaks 1, exchange it with the least number (>this one)
	if (i >= 0) {
		int j;
		for (j = i + 1; j < n && nums[j] > nums[i]; ++j);
		swap(nums[i], nums[j - 1]);
	}
	// reverse sort the numbers after the swapped one
	++i;
	for (int j = n - 1; i < j; ++i, --j) {
		swap(nums[i], nums[j]);
	}
}

// 163. Missing Ranges
string get_range(int start, int end) {
	return (start == end ? to_string(start) : to_string(start) + "->" + to_string(end));
}
vector<string> findMissingRanges(vector<int>& nums, int lower, int upper) {
	vector<string> result;
	int pre = lower - 1;
	for (int i = 0; i <= (int)nums.size(); ++i) {
		int cur = (i == (int)nums.size() ? upper + 1 : nums[i]);
		if (cur - pre >= 2)
			result.push_back(get_range(pre + 1, cur - 1));
		pre = cur;
	}
	return result;
}

// 259. 3Sum Smaller
int threeSumSmaller(vector<int>& nums, int target) {
	sort(nums.begin(), nums.end());
	int n = nums.size(), res = 0;
	for (int i = 0; i < n - 2; ++i) {
		int front = i + 1, back = n - 1;
		while (front < back) {
			int sum = nums[i] + nums[front] + nums[back];
			if (sum < target) res += back - front, ++front;
			else --back;
		}
	}
	return res;
}

// 209. Minimum Size Subarray Sum
int minSubArrayLen(int s, vector<int>& nums) {
	int fpos = 0, sum = 0, res = INT_MAX;
	for (int i = 0; i < (int)nums.size(); ++i) {
		sum += nums[i];
		while (sum >= s) {
			res = min(res, i - fpos + 1);
			sum -= nums[fpos++];
		}
	}
	return (res == INT_MAX ? 0 : res);
}

// 64. Minimum Path Sum
int minPathSum(vector<vector<int>>& grid) {
	int m = grid.size(), n = grid[0].size();
	vector<vector<int> > sum(m, vector<int>(n, grid[0][0]));
	for (int i = 1; i < m; ++i) {
		sum[i][0] = sum[i - 1][0] + grid[i][0];
	}
	for (int j = 1; j < n; ++j) {
		sum[0][j] = sum[0][j - 1] + grid[0][j];
	}
	for (int i = 1; i < m; i++) {
		for (int j = 1; j < n; j++) {
			sum[i][j] = min(sum[i - 1][j], sum[i][j - 1]) + grid[i][j];
		}
	}
	return sum[m - 1][n - 1];
}

// 79. Word Search
bool wordSearch(vector<vector<char>>& board, string& word,
	int row, int col, int begin, int M, int N, int len) {
	char c = board[row][col];
	if (c != word[begin]) return false;
	if (begin == len - 1) return true;
	board[row][col] = '*';
	bool res = false;
	if (row > 0)
		res = wordSearch(board, word, row - 1, col, begin + 1, M, N, len);
	if (!res && row < M - 1)
		res = wordSearch(board, word, row + 1, col, begin + 1, M, N, len);
	if (!res && col > 0)
		res = wordSearch(board, word, row, col - 1, begin + 1, M, N, len);
	if (!res && col < N - 1)
		res = wordSearch(board, word, row, col + 1, begin + 1, M, N, len);
	board[row][col] = c;
	return res;
}
bool wordSearch(vector<vector<char>>& board, string word) {
	if (board.empty() || board[0].empty()) return false;
	int M(board.size()), N(board[0].size()), len(word.size());
	for (int i = 0; i<M; ++i)
		for (int j = 0; j<N; ++j)
			if (wordSearch(board, word, i, j, 0, M, N, len)) return true;
	return false;
}

// 55. Jump Game
bool canJump(vector<int>& nums) {
	int n = nums.size(), i = 0;
	for (int reach = 0; i < n && i <= reach; ++i)
		reach = max(i + nums[i], reach);
	return i == n;
}

// 45. Jump Game II
int jump(vector<int>& nums) {
	int n = nums.size();
	if (n <= 1) return 0;
	int steps = 1, i = 0, next = 0;
	for (int reach = 0; i < n && i <= reach; ++i) {
		next = max(next, i + nums[i]);
		if (next >= n - 1) return steps;
		if (i == reach) { reach = next; ++steps; }
	}
	return INT_MAX;
}

// 289. Game of Life
void gameOfLife(vector<vector<int>>& board) {
	if (board.empty() || board[0].empty()) return;
	int m = board.size(), n = board[0].size();
	for (int i = 0; i<m; ++i) {
		for (int j = 0; j<n; ++j) {
			int count = 0;
			for (int I = max(i - 1, 0); I<min(i + 2, m); ++I)
				for (int J = max(j - 1, 0); J<min(j + 2, n); ++J)
					count += board[I][J] & 1;
			if (count == 3 || count - board[i][j] == 3)
				board[i][j] |= 2;
		}
	}
	for (int i = 0; i<m; ++i)
		for (int j = 0; j<n; ++j)
			board[i][j] >>= 1;
}

// 54. Spiral Matrix
vector<int> spiralOrder(vector<vector<int>>& matrix) {
	if (matrix.empty()) return vector<int>{};
	int m = matrix.size(), n = matrix[0].size();
	vector<int> spiral(m * n);
	int u = 0, d = m - 1, l = 0, r = n - 1, k = 0;
	while (true) {
		for (int col = l; col <= r; ++col) // up->right
			spiral[k++] = matrix[u][col];
		if (++u > d) break;

		for (int row = u; row <= d; ++row) // right->down
			spiral[k++] = matrix[row][r];
		if (--r < l) break;

		for (int col = r; col >= l; --col) // down->left
			spiral[k++] = matrix[d][col];
		if (--d < u) break;

		for (int row = d; row >= u; --row) // left->up
			spiral[k++] = matrix[row][l];
		if (++l > r) break;
	}
	return spiral;
}

// 59. Spiral Matrix II
vector<vector<int> > generateMatrix(int n) {
	vector<vector<int> > mat(n, vector<int>(n));
	int u = 0, d = n - 1, l = 0, r = n - 1, num = 1;
	while (true) {
		for (int col = l; col <= r; col++) // up-> right
			mat[u][col] = num++;
		if (++u > d) break;

		for (int row = u; row <= d; row++) // right->down
			mat[row][r] = num++;
		if (--r < l) break;

		for (int col = r; col >= l; col--) // down->left
			mat[d][col] = num++;
		if (--d < u) break;

		for (int row = d; row >= u; row--) // left->up
			mat[row][l] = num++;
		if (++l > r) break;
	}
	return mat;
}

// 162. Find Peak Element
int findPeakElement(vector<int>& nums) {
	int n = nums.size(), low = 0, high = n - 1;
	while (low < high) {
		int mid = low + (high - low) / 2;
		if (mid + 1 < n && nums[mid] < nums[mid + 1])
			low = mid + 1;
		else if (mid - 1 >= 0 && nums[mid - 1] > nums[mid])
			high = mid - 1;
		else
			return mid;
	}
	return nums[low];
}

// 75. Sort Colors
void sortColors(vector<int>& nums) {
	int zero = 0, second = nums.size() - 1;
	for (int i = 0; i <= second; ++i) {
		while (nums[i] == 2 && i < second)
			swap(nums[i], nums[second--]);
		while (nums[i] == 0 && i > zero)
			swap(nums[i], nums[zero++]);
	}
}

// 11. Container with Most Water
int maxArea(vector<int>& height) {
	int water = 0, i = 0, j = height.size() - 1;
	while (i < j) {
		int h = min(height[i], height[j]);
		water = max(water, (j - i) * h);
		while (height[i] <= h && i < j) ++i;
		while (height[j] <= h && i < j) --j;
	}
	return water;
}

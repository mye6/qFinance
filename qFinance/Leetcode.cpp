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

ListNode* deleteDuplicates(ListNode* head) {
	ListNode* cur = head;
	while (cur) {
		while (cur->next && cur->val == cur->next->val)
			cur->next = cur->next->next;
		cur = cur->next;
	}
	return head;
}

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

// 189
void rotate(vector<int>& nums, int k) {
	int n = nums.size();
	reverse(nums.begin(), nums.end());
	reverse(nums.begin(), nums.begin() + k%n);
	reverse(nums.begin() + k%n, nums.end());
}

// 189.2
void rotate2(vector<int>& nums, int k) {
	int n = nums.size();
	if (n == 0 || k <= 0) return;
	vector<int> tmp(nums);
	for (int i = 0; i < n; ++i) nums[(i + k) % n] = tmp[i];
}

// 153
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
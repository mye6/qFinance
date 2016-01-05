#include "Solver.h"
#include "Leetcode.h"
/*
#include "Finance.h"
#include "Puzzle.h"

#include "ThinkCPP.h"
#include "FaqCPP.h"
*/


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
	ListNode* tmp = t1->next;
	t1->next = t1->next->next;
	delete tmp;
	return dummy.next;
}

ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
	ListNode dummy(0);
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


int main(){
	vector<int>	nums1{ 1, 3, 5, 7, 9 };
	vector<int>	nums2{ -2, 3, 4, 8, 10 };
	ListNode *l1 = genList(nums1), *l2 = genList(nums2);
	PRINT(l1);
	PRINT(l2);

	l1 = mergeTwoLists(l1, l2);
	PRINT(l1);
	

	system("pause");
	return 0;
}  
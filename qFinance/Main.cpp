#include "Solver.h"
#include "Finance.h"
#include "Leetcode.h"

// A utility function to find deepest leaf node.
// lvl:  level of current node.
// maxlvl: pointer to the deepest left leaf node found so far
// isLeft: A bool indicate that this node is left child of its parent
// resPtr: Pointer to the result

/*
void deepestLeftLeafUtil(TreeNode *root, int lvl, int *maxlvl, bool isLeft, TreeNode **resPtr) {
	// Base case
	if (root == NULL) return;
	// Update result if this node is left leaf and its level is more
	// than the maxl level of the current result
	if (isLeft && !root->left && !root->right && lvl > *maxlvl) {
		*resPtr = root;
		*maxlvl = lvl;
		return;
	}
	// Recur for left and right subtrees
	deepestLeftLeafUtil(root->left, lvl + 1, maxlvl, true, resPtr);
	deepestLeftLeafUtil(root->right, lvl + 1, maxlvl, false, resPtr);
}

// A wrapper over deepestLeftLeafUtil().
TreeNode* deepestLeftLeaf(TreeNode *root) {
	int maxlevel = 0;
	TreeNode *result = NULL;
	deepestLeftLeafUtil(root, 0, &maxlevel, false, &result);
	return result;
}
*/


int main() {	
	for (int i = 0; i < 100; ++i)
		PRINT(randi(5));

	system("pause");
	return 0;
}
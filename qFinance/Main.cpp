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



/*
srand((unsigned int)time(NULL));
// Use a different seed value so that we don't get same
// result each time we run this program
for (int i = n - 1; i > 0; --i) {
	int j = rand() % (i + 1); // Pick a random index from 0 to i
	*/
int main() {	
	TreeNode* root = new TreeNode(1);
	root->left = new TreeNode(2);
	root->right = new TreeNode(3);
	root->left->left = new TreeNode(4);
	root->right->left = new TreeNode(5);
	root->right->right = new TreeNode(6);
	root->right->left->right = new TreeNode(7);
	root->right->right->right = new TreeNode(8);
	root->right->left->right->left = new TreeNode(9);
	root->right->right->right->right = new TreeNode(10);

	TreeNode *result = deepestLeftLeaf(root);
	if (result)
		cout << "The deepest left child is " << result->val << endl;
	else
		cout << "There is no left leaf in the given tree";

	system("pause");
	return 0;
}
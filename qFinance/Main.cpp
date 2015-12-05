#include "Solver.h"
#include "Finance.h"
#include "Leetcode.h"
#include "LinkedList.h"





int main() {
	List<int>    theList;
	ListItr<int> theItr = theList.zeroth();
	printList(theList);

	for (int i = 0; i < 10; i++) {
		theList.insert(i, theItr);
		printList(theList);
		theItr.advance();
	}

	for (int i = 0; i < 10; i += 2)
		theList.remove(i);

	for (int i = 0; i < 10; i++)
		if ((i % 2 == 0) != (theList.find(i).isPastEnd()))
			cout << "Find fails!" << endl;

	cout << "Finished deletions" << endl;
	printList(theList);

	List<int> list2;
	list2 = theList;
	printList(list2);
	

	system("pause");
	return 0;
}
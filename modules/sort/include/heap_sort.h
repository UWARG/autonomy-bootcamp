#ifndef __HEAP_SORT_H__
#define __HEAP_SORT_H__

// heap sort using a min heap
class HeapSort{
    private:
    // fields
    int *theHeap; // copy of original array, obeys heap structure
    int *aryOriginal; // pointer of original array, eventually be sorted
    const int aryLen; // size of original array
    int heapSize; // marks the size of Heap, modified when pop() is invoked
    // methods
    void heapify(); // bubble down from the lowest heap. o(n)
    void pop(); // pop the smallest item, deduct heapSize by 1, O(log n) 
    void bubble_down(int); // move a node to correct index in subheap, O(log n)
    int get_left(int) const; // get index of left child, 0->no left, O(1)
    int get_right(int) const; // get index of right child, 0->no right, O(1)
    int get_parent(int) const; // get index of parent, -1->no parent, O(1)
    int get_sibling(int) const; // get index of node with same parent, 0->no sibling, O(1)
    public:
    HeapSort(int *, int ); // ctor
    explicit HeapSort(const HeapSort &); // copy ctor
    ~HeapSort(); // dtor
    void sort(); // sorts an int[]
};
#endif

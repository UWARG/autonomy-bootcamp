/**
 * @file heap_sort.h
 * @auhor Bingzheng Feng
 * 
 * @section LICENSE
 * 
 * Copyright (c) 2016-2017, Waterloo Aerial Robotics Group (WARG)
 * All rights reserved.
 * 
 * This software is licensed under a modified version of the BSD 3 clause license
 * that should have been included with this software in a file called COPYING.txt
 * Otherwise it is available at:
 * https://raw.githubusercontent.com/UWARG/computer-vision/master/COPYING.txt 
 */
#ifndef __HEAP_SORT_H__
#define __HEAP_SORT_H__

/** 
 * heap sort using a min heap
 * @class HeapSort
 * 
 * Sorting an integer array by heapsort convention
 * Memory usage: O(n), n is the length of input array
 * Capable for arrays with length less than 2147483647 
 *
 * @brief:
 * readin an array A, call ctor HeapSort H(A,len), len is length of A.call H.sort(), done.
 */
class HeapSort{
    private:
    /**
     * @brief A copy of original array, obeys heap sort structure.
     */
    int *theHeap;

    /**
     * @brief A pointer to original array, eventually will be sorted.
     */
    int *aryOriginal;

    /**
     * @brief The size of original array.
     */
    const int aryLen;

    /**
     * Marks the size of theHeap, modifed when pop() is invoked. May not be 
     * larger than aryLen
     */
    int heapSize;

    /**
     * Bubbles down from the smallest heap up to the top node.
     * Runtime: O(n)
     * Inplace modification.
     */
    void heapify();

    /**
     * Pop the smallest item from heap, deduct heapSize by 1.
     * Runtime: O(log n)
     * Does not shrink the memory immediately.
     */
    void pop();

    /**
     * Maintain the heap structure of the heap with given index.
     *
     * @param the array index of the top node from a subheap.
     */
    void bubble_down(int);

    /**
     * Get the left child's index of the node at given index.
     * Runtime: O(1)
     *
     * @param the array index of the parent node
     *
     * @returns returns left child's index. 0 means it doesn't have a left.
     */
    int get_left(int) const;

    /**
     * Get the right child's index of the node at given index.
     * Runtime: O(1)
     *
     * @param the aray index of the parent node
     * 
     * @returns returns right child's index.0 means it doesn't have a right.
     */
    int get_right(int) const;

    /**
     * Get the parent node's index of the node at given index.
     * Runtime: O(1)
     * 
     * @param the array index of the child node.
     *
     * @returns returns the parent's index. -1 means it has no parent.
     */
    int get_parent(int) const;

    /**
     * Get the index of another node with same parent as given node.
     * Runtime: O(1)
     *
     * @param the array index of the child node.
     *
     * @returns returns the sibling's index regardless right or left. 0 means
     * it has no sibling.
     */
    int get_sibling(int) const;
    public:

    /**
     * Constructor for HeapSort
     * @param an int array, a length of int array
     */
    HeapSort(int *, int );

    /**
     * Copy constructor for HeapSort
     */
    explicit HeapSort(const HeapSort &);

    /**
     * Destructor for HeapSort
     */
    ~HeapSort();

    /**
     * The actual sorting fuction of HeapSort
     *
     * eventually copies the sorted array back to original arary
     * Runtime: O(n log(n))
     */
    void sort(); // sorts an int[]
};
#endif

/**
 * @file merge_sort.cpp
 * @author Nolan Kornelsen
 *
 * @section LICENSE
 *
 *  Copyright (c) 2015, Waterloo Aerial Robotics Group (WARG)
 *  All rights reserved.
 *
 *  This software is licensed under a modified version of the BSD 3 clause license
 *  that should have been included with this software in a file called COPYING.txt
 *  Otherwise it is available at:
 *  https://raw.githubusercontent.com/UWARG/computer-vision/master/COPYING.txt
 */

#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

void merge_sort(vector<int> *list) {
    size_t startIdx = 0;
    size_t endIdx = list->size() - 1;
    if (list->size() <= 1) {
        return;
    }
    size_t midIdx = startIdx + endIdx / 2;

    vector<int> left(1 + midIdx - startIdx);
    vector<int> right(endIdx - midIdx);
    copy(list->begin() + startIdx, list->begin() + midIdx + 1, &left.at(0));
    // copy(list.at(midIdx+1), list.at(endIdx+1), &right.at(0));
    copy(list->begin() + midIdx + 1, list->begin() + endIdx + 1, &right.at(0));

    merge_sort(&left);
    merge_sort(&right);

    // two lists now sorted, now merge
    int leftIdx = 0;
    int rightIdx = 0;
    int listIdx = 0;

    while (leftIdx < left.size() || rightIdx < right.size()) {
        // while there are numbers remaining
        // check left array
        if (leftIdx < left.size()) {
            if (rightIdx >= right.size() || left[leftIdx] < right[rightIdx]) {
                (*list)[listIdx++] = left[leftIdx++];
            }
        }
        // check right array
        if (rightIdx < right.size()) {
            if (leftIdx >= left.size() || right[rightIdx] <= left[leftIdx]) {
                (*list)[listIdx++] = right[rightIdx++];
            }
        }
    }
}
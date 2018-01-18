/**
 * @file merge_sort.h
 * @author WARG
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

#ifndef MERGE_SORT_H_INCLUDED
#define MERGE_SORT_H_INCLUDED

#include <vector>

/**
 *  @brief Sorts the elements in the given array
 *
 *  A recursive implementation of merge sort to sort the elements in
 *  the given array
 *
 *  @param arr the array to be sorted
 *  @param low the starting(lowest) index from which to sort
 *  @param high the last(highest) index to which to sort (inclusive)
 *  @return the sorted array
 */

std::vector<int> merge_sort(std::vector<int> &arr, int low, int high);

/**
 *  @brief A helper function for the merge_sort function
 *
 *  A helper function for the merge_sort function. Carries out the
 *  "merge" step of merge sort
 *
 *  @param arr the array to be sorted
 *  @param low the starting(lowest) index from which to sort
 *  @param high the last(highest) index to which to sort (inclusive)
 *  @return the sorted array
 */

std::vector<int> merge(std::vector<int> &arr, int low, int mid, int high);


#endif // MERGE_SORT_H_INCLUDED

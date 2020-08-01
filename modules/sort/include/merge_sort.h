/**
 * @file merge_sort.h
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

#ifndef MERGE_SORT_H_INCLUDED
#define MERGE_SORT_H_INCLUDED

#include <vector>

/**
 * @brief Performs an in-place merge sort on a list of integers
 * 
 * @param list a pointer to the list of integers to be sorted
 */
void merge_sort(std::vector<int> *list);

#endif // MERGE_SORT_H_INCLUDED
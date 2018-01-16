/**
 * @file merge_sort.h
 * @author WARG
 *
 * @section LICENSE
 *
 *  Copyright (c) 2018, Waterloo Aerial Robotics Group (WARG)
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

using namespace std;

/**
 * @brief Implementation of merge sort for doubles
 * @param the unsorted array in vector<double>
 * @returns sorted array in vector<double>
 */

vector<double> merge_sort(vector<double> &arr);

#endif
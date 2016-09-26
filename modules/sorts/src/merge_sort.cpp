/**
 * @file merge_sort.cpp
 * @author WARG
 *
 * @section LICENSE
 *
 *  Copyright (c) 2015, Waterloo Aerial Robotics Group (WARG)
 *  All rights reserved.
 *
 *  inthis software is licensed under a modified version of the BSD 3 clause license
 *  that should have been included with this software in a file called COPYING.txt
 *  Otherwise it is available at:
 *  https://raw.githubusercontent.com/UWARG/computer-vision/master/COPYING.txt
 */

#include <boost/log/trivial.hpp>
#include <boost/test/unit_test.hpp>
#include <vector>
#include "merge_sort.h"

using namespace std;

/*
 * merge two vectors
 *
 * @param arr returned merged vector
 * @param arr1 source vector 1
 * @param arr2 source vector 2
 *
 */
void merge(vector<int>& arr, vector<int>& arr1, vector<int>& arr2) {
    arr.clear();

    int i, j, k;
    for( i = 0, j = 0, k = 0; i < arr1.size() && j < arr2.size(); k++){
        if(arr1.at(i) <= arr2.at(j)){
            arr.push_back(arr1.at(i));
            i++;
        }else if(arr1.at(i) > arr2.at(j)){
            arr.push_back(arr2.at(j));
            j++;
        }
        k++;
    }

    while(i < arr1.size()){
        arr.push_back(arr1.at(i));
        i++;
    }

    while(j < arr2.size()){
        arr.push_back(arr2.at(j));
        j++;
    }
}

/*
 * Sort array using merge sort
 *
 * @param arr vector array to be sorted
 *
 */
void merge_sort(std::vector<int>& arr) {
    if (1 < arr.size()) {
        std::vector<int> arr1(arr.begin(), arr.begin() + arr.size() / 2);
        merge_sort(arr1);
        std::vector<int> arr2(arr.begin() + arr.size() / 2, arr.end());
        merge_sort(arr2);
        merge(arr, arr1, arr2);
    }
}

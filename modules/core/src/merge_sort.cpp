/** @file merge_sort.cpp
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

#include <vector>
#include <iostream>
#include <cassert>


/**
 * @brief consumes two sorted vectors to produce one sorted vector containing elements of both input vectors
 *
 * @param v1 is the first of two int vectors to be merged
 * @param v2 is the second of two int vector to be merged
 * @return result, a sorted vector containing all elements of both v1 and v2
 */

std::vector<int> merge(std::vector<int> v1, std::vector<int> v2){
    std::vector<int> result;
    while(v1.size() != 0 || v2.size() != 0){
        if (v1.empty()){
            result.insert(result.end(), v2.begin(), v2.end());
            v2.clear();
        }
        else if (v2.empty()){
            result.insert(result.end(), v1.begin(), v1.end());
            v1.clear();

        }
        else if (v1[0] > v2[0]){
            result.insert(result.end(), v2[0]);
            assert(!v2.empty());
            v2.erase(v2.begin());
        }
        else if (v2[0] > v1 [0]){
            result.insert(result.end(), v1[0]);
            assert(!v1.empty());
            v1.erase(v1.begin());
        }
        else {
            result.insert(result.end(), v1[0]);
            result.insert(result.end(), v2[0]);
            assert(!v2.empty());
            assert(!v1.empty());
            v2.erase(v2.begin());
            v1.erase(v1.begin());
        }
    }
    return result;
}


/**
 *
 * @brief sorts an int vector
 *
 * @param v is the int vector to be sorted in mergeSort
 * @return a sorted int vector containing all elements in v
 */
std::vector<int> mergeSort(std::vector<int> v){
    if (v.size() == 1){
        return v;
    }

    std::vector<int> v1(v.begin(), v.begin() + v.size()/2);
    std::vector<int> v2( v.begin() + v.size()/2, v.end());

    v1 = mergeSort(v1);
    v2 = mergeSort(v2);

    return merge(v1, v2);
}


/**
 * @file merge_sort_test_suite.h
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

#define BOOST_TEST_MODULE SortModule
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <cstdlib>
#include "merge_sort.h"

BOOST_AUTO_TEST_SUITE(MergeSortTestSuite);

BOOST_AUTO_TEST_CASE(MergeSortTest) {
    int n = rand() % 100 + 1;
    std::vector<int> testVector;
    for (int i = 0; i < n; i++) {
        testVector.push_back(rand() % 1000 + 1);
    }
    std::cout << "Sorting list: ";
    for (int i : testVector) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
    merge_sort(&testVector);
    std::cout << "Sorted list: ";
    for (int i : testVector) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
    // test if sorted
    for (int i = 0; i < testVector.size() - 1; i++) {
        BOOST_CHECK(testVector[i] <= testVector[i+1]);
    }
}

BOOST_AUTO_TEST_SUITE_END();
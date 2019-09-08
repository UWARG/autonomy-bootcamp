//
// Created by huber stark on 9/7/19.
//
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mergesort_test
#include <boost/test/unit_test.hpp>
#include "../include/merge_sort.h"
#include <iostream>
using namespace std;
BOOST_AUTO_TEST_SUITE(merge_sort_test_suite);
BOOST_AUTO_TEST_CASE(merge_sort_test_case1){
    int arr[] = {3,4,5,1,7,8,12};
    int sorted[] = {1,3,4,5,7,8,12};
    int *temp = new int[sizeof(arr)/ sizeof(arr[0])];
    msort(arr, 0, (sizeof(arr)/ sizeof(arr[0]))-1, temp);
    bool flag = 0;
    for(int i=0; i<sizeof(arr)/ sizeof(arr[0]); i++){
        if(arr[i] != sorted[i]){
            flag = 1;
        }
        BOOST_CHECK(flag == 0);
    }

    cout<<"Mergesort test complete"<<endl;
}
BOOST_AUTO_TEST_SUITE_END();

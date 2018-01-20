#ifndef BOOST_TEST_DYN_LINK
#define MERGE_SORT_MODULE MERGE_SORT_TEST
#include <boost/test/unit_test.hpp>
#include <merge_sort.h>

BOOST_TEST_AUTO_CASE( merge_test ){
        std::vector<int> v1 = {1, 1, 1, 1};
        std::vector<int> v2 = {0, 2, 2, 3};
        std::vector<int> v3 = {0, 1, 1, 1, 1, 2, 2, 3};

        v1 = merge(v1, v2);

        BOOST_CHECK ( v1 == v3);
}

BOOST_TEST_AUTO_CASE (sort_test){
        std::vector<int> v1 = { 10, 9, 8, 7, 6, 5};
        std::vector<int> v2 = { 5, 6, 7, 8, 9, 10};

        v1 = mergeSort(v1);

        BOOST_CHECK (v1 == v2);
}



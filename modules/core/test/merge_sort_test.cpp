#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE MERGE_SORT_TEST
#include <boost/test/unit_test.hpp>
#include "merge_sort.h"
#include <algorithm>


BOOST_AUTO_TEST_CASE(integer_sorting){
    std::vector<int> a = {3, 2, 2, 1};
    std::vector<int> sorted = {1, 2, 2, 3};

    a = merge_sort<int>(a, [](int x1, int x2) -> int {
        if(x1 < x2){
            return -1;
        }else if(x1 == x2){
            return 0;
        }else{
            return 1;
        }
    });

    BOOST_CHECK(a == sorted);
}


BOOST_AUTO_TEST_CASE(float_sorting) {
    std::vector<float> a = {3.32, 2.21, 2.64, 1.005};
    std::vector<float> sorted = {1.005, 2.21, 2.64, 3.32};

    a = merge_sort<float>(a, [](float x1, float x2) -> int {
        if(x1 < x2){
            return -1;
        }else if(x1 == x2){
            return 0;
        }else{
            return 1;
        }
    });

    BOOST_CHECK(a == sorted);
}



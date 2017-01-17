#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE merge_sort_test
#include <boost/test/unit_test.hpp>
#include "contour_comparison.h"
#include "merge_sort.h"

BOOST_AUTO_TEST_SUITE(MERGE_SORT_SUIT);

BOOST_AUTO_TEST_CASE(BASIC_TEST_1){
        std::vector<int> a = {4, 1, 6, 7, 3, 2, 1, 5, -5, 12, 51, 643, 31, 7, 3, 63, 42, 26, 24, 63};
        a = merge_sort<int>(a, [](int a, int b)->int{
            if(a < b){
                return -1;
            }else if(a == b){
                return 0;
            }else{
                return 1;
            }
        });
        std::vector<int> sorted = {-5, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 12, 24, 26, 31, 42, 51, 63, 63, 643 };
        BOOST_TEST(sorted == a);
    }




BOOST_AUTO_TEST_SUITE_END();
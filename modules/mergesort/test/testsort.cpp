#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE sort_tests
#include <boost/test/unit_test.hpp>
#include <algorithm>
#include "mergesort.h" 

BOOST_AUTO_TEST_SUITE(merge_sort_test);
BOOST_AUTO_TEST_CASE(merge_sort_test_case1) {
    int test[] = {-5,2,6,4,12,-7};
    int testans[] = {-7,-5,2,4,6,12};
    merge_sort(test,sizeof(test)/sizeof(test[0]));
    BOOST_CHECK(std::equal(test,test+6,testans));
}
BOOST_AUTO_TEST_SUITE_END();

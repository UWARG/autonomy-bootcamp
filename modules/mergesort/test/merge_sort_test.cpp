#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE sort_tests
#include <boost/test/unit_test.hpp>
#include "/media/sf_ubuntu-sf/UWARG/computer-vision-bootcamp/modules/mergesort/include/merge_sort.h"
#include <vector>
BOOST_AUTO_TEST_SUITE(merge_sort_test_suite);
BOOST_AUTO_TEST_CASE(merge_sort_test_case1) {
    std::vector<float> uns {1.3, 6.5, 2.4 ,2 ,6, 99, 14, 21};
    std::vector<float> sorted {1.3, 2, 2.4, 5, 6.5, 14, 21, 99};
    merge_sort(uns);
    BOOST_CHECK(uns == sorted);
}
BOOST_AUTO_TEST_SUITE_END();
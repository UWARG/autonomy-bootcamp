#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestSuiteName
#include <boost/test/unit_test.hpp>
#include "merge_sort.h" // declares type foo, header is in the core module
#include <vector>

BOOST_AUTO_TEST_CASE(TestName) {
    std::vector<int> uns {1, 5, 2 ,2 ,6, 99, 14, 21};
    std::vector<int> sorted {1, 2, 2, 5, 6, 14, 21, 99};
    merge_sort(uns);
    BOOST_CHECK(uns == sorted);
}

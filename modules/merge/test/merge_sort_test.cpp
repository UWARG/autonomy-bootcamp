#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SortTest
#include <boost/test/unit_test.hpp>
#include "merge_sort.h"
#include <vector>

using namespace std;

BOOST_AUTO_TEST_SUITE(SortTestSuite);

BOOST_AUTO_TEST_CASE(SortTestCase1){
	vector <double> unsorted = {0, 1.5, 3.14, 0, -1000, 10000, 10};
	vector <double> sorted = {-1000, 0, 0, 1.5, 3.14, 10, 10000};
	merge_sort(unsorted);
	BOOST_CHECK(unsorted == sorted);
}

BOOST_AUTO_TEST_SUITE_END();
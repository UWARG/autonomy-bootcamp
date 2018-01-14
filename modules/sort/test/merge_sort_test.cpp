#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE sort
#include <boost/test/unit_test.hpp>
#include "merge_sort.h" // declares type foo, header is in the core module

BOOST_AUTO_TEST_SUITE(Sort);

BOOST_AUTO_TEST_CASE( sort_test )
{

  // create a test vector to sort
  std::vector<int> example = std::vector<int>();
  example.push_back(4);
  example.push_back(1);
  example.push_back(3);
  example.push_back(6);
  example.push_back(5);
  example.push_back(2);
  example.push_back(2);
  example.push_back(6);

  int max = example.size();

  merge_sort(example, 0, max - 1);


  for(int i = 1; i < max; i++){
    BOOST_TEST(example[i-1] <= example[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END();


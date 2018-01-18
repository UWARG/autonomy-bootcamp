/*
    This file is part of WARG's computer-vision

    Copyright (c) 2015, Waterloo Aerial Robotics Group (WARG)
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. Usage of this code MUST be explicitly referenced to WARG and this code
       cannot be used in any competition against WARG.
    4. Neither the name of the WARG nor the names of its contributors may be used
       to endorse or promote products derived from this software without specific
       prior written permission.

    THIS SOFTWARE IS PROVIDED BY WARG ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL WARG BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SORT
#include <boost/test/unit_test.hpp>
#include "merge_sort.h"

using namespace boost;

BOOST_AUTO_TEST_CASE(MergeSortTest)
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

  // this is the expected output
  std::vector<int> soln = {1, 2, 2, 3, 4, 5, 6, 6};

  int max = example.size();

  merge_sort(example, 0, max - 1);


  for(int i = 0; i < max; i++){
    BOOST_TEST(example[i] == soln[i]);
  }

  BOOST_TEST_MESSAGE("Test completed");
}

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Sort
#define SUPERLONG_TEST 0
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <fstream>
#include "heap_sort.h"

bool test_func(std::string testName){
    std::ifstream inputFile;
    std::ifstream expectedOut;
    int aryLen = 0;
    int numTemp;
    int *aryToBeSort = nullptr;
    int *aryExpected = nullptr;
    inputFile.open(testName + ".in");
    expectedOut.open(testName + ".exp");
    // readin size of array
    inputFile >> numTemp;
    if(numTemp > aryLen){
        // init two array
        aryToBeSort = new int[numTemp];
	aryExpected = new int[numTemp];
	aryLen = numTemp;
	// read in input
	for(int i = 0; i < aryLen; i++){
	    inputFile >> numTemp;
	    aryToBeSort[i] = numTemp;
	    expectedOut >> numTemp;
	    aryExpected[i] = numTemp;
	}
	// sort
	HeapSort h(aryToBeSort, aryLen);
	h.sort();
	// check if it is sorted correctly
	bool match = true;
	for(int i = 0; i < aryLen; i++){
	    if(aryToBeSort[i] - aryExpected[i] != 0){
	        match = false;
		break;
	    }
	}
	delete[] aryToBeSort;
	delete[] aryExpected;
	return match;
    }
}

BOOST_AUTO_TEST_SUITE(GeneralTest);

BOOST_AUTO_TEST_CASE(Negative) {
    std::string name = "Negative";
    BOOST_CHECK(test_func(name));
}

BOOST_AUTO_TEST_CASE(Large) {
    BOOST_CHECK(test_func("Large"));
}

BOOST_AUTO_TEST_CASE(Integer) {
    BOOST_CHECK(test_func("Integer"));
}

BOOST_AUTO_TEST_CASE(Long) {
    BOOST_CHECK(test_func("Long"));
}

#if SUPERLONG_TEST

BOOST_AUTO_TEST_CASE(Superlong) {
    BOOST_CHECK(test_func("Superlong"));
}

#endif

BOOST_AUTO_TEST_SUITE_END();

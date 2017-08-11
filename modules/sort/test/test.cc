#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Sort
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <fstream>
#include "heap_sort.h"

BOOST_AUTO_TEST_SUITE(MySuite)

BOOST_AUTO_TEST_CASE(allTests) {
    std::ifstream suiteFile("suite.txt");
    std::string testName;
    std::ifstream inputFile;
    std::ifstream expectedOut;
    int aryLen = 0;
    int numTemp;
    int *aryToBeSort = nullptr;
    int *aryExpected = nullptr;
    while(suiteFile >> testName){
        inputFile.open(testName + ".in");
	expectedOut.open(testName + ".exp");
	// readin size of array
	inputFile >> numTemp;
	if(numTemp > aryLen){
	    // init two array
	    if(aryToBeSort != nullptr){
	        delete[] aryToBeSort;
		delete[] aryExpected;
	    }
	    aryToBeSort = new int[numTemp];
	    aryExpected = new int[numTemp];
	}
	aryLen = numTemp;
	// read in input
	for(int i = 0; i < aryLen; i++){
	  inputFile >> numTemp;
	  aryToBeSort[i] = numTemp;
	  expectedOut >> numTemp;
	  aryExpected[i] = numTemp;
	}
	// close files for next open
	inputFile.close();
	inputFile.clear();
	expectedOut.close();
	expectedOut.clear();
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
	BOOST_CHECK(match);
    }
    delete[] aryToBeSort;
    delete[] aryExpected;
}

BOOST_AUTO_TEST_SUITE_END();

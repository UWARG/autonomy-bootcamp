/**
 * @file merge_sort.cpp
 * @author WARG
 *
 * @section LICENSE
 *
 *  Copyright (c) 2018, Waterloo Aerial Robotics Group (WARG)
 *  All rights reserved.
 *
 *  This software is licensed under a modified version of the BSD 3 clause license
 *  that should have been included with this software in a file called COPYING.txt
 *  Otherwise it is available at:
 *  https://raw.githubusercontent.com/UWARG/computer-vision/master/COPYING.txt
 */

#include <vector>
#include <iostream>
#include <merge_sort.h>


using namespace std;

void split_merge(vector<double> & a, int index, int n){
	if (n <= 1) return;
	int middle = n/2;
	int lowerIndex = index;
	int upperIndex = index + middle;
	split_merge(a, lowerIndex, middle);
	split_merge(a, upperIndex, n - middle);
	
	vector <double> temp;
	while (lowerIndex < index + middle && upperIndex < index + n){
		if (a[lowerIndex] <= a[upperIndex]){
			temp.push_back(a[lowerIndex++]);	
		} else{
			temp.push_back(a[upperIndex++]);
		}
	}

	while (lowerIndex < index + middle) temp.push_back(a[lowerIndex++]);
	while (upperIndex < index + n) temp.push_back(a[upperIndex++]);
	for (int i = 0; i < temp.size(); i++){
		a[index + i] = temp[i];
	}
}

vector<double> merge_sort(vector<double> & arr){
	split_merge(arr, 0 , arr.size());
	return arr;
}

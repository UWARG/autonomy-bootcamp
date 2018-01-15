/**
 * @file contour_comparison.cpp
 * @author WARG
 *
 * @section LICENSE
 *
 *  Copyright (c) 2015, Waterloo Aerial Robotics Group (WARG)
 *  All rights reserved.
 *
 *  This software is licensed under a modified version of the BSD 3 clause license
 *  that should have been included with this software in a file called COPYING.txt
 *  Otherwise it is available at:
 *  https://raw.githubusercontent.com/UWARG/computer-vision/master/COPYING.txt
 */

#include <vector>
#include <boost/log/trivial.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include "merge_sort.h"

using namespace std;

template <class T>
vector<T> merge_sort(vector<T> &arr, int low, int high){
  int mid;
  if(low < high){
    mid = std::floor((low + high) / 2);
    merge_sort(arr, low, mid);
    merge_sort(arr, mid + 1, high);
    merge(arr, low, mid, high);
  }
  return arr;
}

template <class T>
vector<T> merge(vector<T> &arr, int low, int mid, int high){

  vector<int> arr1 = vector<T>();
  vector<int> arr2 = vector<T>();

  for(int i = low; i <= mid; i++){
    arr1.push_back(arr[i]);
  }

  for(int i = mid + 1; i <= high; i++){
    arr2.push_back(arr[i]);
  }

  int index1 = 0;
  int index2 = 0;

  int arr1_size = arr1.size();
  int arr2_size = arr2.size();

  int index = low;

  while(index1 < arr1_size && index2 < arr2_size){
    if(arr1[index1] < arr2[index2]){
      arr[index] = arr1[index1];
      index1++;
    }
    else{
      arr[index] = arr2[index2];
      index2++;
    }
    index++;
  }

  while(index1 < arr1_size){
    arr[index] = arr1[index1];
    index1++;
  }

  while(index2 < arr2_size){
    arr[index] = arr2[index2];
    index2++;
  }
  return arr;
}

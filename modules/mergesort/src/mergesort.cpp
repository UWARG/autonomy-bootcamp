/*
 * This file is part of WARG's computer-vision
 *
 * Copyright (c) 2015-2016, Waterloo Aerial Robotics Group (WARG)
 * All rights reserved.
 *
 * This software is licensed under a modified version of the BSD 3 clause license
 * that should have been included with this software in a file called COPYING.txt
 * Otherwise it is available at:
 * https://raw.githubusercontent.com/UWARG/computer-vision/master/COPYING.txt
 */

#include <iostream>
#include <boost/log/trivial.hpp>
#include <boost/test/unit_test.hpp>
#include "mergesort.h"

using namespace std;

void merge_sort_helper(int a[], int n, int aux[], int low, int high);
void merge(int a[], int n, int aux[], int low, int mid, int high);

void merge_sort(int a[], int n){
    int aux[n];
    for (int i=0; i<n; i++){
        aux[i]=0;
    }

    merge_sort_helper(a,n,aux,0, n-1);
}

void merge_sort_helper(int a[], int n,int aux[],int low,int high){
    if (low>=high){
        return;
    }
    int mid=low + (high-low)/2;
    merge_sort_helper(a, n, aux, low, mid);
    merge_sort_helper(a, n, aux, mid+1, high);
    merge(a, n, aux, low, mid, high);
}

void merge(int a[], int n, int aux[], int low, int mid,int high){
    for (int i=low; i <=high;i++){
        aux[i]=a[i];
    }
    int b=low;
    int c=mid+1;
    for (int i=low; i<=high; i++){
        if (b>mid){
            a[i] = aux[c++];
        }else if (c>high){
            a[i]=aux[b++];
        }else if (aux[b] < aux[c]){
            a[i] = aux[b++];
        }else{
            a[i] = aux[c++];
        }
    }
}



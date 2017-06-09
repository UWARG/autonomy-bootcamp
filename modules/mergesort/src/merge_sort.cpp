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

#include <vector>

std::vector<float> merge(std::vector<float>& a,std::vector<float> & b){
    std::vector<float> c;
    while(a.size()!=0 && b.size()!= 0){
        if (a[0] < b[0]) {
            c.push_back(a[0]);
             a.erase(a.begin());
        } else {
            c.push_back(b[0]);
            b.erase(b.begin());
        }
    }
    while(a.size()!= 0) {
        c.push_back(a[0]);
        a.erase(a.begin());
    }

    while(b.size()!= 0) {
        c.push_back(b[0]);
        b.erase(b.begin());
    }
    return c;
}

std::vector<float> merge_sort(std::vector<float>& a){
    if (a.size() == 1){
        return a;
    }
    int halfLength = (int) a.size()/2;
    std::vector<float> leftHalf(a.begin(), a.begin() + halfLength);
    std::vector<float> rightHalf(a.begin() + halfLength, a.end());

    rightHalf = merge_sort(rightHalf);
    leftHalf = merge_sort(leftHalf);
    a = merge(leftHalf,rightHalf);
    return a;
}
// int main(){
// 	// Following code was for debugging.
//     // float arr[] = {14.2,14,7,6,18,10.2};
//     // std::vector<float> v(arr ,arr + sizeof(arr)/sizeof(arr[0]) );
      		
//     // for(auto i = v.begin(); i !=v.end() ; i += 1){
//     //     std::cout<< *i <<' ';
//     // }
//     // merge_sort(v);
  
//     // std::cout <<std::endl;
//     // for(auto i = v.begin() ; i != v.end() ; i += 1){
//     //     std::cout<< *i <<' ';
//     // }
// }

#include <iostream>

using namespace std;

void merge_sort(int arr[], int leftIndex, int rightIndex);
void merge_sorted_arrays(int arr[], int leftIndex, int rightIndex, int middleIndex);
void printArray(int arr[], int arrSize);
int main();

//problem: initial problem was that it is difficult to return a local array in c++
//Date: 10/24/2018
//-solution - separate array using only indexes - so you dont actually return any arrays, but only making modifications on top of the original array

void merge_sort(int arr[], int leftIndex, int rightIndex) { //you are not passing a new array, but rather the original one but with index restriction (in a sense)

    if (leftIndex < rightIndex) { //if both are 0, then its a size 1 array

        int middleIndex = (leftIndex + rightIndex) / 2; //if size is 7, we get 7/2 = 3.5 = 3 as middle index -> [0,3] = size 4 , [4,6] = size 3 -> so left side will always be bigger than right side

        merge_sort(arr, 0, middleIndex);
        merge_sort(arr, middleIndex + 1, rightIndex); 
        
        //after it has all been divided into arrays of size 1, we have to merge them together in order 
        //(in this case, changing directly on top of the array)
        merge_sorted_arrays(arr, leftIndex, rightIndex, middleIndex);

    }
}

void merge_sorted_arrays(int arr[], int leftIndex, int rightIndex, int middleIndex) {

    int leftSize = middleIndex - leftIndex + 1;
    int rightSize = rightIndex - middleIndex;

    int leftArr[leftSize];
    int rightArr[rightSize];

    for (int i = 0; i < leftSize; ++i) {
        leftArr[i] = arr[i + leftIndex];
    }
    for (int i = 0; i < rightSize; ++i) {
        rightArr[i] = arr[i + middleIndex + 1];
    }

    int i = 0, j = 0, k = leftIndex;

    while (i < leftSize && j < rightSize) {
        if (leftArr[i] < rightArr[j]) {
            arr[k] = leftArr[i];
            ++i;
        }
        else {
            arr[k] = rightArr[j];
            ++j;
        }
        ++k;
    }

    while (i < leftSize) {
        arr[k] = leftArr[i];
        ++i;
        ++k;
    }
    while (j < rightSize) {
        arr[k] = rightArr[j];
        ++j;
        ++k;
    }
}

void printArray(int arr[], int arrSize) 
{ 
    cout << "Sorted array: ";
    for (int i = 0; i < arrSize; ++i) 
        cout << arr[i] << " ";
} 

int main() {
    int arr[] = {12, 11, 13, 5, 6, 7}; 
    int arrSize = sizeof(arr)/sizeof(arr[0]); //sizeof array gives bits used, so u must divide by size of individual types
  
    merge_sort(arr, 0, arrSize - 1);

    printArray(arr, arrSize); 

    return 0; 
}

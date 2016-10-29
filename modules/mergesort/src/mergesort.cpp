#include <iostream>  
#include "mergesort.h"

using namespace std;

int main(void){
    int test[] = {231,2,34,123,421,654,23,9,23,4,-435,2,-23};
    MergeSort(test,sizeof(test)/sizeof(test[0]));

    for (int i=0; i < sizeof(test)/sizeof(test[0]); i++){
        printf("%d, ", test[i]);
    }
    printf("\n");
    return 0;
}

void MergeSort(int a[], int n){
    int aux[n];
    for (int i =0; i < n; i++){
        aux[i]=0;
    }

    MergeSortHelper(a,n,aux,0, n-1);
}

void MergeSortHelper(int a[], int n,int aux[],int low,int high){
    if (low>=high){
        return;
    }
    int mid = low + (high-low)/2;
    MergeSortHelper(a, n,aux,low,mid);
    MergeSortHelper(a,n,aux,mid+1,high);
    Merge(a,n,aux,low,mid,high);
}

void Merge(int a[], int n, int aux[], int low, int mid,int high){
    for (int i=low; i <=high;i++){
        aux[i]=a[i];
    }
    int b=low;
    int c=mid+1;
    for (int i =low; i <=high; i++){
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



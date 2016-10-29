#ifndef MERGESORT_H
#define MERGESORT_H

void MergeSort(int a[], int n);
void MergeSortHelper(int a[], int n, int aux[], int low, int high);
void Merge(int a[], int n, int aux[], int low, int mid, int high);

#endif

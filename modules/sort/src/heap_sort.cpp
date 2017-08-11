#include "heap_sort.h"
#include <utility>
//class HeapSort{
//private:
//int *ary_cpy,aryOriginal;
//int aryLen
//int heapSize
void HeapSort::bubble_down(int index){
    int selfIndex = index;
    int leftIndex = get_left(selfIndex);
    int rightIndex = get_right(selfIndex);
    while(leftIndex || rightIndex){
        // if right exists and is smaller than left and parent, swap 

        if(rightIndex && theHeap[rightIndex] < theHeap[leftIndex] && theHeap[ rightIndex] < theHeap[selfIndex]){
            std::swap(theHeap[rightIndex], theHeap[selfIndex]);
	    selfIndex = rightIndex;
	    // if right doesn't exist or larger than left, left smaller than parent
	}else if(theHeap[leftIndex] < theHeap[selfIndex]){
  	    std::swap(theHeap[leftIndex], theHeap[selfIndex]);
	    selfIndex = leftIndex;
	    // if right doesn't exist or right smaller than parent, so as left, done
	}else{
	    break;
	}
	leftIndex = get_left(selfIndex);
	rightIndex = get_right(selfIndex);
    }
}

void HeapSort::heapify(){
    int parent;
    for(int i = aryLen -1; i>0; i--){
        if(!(i % 2)){
	    i -= 1;
	}
	parent = get_parent(i);
	if(parent != -1){
	    bubble_down(parent);
	}else break;
	i -= 1;
    }
}

void HeapSort::pop(){
    std::swap(theHeap[heapSize-1], theHeap[0]);
    heapSize -= 1;
    bubble_down(0);
}

// since index 0 is the top node, it always has no parent
// thus returning index 0 means no left(right)
int HeapSort::get_left(int index) const{
    int left = 2*index+1;
    return (left >= heapSize)? 0:left;
}

int HeapSort::get_right(int index) const{
    int right = 2*index+2;
    return (right >= heapSize)? 0:right;
}

// returning -1 means no parent
int HeapSort::get_parent(int index) const{
    int parent = -1;
    if(index){
        parent = (index % 2)? (index-1)/2:index/2-1;
    }
    return parent;
}

int HeapSort::get_sibling(int index) const{
    int sibling = -1;
    if(index){
        sibling = (index % 2)? index+1:index-1;
    }
    return sibling;
}
  

//public:
HeapSort::HeapSort(int *array,int len):
    aryOriginal{array},
    aryLen{len},
    heapSize{len}
{
    theHeap = new int[len];
    for(int i = 0; i<len; i++){
        theHeap[i] = array[i];
    }
}

HeapSort::HeapSort(const HeapSort &h):
    aryOriginal{h.aryOriginal},
    aryLen{h.aryLen},
    heapSize{h.heapSize}
{
    this->theHeap = new int[aryLen];
    for(int i = 0; i<aryLen; i++){
        this->theHeap[i] = h.theHeap[i];
    }
}

HeapSort::~HeapSort(){
    delete[] theHeap;
}

void HeapSort::sort(){
    heapify();
    for(int i = 0; i < aryLen; i++){
        aryOriginal[i] = theHeap[0];
	pop();
    }
}

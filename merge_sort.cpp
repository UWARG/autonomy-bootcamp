//Hashem Alnader
#include "cmath"
#include "cstdlib"
#include "iostream"


using namespace std;

bool mergeSort(int list[], int start, int end){
	int size = end - start + 1;
	if(size == 1)
		return true;
	if(mergeSort(list, start, size/2 - 1)){
		for(int i=start; i<size/2 + 1; i++){
			for(int j=0; j<end + 1; j++){
				if(list[i] < list[j]){
					int temp = list[j];
					list[j] = list[i];
					list[i] = temp;
				}
			}
		}
		for(int i=size/2 + 1; i<end + 1; i++){
			for(int j=0; j<end + 1; j++){
				if(list[i] < list[j]){
					int temp = list[j];
					list[j] = list[i];
					list[i] = temp;
				}
			}
		}
		return true;
	}	
}

int main() {
    cout << "Enter list size: ";
    int listSize = 0;
    cin >> listSize;
    
    int array [listSize];
    cout << "List size is: " << listSize << endl;
    
    for(int i=0; i<listSize; i++){
    	cout << "Enter array element: ";
    	cin >> array[i];
    }
	
	cout << "sorting..." << endl;
		
	if(mergeSort(array, 0, listSize - 1)){	
		for(int j=0; j<listSize; j++){
			cout << array[j] << ' ';
		}
	}
	return EXIT_SUCCESS;
}

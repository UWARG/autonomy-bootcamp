/**
 * @file merge_sort.h
 * @author Azhng
 */

#ifndef CONTOUR_MERGESORT_H_INCLUDED
#define CONTOUR_MERGESORT_H_INCLUDED


#include <vector>
#include <cassert>
#include <functional>

/**
 * @brief Pop the first element in the vector
 *
 * The function takes in a vector, then remove the first element in the
 * vector
 *
 * @tparam V
 * @param v Vector
 */
template <typename V>
void pop_front(V & v){
    assert(!v.empty());
    v.erase(v.begin());
}

/**
 * @brief merge two sorted vectors using the provided "comp" function
 *
 * the comp function that compares different values (similar to qsort in C)
 *
 * @tparam T
 * @param a1 first sorted vector
 * @param a2 second sorted vector
 * @param comp std::function object used to compared each element in a1 and a2
 * @return sorted vector that includes all elements from a1 and a2
 */
template <typename T>
std::vector<T> merge(std::vector<T> a1, std::vector<T> a2, std::function<int(T t1, T t2)> comp){
    std::vector<T> merged;
    while(a1.size() != 0 || a2.size() != 0){
        if(a1.empty()){
            merged.insert(merged.end(), a2.begin(), a2.end());
            a2.clear();
        }
        else if(a2.empty()){
            merged.insert(merged.end(), a1.begin(), a1.end());
            a1.clear();
        }
        else if(comp(a1[0], a2[0]) > 0){
            merged.push_back(a2[0]);
            pop_front(a2);
        }
        else if(comp(a1[0], a2[0]) < 0){
            merged.push_back(a1[0]);
            pop_front(a1);
        }
        else if(comp(a1[0], a2[0]) == 0){
            merged.push_back(a1[0]);
            merged.push_back(a2[0]);
            pop_front(a1);
            pop_front(a2);
        }
    }

    return merged;
}

/**
 * @brief merge-sorting given vector using the provided comp function
 *
 * the comp function that compares different values (similar to qsort in C)
 *
 * @tparam T
 * @param ve: the vector of object that need to be sorted
 * @param comp: function used to compare each element in ve
 * @return
 */
template <typename T>
std::vector<T> merge_sort(std::vector<T> ve, std::function<int(T t1, T t2)> comp){
    if(ve.size() == 1){ return ve; }


    std::vector<T> l1(ve.begin(), ve.begin() + ve.size() / 2);
    std::vector<T> l2(ve.begin() + ve.size() / 2, ve.end());


    l1 = merge_sort(l1, comp);
    l2 = merge_sort(l2, comp);

    return merge(l1, l2, comp);
}

#endif //CONTOUR_MERGESORT_H_INCLUDED

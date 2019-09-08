include_directories(include)
add_library(Core src/contour_comparison.cpp ../MergeSort/include/merge_sort.h ../MergeSort/src/merge_sort.cpp ../MergeSort/test/mergesort_test.cpp)
target_link_libraries(Core ${Boost_LIBRARIES} ${OpenCV_LIBS})


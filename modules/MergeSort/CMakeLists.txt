include_directories(include)
add_library(MergeSort src/merge_sort.cpp)
target_link_libraries(MergeSort ${Boost_LIBRARIES})
add_subdirectory(test)
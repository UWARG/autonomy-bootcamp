include_directories(include)
add_library(MergeSort src/mergesort.cpp)
target_link_libraries(MergeSort ${Boost_LIBRARIES})
add_subdirectory(test)

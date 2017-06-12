include_directories(include)
add_library(Mergesort src/merge_sort.cpp)
target_link_libraries(Mergesort ${Boost_LIBRARIES})
add_subdirectory(test)



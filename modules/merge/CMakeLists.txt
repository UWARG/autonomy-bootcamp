include_directories(include)
add_library(Merge_sort src/merge_sort.cpp)
target_link_libraries(Merge_sort ${Boost_LIBRARIES} )
add_subdirectory(test)

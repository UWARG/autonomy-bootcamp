include_directories(include)
add_library(Sorts src/merge_sort.cpp)
target_link_libraries(Sorts ${Boost_LIBRARIES})
add_subdirectory(test)

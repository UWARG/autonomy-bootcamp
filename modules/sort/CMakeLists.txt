include_directories(include)
add_library(Sort src/merge_sort.cpp)
target_link_libraries(Sort ${Boost_LIBRARIES} ${OpenCV_LIBS})
add_subdirectory(test)

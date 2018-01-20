include_directories(include)
add_library(Core src/contour_comparison.cpp merge_sort.cpp)
target_link_libraries(Core ${Boost_LIBRARIES} ${OpenCV_LIBS})
add_subdirectory(Test)

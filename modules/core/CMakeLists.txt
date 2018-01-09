include_directories(include)
add_library(Core src/contour_comparison.cpp)
add_library(Core src/merge_sort.cpp)
target_link_libraries(Core ${Boost_LIBRARIES} ${OpenCV_LIBS})

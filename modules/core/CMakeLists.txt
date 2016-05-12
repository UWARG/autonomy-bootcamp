include_directories(include)
add_library(Core src/contour_comparison.cpp)
target_link_libraries(Core ${Boost_LIBRARIES} ${OpenCV_LIBS})

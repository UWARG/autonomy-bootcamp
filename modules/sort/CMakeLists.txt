add_subdirectory(test)
include_directories(include)
add_library(Sort src/heap_sort.cpp)
target_link_libraries(Sort ${Boost_LIBRARIES} ${OpenCV_LIBS})

enable_testing()

if(Boost_FOUND AND OpenCV_FOUND)
    add_executable(merge_sort_test test.cpp)
    target_link_libraries(merge_sort_test ${Boost_LIBRARIES} ${OpenCV_LIBS} Sort)
	
	# Tests
	add_test("MergeSortTest" merge_sort_test)
endif()

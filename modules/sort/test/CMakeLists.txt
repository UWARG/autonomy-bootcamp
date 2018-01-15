enable_testing()

if(Boost_FOUND AND OpenCV_FOUND)
    add_executable(test test.cpp)
    target_link_libraries(test.out ${Boost_LIBRARIES} ${OpenCV_LIBS} Sort)
	
	# Tests
	add_test("MergeSortTest")
endif()

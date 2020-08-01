include_directories(${Sort_INCLUDE_DIR})
add_executable(MergeSortTest merge_sort_test_suite.cpp)

target_link_libraries(MergeSortTest Sort ${Boost_LIBRARIES})

add_test(
    NAME "Merge Sort Test"
    COMMAND MergeSortTest --log_format=XML --log_sink=TEST-MergeSortTest.xml --log_level=all --report_level=no
)
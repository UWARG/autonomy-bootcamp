add_executable(merge_sort_test merge_sort_test.cpp)
target_link_libraries(merge_sort_test Merge_sort)

# Tests
add_test(
    NAME "merge sort test"
    COMMAND merge_sort_test --log_format=XML --log_sink=TEST-merge_sort_test.xml --log_level=all --report_level=no
)
add_executable(merge_sort_test merge_sort_test.cpp)
target_link_libraries(merge_sort_test Sorts)

# Tests
add_test(
    NAME "Merge_Sort_Test"
    COMMAND merge_sort_test --log-format=XML --log-sink=TEST-merge_sort_test.xml --log-level=all --report-level=no
)

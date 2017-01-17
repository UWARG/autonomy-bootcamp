include_directories(${Core_INCLUDE_DIR})
add_executable(merge_sort_test merge_sort_test.cpp)
target_link_libraries(merge_sort_test Core)

# Tests
add_test(
        NAME "merge_sort_test test"
        COMMAND merge_sort_test --log_format=XML --log_sink=TEST-mytest.xml --log_level=all --report_level=no
)
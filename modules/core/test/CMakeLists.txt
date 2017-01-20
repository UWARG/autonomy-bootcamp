include_directories(${Core_INCLUDE_DIR})
add_executable(merge_sort_test merge_sort_test.cpp)
target_link_libraries(merge_sort_test Core)

enable_testing()

# Tests

add_test("integer_sorting" merge_sort_test --log_format=XML --log_sink=TEST-mytest.xml --log_level=all --report_level=no)
add_test("dummy_test" merge_sort_test --log_format=XML --log_sink=TEST-mytest.xml --log_level=all --report_level=no)

#add_test(
#    NAME "merge_sort_test"
#    COMMAND merge_sort_test --log_format=XML --log_sink=TEST-mytest.xml --log_level=all --report_level=no
#)

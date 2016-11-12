add_executable(test test.cpp)
target_link_libraries(test MergeSort )


add_test(
    NAME "test"
    COMMAND test --log_format=XML --log_sink=TEST-test.xml --log_level=all --report_level=no
)
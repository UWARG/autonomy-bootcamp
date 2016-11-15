add_executable(testsort testsort.cpp)
target_link_libraries(testsort MergeSort )


add_test(
    NAME "testsort"
    COMMAND testsort --log_format=XML --log_sink=TEST-testsort.xml --log_level=all --report_level=no
)
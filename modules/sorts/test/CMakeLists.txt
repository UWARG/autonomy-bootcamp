add_executable(stest test.cpp)
target_link_libraries(stest Sorts)

# Tests
add_test(
    NAME "example test"
    COMMAND stest
)

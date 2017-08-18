set(Sort_INCLUDE_DIR ../include)
include_directories(${Sort_INCLUDE_DIR})
add_executable(mytest general_test.cpp)
target_link_libraries(mytest Sort)

# Tests
add_test(
  NAME "general"
  COMMAND mytest --log_format=XML --log_sink=TEST-mytest.xml --log_level=all --report_level=no
  WORKING_DIRECTORY ${TESTDATA_DIR}
)

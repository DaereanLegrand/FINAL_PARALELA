# Generated file

if (DEFINED ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT $ENV{CUTLASS_TEST_EXECUTION_ENVIRONMENT})
else()
  set(_CUTLASS_TEST_EXECUTION_ENVIRONMENT )
endif()

if (NOT "" STREQUAL "")
  set(TEST_EXE_PATH /$<TARGET_FILE_NAME:06_splitK_gemm>)
else()
  set(TEST_EXE_PATH $<TARGET_FILE_NAME:06_splitK_gemm>)
endif()

add_test("ctest_examples_06_splitk_gemm" ${_CUTLASS_TEST_EXECUTION_ENVIRONMENT} "${TEST_EXE_PATH}" )

if (NOT "./bin" STREQUAL "")
  set_tests_properties("ctest_examples_06_splitk_gemm" PROPERTIES WORKING_DIRECTORY "./bin")
endif()

set_tests_properties(ctest_examples_06_splitk_gemm PROPERTIES DISABLED OFF)

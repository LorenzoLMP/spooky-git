### these do not need to be modified
include_directories(
${PROJECT_SOURCE_DIR}/src/
${PROJECT_SOURCE_DIR}/src/cuda/
${PROJECT_SOURCE_DIR}/src/fields/
${PROJECT_SOURCE_DIR}/src/io/
${PROJECT_SOURCE_DIR}/src/physics/
${PROJECT_SOURCE_DIR}/src/timestepping/
${PROJECT_SOURCE_DIR}/src/parameters/
${PROJECT_SOURCE_DIR}/src/supervisor/
${PROJECT_SOURCE_DIR}/src/rkl/
${PROJECT_SOURCE_DIR}/src/user/
)


### do not modify
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data/)
add_executable(test_mti-eigenmode
initflow.cu)
target_link_libraries(test_mti-eigenmode spooky-lib)


### these can be modified
add_test(
  NAME spooky-mti-eigenmode
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/check-mti-eigenmode.py --executable $<TARGET_FILE:test_mti-eigenmode> --input-dir ${CMAKE_CURRENT_LIST_DIR} --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  )


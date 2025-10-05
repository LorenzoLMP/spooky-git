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
# add_executable(test_advected-mhd-vortex
# initflow.cu)
# target_link_libraries(test_advected-mhd-vortex spooky-lib)


### these can be modified
add_test(
  NAME spooky-advected-mhd-vortex
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/check-advected-mhd-vortex.py --executable $<TARGET_FILE:spooky> --input-dir ${CMAKE_CURRENT_LIST_DIR} --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  )


set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1D_advection_von_Neumann.prm 1D_advection_von_Neumann.prm COPYONLY)
add_test(
  NAME 1D_VON_NEUMANN_ADVECTION
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_advection_von_Neumann.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(1D_burgers_von_Neumann.prm 1D_burgers_von_Neumann.prm COPYONLY)
add_test(
  NAME 1D_VON_NEUMANN_BURGERS
  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_von_Neumann.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

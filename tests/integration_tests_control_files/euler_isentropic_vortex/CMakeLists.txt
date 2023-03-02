set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(2D_euler_isentropic_vortex.prm 2D_euler_isentropic_vortex.prm COPYONLY)
add_test(
 NAME MPI_2D_EULER_ISENTROPIC_VORTEX
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_euler_isentropic_vortex.prm
 WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)                                                                                                                       

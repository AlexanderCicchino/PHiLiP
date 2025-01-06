set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(1D_euler_density_wave.prm 1D_euler_density_wave.prm COPYONLY)
add_test(
 NAME 1D_EULER_DENSITY_WAVE
 COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1D_euler_density_wave.prm
 WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)     

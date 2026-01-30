set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(2D_euler_density_wave.prm 2D_euler_density_wave.prm COPYONLY)
add_test(
 NAME 2D_EULER_DENSITY_WAVE
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_euler_density_wave.prm
 WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)    


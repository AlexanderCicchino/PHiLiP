set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(2D_euler_acoustic_wave.prm 2D_euler_acoustic_wave.prm COPYONLY)
add_test(
 NAME 2D_EULER_ACOUSTIC_WAVE
 COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_euler_acoustic_wave.prm
 WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)    

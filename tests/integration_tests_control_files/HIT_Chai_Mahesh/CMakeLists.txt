set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})                                                                     
configure_file(3D_HIT.prm 3D_HIT.prm COPYONLY)
add_test(
 NAME 3D_HIT_Chai_Mahesh
 COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_HIT.prm
 WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)    


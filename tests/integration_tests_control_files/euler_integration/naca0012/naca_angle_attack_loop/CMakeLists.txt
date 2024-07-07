configure_file(2d_euler_naca0012_subsonic_05_200_angle0.prm 2d_euler_naca0012_subsonic_05_200_angle0.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_SUBSONIC_LONG_Angle0
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012_subsonic_05_200_angle0.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_naca0012_subsonic_05_200_angle1.prm 2d_euler_naca0012_subsonic_05_200_angle1.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_SUBSONIC_LONG_Angle1
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012_subsonic_05_200_angle1.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
configure_file(2d_euler_naca0012_subsonic_05_200_angle2.prm 2d_euler_naca0012_subsonic_05_200_angle2.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_SUBSONIC_LONG_Angle2
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012_subsonic_05_200_angle2.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
configure_file(2d_euler_naca0012_subsonic_05_200_angle3.prm 2d_euler_naca0012_subsonic_05_200_angle3.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_SUBSONIC_LONG_Angle3
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012_subsonic_05_200_angle3.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
configure_file(2d_euler_naca0012_subsonic_05_200_angle4.prm 2d_euler_naca0012_subsonic_05_200_angle4.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_SUBSONIC_LONG_Angle4
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012_subsonic_05_200_angle4.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
configure_file(2d_euler_naca0012_subsonic_05_200_angle5.prm 2d_euler_naca0012_subsonic_05_200_angle5.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_SUBSONIC_LONG_Angle5
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012_subsonic_05_200_angle5.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
configure_file(2d_euler_naca0012_subsonic_05_200_angle6.prm 2d_euler_naca0012_subsonic_05_200_angle6.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_SUBSONIC_LONG_Angle6
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012_subsonic_05_200_angle6.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
configure_file(2d_euler_naca0012_subsonic_05_200_angle7.prm 2d_euler_naca0012_subsonic_05_200_angle7.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_SUBSONIC_LONG_Angle7
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012_subsonic_05_200_angle7.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

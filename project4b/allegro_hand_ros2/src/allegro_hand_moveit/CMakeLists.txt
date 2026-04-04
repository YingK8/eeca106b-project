cmake_minimum_required(VERSION 3.22)
project(allegro_hand_moveit)


find_package(ament_cmake_python REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(std_msgs REQUIRED)


add_executable(save_joint_angles src/save_joint_angles.cpp)
ament_target_dependencies(save_joint_angles 
  rclcpp 
  std_msgs 
  sensor_msgs 
  moveit_ros_planning_interface 
  yaml_cpp_vendor
)

ament_package()

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.launch.py"
  PATTERN "setup_assistant.launch" EXCLUDE)

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

install(TARGETS
  save_joint_angles
  # allegro_node_torque
  # allegro_node_velsat
  DESTINATION lib/${PROJECT_NAME}
)

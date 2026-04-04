cmake_minimum_required(VERSION 3.8)
project(allegro_hand_isaacsim)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# -----------------------
# C++ executable
# -----------------------
add_executable(allegro_hand_sim2real src/sim2real.cpp)

ament_target_dependencies(allegro_hand_sim2real
  rclcpp
  sensor_msgs
)

install(TARGETS
  allegro_hand_sim2real
  DESTINATION lib/${PROJECT_NAME}
)

# -----------------------
# Python executable
# -----------------------
install(PROGRAMS
  src/test_joints.py
  src/test_policy.py
  src/test_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
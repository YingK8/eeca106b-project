cmake_minimum_required(VERSION 3.5)
project(allegro_hand_gui)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(yaml-cpp REQUIRED)

# Include directories
include_directories(
  include
  ${Qt5Widgets_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

# Source files
set(SOURCES
  src/main.cpp
  src/mainwindow.cpp
)

# UI files
set(UIS
  ui/mainwindow.ui
)

# Qt5 setup
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Ensure CMake knows about the UI files directory
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Add executable
add_executable(${PROJECT_NAME}_node ${SOURCES} ${UIS})

# Link libraries
target_link_libraries(${PROJECT_NAME}_node
  Qt5::Widgets yaml-cpp
)
ament_target_dependencies(${PROJECT_NAME}_node rclcpp std_msgs yaml-cpp)
# Install targets
install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install directories
install(DIRECTORY ui
  DESTINATION share/${PROJECT_NAME}
)

# Export package
ament_package()


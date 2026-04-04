from moveit_configs_utils import MoveItConfigsBuilder
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg


def launch_setup(context, *args, **kwargs):
    # Retrieve HAND and TYPE from the context
    hand = LaunchConfiguration("HAND").perform(context)
    hand_type = LaunchConfiguration("TYPE").perform(context)

    # Dynamically construct the robot name
    robot_name = f"allegro_hand_{hand}_{hand_type}"

    # Build the MoveIt config using the dynamically generated robot name
    moveit_config = MoveItConfigsBuilder(
        robot_name=robot_name, package_name="allegro_hand_moveit"
    ).to_moveit_configs()

    # Define the actions (list of launch elements)
    actions = []

    # Declare boolean launch arguments
    actions.append(
        DeclareBooleanLaunchArg(
            "db",
            default_value=False,
            description="By default, we do not start a database (it can be large)",
        )
    )
    actions.append(
        DeclareBooleanLaunchArg(
            "debug",
            default_value=False,
            description="By default, we are not in debug mode",
        )
    )
    actions.append(DeclareBooleanLaunchArg("use_rviz", default_value=True))

    # Include rsp.launch.py
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(moveit_config.package_path / "launch/rsp.launch.py")
            ),
            launch_arguments={
                "HAND": hand,
                "TYPE": hand_type,
            }.items(),
        )
    )

    # Include move_group.launch.py
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(moveit_config.package_path / "launch/move_group.launch.py")
            ),
            launch_arguments={
                "HAND": hand,
                "TYPE": hand_type,
            }.items(),
        )
    )

    # Include moveit_rviz.launch.py (optional)
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(moveit_config.package_path / "launch/moveit_rviz.launch.py")
            ),
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        )
    )

    # Include warehouse_db.launch.py (optional)
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(moveit_config.package_path / "launch/warehouse_db.launch.py")
            ),
            condition=IfCondition(LaunchConfiguration("db")),
        )
    )

    # Fake joint driver
    actions.append(
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[
                moveit_config.robot_description,
                str(moveit_config.package_path / "config/ros2_controllers.yaml"),
            ],
        )
    )

    # Spawner for joint_state_broadcaster
    actions.append(
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
            output="screen",
        )
    )

    # Spawner for HAND_controller
    actions.append(
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["HAND_controller", "-c", "/controller_manager"],
            output="screen",
        )
    )

    # Add save_joint_angles_node
    actions.append(
        Node(
            package="allegro_hand_moveit",
            executable="save_joint_angles",
            name="save_joint_angles",
            output="screen",
        )
    )

    return actions


def generate_launch_description():
    # Declare HAND and TYPE as launch arguments
    hand_arg = DeclareLaunchArgument(
        "HAND", default_value="right", description="The hand side (left or right)"
    )
    type_arg = DeclareLaunchArgument(
        "TYPE", default_value="A", description="The hand type (e.g., A, B, C)"
    )

    # Create and return the LaunchDescription
    return LaunchDescription([
        hand_arg,
        type_arg,
        OpaqueFunction(function=launch_setup),
    ])

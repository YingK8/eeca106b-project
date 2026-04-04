from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def launch_setup(context, *args, **kwargs):
    hand = LaunchConfiguration('HAND').perform(context)
    type = LaunchConfiguration('TYPE').perform(context)

    robot_name = f"allegro_hand_{hand}_{type}"
    


    moveit_config = MoveItConfigsBuilder(robot_name, package_name="allegro_hand_moveit").to_moveit_configs()

    return [generate_move_group_launch(moveit_config)]

def generate_launch_description():
    hand_arg = DeclareLaunchArgument('HAND', default_value='right')
    type_arg = DeclareLaunchArgument('TYPE', default_value='A')

    return LaunchDescription([
        hand_arg,
        type_arg,
        OpaqueFunction(function=launch_setup)      
    ])
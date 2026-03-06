# Filename: .bashrc
# Description: Sources in on the class MASTER version for settings information
# 
# Please (DO NOT) edit this file unless you are sure of what you are doing.
# This file and other dotfiles have been written to work with each other.
# Any change that you are not sure off can break things in an unpredicatable
# ways.

# Set the Class MASTER variable and source the class master version of .cshrc

[[ -z ${MASTER} ]] && export MASTER=${LOGNAME%-*}
[[ -z ${MASTERDIR} ]] && export MASTERDIR=$(eval echo ~${MASTER})

# Set up class wide settings
for file in ${MASTERDIR}/adm/bashrc.d/* ; do [[ -x ${file} ]] && . "${file}"; done

# Set up local settings
for file in ${HOME}/bashrc.d/* ; do [[ -x ${file} ]] && . "${file}"; done

#ee106b shortcuts
alias dr='distrobox enter ros2'
alias srci="source install/setup.bash"
alias tuck="ros2 run ur7e_utils tuck"
alias ec="ros2 run ur7e_utils enable_comms"
alias moveit="ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur7e launch_rviz:=true"
alias fd="ros2 run ur7e_utils freedrive"
alias rs="ros2 run ur7e_utils reset_state"

set_speed() {
    ros2 service call /io_and_status_controller/set_speed_slider ur_msgs/srv/SetSpeedSliderFraction "{speed_slider_fraction: $1}"
}

default_tuck () {
    ros2 control switch_controllers --deactivate forward_velocity_controller --activate scaled_joint_trajectory_control;
    ros2 run ur7e_utils tuck
}

alias fc="ros2 control switch_controllers --activate forward_velocity_controller --deactivate scaled_joint_trajectory_control"

echo "SUCCESS.. BOINK!"


#
<img src="https://github.com/user-attachments/assets/b198accd-5de4-4cfc-a347-05899c0391b3" width="400" align="left">
<img src="https://github.com/user-attachments/assets/48640d7e-a8db-4758-9770-bcc1b3e68434" width="400" align="center">

# Allegro Hand V5 ROS2
> [!IMPORTANT]</br>
> **Date : 25/07/10**</br>
> **We have recently resolved an issue where some fingertip sensor values were delayed. To address this, we have uploaded the updated firmware at the link below. Please update the firmware and download the modified ROS2 package.**</br>
> [allegro_hand_v5_firmware](https://github.com/Wonikrobotics-git/allegro_hand_v5_firmware)</br>
> **Please update the firmware by referring to the "Firmware Update" section in the user manual.**



This is the official release to control Allegro Hand V5 with ROS2(**Only V5 supported, V4 is not supported**). Mostly, it is based on the release of Allegro Hand V5 ROS1 package.
You can find our latest Allegro Hand V5 ROS1 package :(https://github.com/Wonikrobotics-git/allegro_hand_ros_v5)
And the interfaces and controllers have been improved and rewritten by Soohoon Yang(Hibo) from Wonik Robotics.

We have integrated the core elements of two ROS packages(allegro_hand_description, allegro_hand_parameters) into a main package(allegro_hand_controllers).

**For now, we support three additional action.**
- Visualize Allegro Hand V5 changing in real time using Rviz2.
- Save customize pose using MOVEIT2 and move to the saved pose.
- Simply control hand with GUI tool instead of using keyboard.

These packages are tested on ROS2 Humble(Ubuntu 22.04). It will likely not work with newer or older versions(Rolling, Foxy ...).

## Useful Links
- Official Allegro Hand Website : https://www.allegrohand.com/
- Community Forum :  https://www.allegrohand.com/forum

## Packages

**From Allegro Hand V5, the hand is fully based on torque controller.** 

- allegro_hand_controllers : Contain two main nodes for control the hand and urdf descriptions,meshes.
	- node : Receive encoder data and compute torque using `computeDesiredTorque`.
	- grasp : Apply various pre-defined grasps or customized poses.
- allegro_hand_driver : Main driver for sending and receiving data with the Allegro Hand.
- allegro_hand_keyboard : Node that sends the command to control Allegro Hand. All commands need to be pre-defined.
- allegro_hand_moveit : Provide MOVEIT2 package for Allegro Hand V5.
- allegro_hand_gui : Node that control the allegro hand with gui program.
- bhand : Library files for the predefined grasps and actions., available on 64 bit versions.
  - ⚠ **Default: x86-64-bit.** If using a **ARM64**, update the symlink accordingly from [here](https://github.com/Wonikrobotics-git/Bhandlib_ARM).
  
  ## Topic description

- Control
	- /allegroHand_(NUM)/lib_cmd :  Hand command.
 	- /allegroHand_(NUM)/joint_cmd : Desired hand joint positions and control REAL Allegro Hand.
  	- /allegroHand_(NUM)/force_chg : Change grasp force of grasping algorithm(grasp_3, grasp_4).
  	- /allegroHand_(NUM)/time_chg : Change time of moving target positions of each joints.
  	- /allegroHand_(NUM)/envelop_torque : Change torque of envelop command.
- Joint States
  	- /allegroHand_(NUM)/joint_states : REAL Allegro Hand current joint positions.
  	- /allegroHand_sim/joint_states : ISAAC SIM Allegro Hand current joint positions.
- Sensors
  	- /allegroHand_(NUM)/tactile_sensors : REAL tactile sensors data {Finger1(Index),Finger2(Middle),Finger3(Pinky),Finger4(Thumb)}.
  	- /allegroHand_sim/contact_sensors : ISAAC SIM contact sensors data {Finger1(Index),Finger2(Middle),Finger3(Pinky),Finger4(Thumb)}.

## Install the PCAN driver

**With ROS2, you don't need to install PCAN driver anymore!**

If you have already installed PCAN driver, please follow instructions below.
- [Optional] Disable previously installed PEAK-CAN driver.
~~~bash
# Temporarily
sudo rmmod pcan
sudo modprobe peak_usb

# Permanent
sudo rm /etc/modprobe.d/pcan.conf
sudo rm /etc/modprobe.d/blacklist-peak.conf
~~~

## Run main controller nodes

1. Make new workspace.
~~~bash
mkdir allegro_ws
~~~

2. Install necessart package.
~~~bash
sudo apt-get update
sudo apt-get install ros-<distro>-xacro
sudo apt install ros-<distro>-moveit
sudo apt install ros-<distro>-controller-manager
sudo apt install ros-<distro>-joint-state-broadcaster
sudo apt install ros-<distro>-joint-trajectory-controller
~~~

3. Download ROS2 package for Allegro Hand V5 using below command.
~~~bash
cd ~/allegro_ws
git clone https://github.com/Wonikrobotics-git/allegro_hand_ros2_v5.git
~~~

4. Build.
~~~bash
cd ~/allegro_ws
source /opt/ros/<distro>/setup.bash
colcon build
~~~

5. Launch allegro main node.
~~~bash
source install/setup.bash
ros2 launch allegro_hand_controllers allegro_hand.launch.py HAND:=right TYPE:=B
~~~
**You need to write your password to open CAN port**


**Please check 'Launch file instructions below'.**

6. Run allegro hand keyboard node.
~~~bash
cd ~/allegro_ws
source install/setup.bash
ros2 run allegro_hand_keyboards allegro_hand_keyboard
~~~

7. Control Hand using Keyboard command.

## Launch file instructions

Same as the ROS1 package, you can simply control Allegro Hand V5 by launching *allegro_hand.launch.py* . At a minimum, you must specify the handedness and the hand type:
~~~bash
ros2 launch allegro_hand_controllers allegro_hand.launch.py HAND:=right|left TYPE:=A|B
~~~

Optional arguments:
~~~
VISUALIZE:=true|false (default is false)
MOVEIT:=true|false (default is false)
GUI:=true|false (default is false)
~~~

- If you want to visualize Allegro Hand on Rviz2:
~~~bash
ros2 launch allegro_hand_controllers allegro_hand.launch.py HAND:=right TYPE:=B VISUALIZE:=true
~~~

- If you want to use Allegro Hand with MOVEIT2:
~~~bash
ros2 launch allegro_hand_controllers allegro_hand.launch.py HAND:=right TYPE:=B MOVEIT:=true
~~~

- If you want to control Allegro Hand with GUI:
~~~bash
ros2 launch allegro_hand_controllers allegro_hand.launch.py HAND:=right TYPE:=B GUI:=true
~~~

## Control more than one hand

To control more than one hand, you must specify CAN port number and Allegro Hand number.

Terminal 1:
~~~bash
ros2 launch allegro_hand_controllers allegro_hand.launch.py HAND:=right TYPE:=A CAN_DEVICE:=can0 NUM:=0
~~~

Termianl 2:
~~~bash
ros2 launch allegro_hand_controllers allegro_hand.launch.py HAND:=left TYPE:=A CAN_DEVICE:=can1 NUM:=1
~~~

To control first hand,

Terminal 3:
~~~bash
ros2 run allegro_hand_keyboards allegro_hand_keyboard --ros-args allegroHand_0/lib_cmd:=allegroHand_0/lib_cmd
~~~

To control second hand,

Terminal 4:
~~~bash
ros2 run allegro_hand_keyboards allegro_hand_keyboard --ros-args allegroHand_0/lib_cmd:=allegroHand_1/lib_cmd
~~~

**These are example commands.You may need to change CAN_DEVICE, PORT and NUM arguments accroding to your system.**

## ISAACSIM

We now introuce two features —REAL2SIM and SIM2REAL—that integrate the Allegro Hand V5 with Isaac Sim.

Please refer to the following guide for detailed information : [allegro_hand_isaacsim](https://github.com/Wonikrobotics-git/allegro_hand_ros2_v5/tree/master-4finger/src/allegro_hand_isaacsim)

## MOVEIT2 

These newly added feature function identically to their ROS1 counterparts. Please refer to the ROS1 manual for guidance.
Our latest Allegro Hand V5 ROS1 package : [ROS1](https://github.com/Wonikrobotics-git/allegro_hand_ros_v5)

**Make sure to install MOVEIT2 first in your PC and should install several additional packages to operate it**
You can find installation guide in here : [MOVEIT2](https://moveit.picknik.ai/main/doc/tutorials/getting_started/getting_started.html)

## GUI

These newly added feature function identically to their ROS1 counterparts. Please refer to the ROS1 manual for guidance.
Our latest Allegro Hand V5 ROS1 package : [ROS1](https://github.com/Wonikrobotics-git/allegro_hand_ros_v5)


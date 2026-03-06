"""
This node locates Aruco AR markers in images and publishes their ids and poses.

Subscriptions:
   /camera/image_raw (sensor_msgs.msg.Image)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)

Published Topics:
    /aruco_poses (geometry_msgs.msg.PoseArray)
       Pose of all detected markers (suitable for rviz visualization)

    /aruco_markers (ros2_aruco_interfaces.msg.ArucoMarkers)
       Provides an array of all poses along with the corresponding
       marker ids.

Published Transforms:
    tf2 transforms from camera frame to each detected marker
    Child frame names: "ar_marker_{marker_id}"

Parameters:
    marker_size - size of the markers in meters (default .0625)
    aruco_dictionary_id - dictionary that was used to generate markers
                          (default DICT_5X5_250)
    image_topic - image topic to subscribe to (default /camera/image_raw)
    camera_info_topic - camera info topic to subscribe to
                         (default /camera/camera_info)

Author: Nathan Sprague
Version: 10/26/2020

"""

import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
# import tf_transformations
# from autolab_core import transformations
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from tf2_ros import TransformBroadcaster

def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = np.empty((4,), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


class ArucoNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("aruco_node")

        # Declare and read parameters
        self.declare_parameter(
            name="marker_size",
            value=0.0625,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Size of the markers in meters.",
            ),
        )

        self.declare_parameter(
            name="aruco_dictionary_id",
            value="DICT_5X5_250",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Dictionary that was used to generate markers.",
            ),
        )

        self.declare_parameter(
            name="image_topic",
            value="/camera/image_raw",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Image topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_info_topic",
            value="/camera/camera_info",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera info topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_frame",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera optical frame to use.",
            ),
        )


        self.marker_size = (
            self.get_parameter("marker_size").get_parameter_value().double_value
        )
        self.get_logger().info(f"Marker size: {self.marker_size}")
        
        self.marker_size_map = {1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15, 5: 0.15, 11: 0.15, 
                                6: 0.15, 7: 0.15, 8: 0.15, 9: 0.15, 10: 0.15}
        self.get_logger().info(f"Marker size map for marker ids is: {self.marker_size_map}")

        dictionary_id_name = (
            self.get_parameter("aruco_dictionary_id").get_parameter_value().string_value
        )
        self.get_logger().info(f"Marker type: {dictionary_id_name}")

        image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Image topic: {image_topic}")

        info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.get_logger().info(f"Image info topic: {info_topic}")

        self.camera_frame = (
            self.get_parameter("camera_frame").get_parameter_value().string_value
        )
        self.get_logger().info(f"Camera frame: {self.camera_frame}")

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            if type(dictionary_id) != type(cv2.aruco.DICT_5X5_100):
                raise AttributeError
        except AttributeError:
            self.get_logger().error(
                "bad aruco_dictionary_id: {}".format(dictionary_id_name)
            )
            options = "\n".join([s for s in dir(cv2.aruco) if s.startswith("DICT")])
            self.get_logger().error("valid options: {}".format(options))

        # Set up subscriptions
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self.info_callback, qos_profile_sensor_data
        )

        self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data
        )

        # Set up publishers
        self.poses_pub = self.create_publisher(PoseArray, "aruco_poses", 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, "aruco_markers", 10)

        # Set up tf2 broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.aruco_dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()

        self.bridge = CvBridge()

    def info_callback(self, info_msg):
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)
        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.info_sub)

    def image_callback(self, img_msg):
        if self.info_msg is None:
            self.get_logger().warn("No camera info has been received!")
            return

        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        markers = ArucoMarkers()
        pose_array = PoseArray()
        if self.camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame

        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        corners, marker_ids, rejected = cv2.aruco.detectMarkers(
            cv_image, self.aruco_dictionary, parameters=self.aruco_parameters
        )

        if marker_ids is not None:
            # process each marker individually to allow for diff marker sizes
            rvecs = []
            tvecs = []
            turtlebot_corners = []
            turtlebot_markers = []
            goal_corners = []
            goal_markers = []
            final_marker_ids = []
            for i, marker_id in enumerate(marker_ids):
                marker_size = self.marker_size_map[marker_id[0]]
                if marker_size == 0.05:
                    turtlebot_corners.append(corners[i])
                    turtlebot_markers.append(marker_id)
                elif marker_size == 0.15:
                    goal_corners.append(corners[i])
                    goal_markers.append(marker_id)

            if len(goal_markers) > 0:
                goal_rvecs, goal_tvecs = [], []
                if cv2.__version__ > "4.0.0":
                    goal_rvecs, goal_tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        goal_corners, 0.15, self.intrinsic_mat, self.distortion
                    )
                else:
                    goal_rvecs, goal_tvecs = cv2.aruco.estimatePoseSingleMarkers(
                        goal_corners, goal_markers, self.intrinsic_mat, self.distortion
                    )
                self.get_logger().info(f"info is {goal_rvecs}, {goal_tvecs}")
                self.get_logger().info(f"info is {goal_markers}")
                rvecs.extend(goal_rvecs)
                tvecs.extend(goal_tvecs)
                final_marker_ids.extend(goal_markers)

            if len(turtlebot_markers) > 0:
                turtlebot_rvecs, turtlebot_tvecs = [], []
                if cv2.__version__ > "4.0.0":
                    turtlebot_rvecs, turtlebot_tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        turtlebot_corners, 0.05, self.intrinsic_mat, self.distortion
                    )
                else:
                    turtlebot_rvecs, turtlebot_tvecs = cv2.aruco.estimatePoseSingleMarkers(
                        turtlebot_corners, turtlebot_markers, self.intrinsic_mat, self.distortion
                    )
                rvecs.extend(turtlebot_rvecs)
                tvecs.extend(turtlebot_tvecs)
                final_marker_ids.extend(turtlebot_markers)

            for i, marker_id in enumerate(final_marker_ids):
                pose = Pose()
                pose.position.x = tvecs[i][0][0]
                pose.position.y = tvecs[i][0][1]
                pose.position.z = tvecs[i][0][2]

                rot_matrix = np.eye(4)
                rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]

                # quat = tf_transformations.quaternion_from_matrix(rot_matrix)
                # quat = transformations.quaternion_from_matrix(rot_matrix)
                quat = quaternion_from_matrix(rot_matrix)

                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]

                # Publish tf transform for this marker
                transform = TransformStamped()
                transform.header.stamp = img_msg.header.stamp
                if self.camera_frame == "":
                    # Note the info_msg header is better practice just testing this
                    # transform.header.frame_id = self.info_msg.header.frame_id
                    transform.header.frame_id = "camera1"
                else:
                    transform.header.frame_id = self.camera_frame
                transform.child_frame_id = f"ar_marker_{marker_id[0]}"
                
                transform.transform.translation.x = pose.position.x
                transform.transform.translation.y = pose.position.y
                transform.transform.translation.z = pose.position.z
                transform.transform.rotation.x = pose.orientation.x
                transform.transform.rotation.y = pose.orientation.y
                transform.transform.rotation.z = pose.orientation.z
                transform.transform.rotation.w = pose.orientation.w
                
                self.tf_broadcaster.sendTransform(transform)
                pose_array.poses.append(pose)
                markers.poses.append(pose)
                markers.marker_ids.append(marker_id[0])

            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/action/execute_trajectory.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <action_msgs/msg/goal_status_array.hpp>


class SaveJointStateNode : public rclcpp::Node
{
public:
    SaveJointStateNode() : Node("save_joint_state")
    {
        joint_cmd_pub_ = this->create_publisher<std_msgs::msg::String>("/allegroHand_0/lib_cmd", 10);
        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, std::bind(&SaveJointStateNode::saveJointStates, this, std::placeholders::_1));

        // Execute Trajectory 액션 상태 구독
        execute_trajectory_status_sub_ = this->create_subscription<action_msgs::msg::GoalStatusArray>(
            "/execute_trajectory/_action/status", 10, 
            std::bind(&SaveJointStateNode::executeTrajectoryStatusCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Node initialized, waiting for execution requests...");
    }

private:
    void saveJointStates(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        if (!should_save_joint_states_) {
            return;
        }

        // Create a vector of pairs (joint_name, position)
        std::vector<std::pair<std::string, double>> joint_positions;
        for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i) {
            joint_positions.push_back(std::make_pair(msg->name[i], msg->position[i]));
        }

        // Sort the vector based on joint names
        std::sort(joint_positions.begin(), joint_positions.end(),
                [](const auto& a, const auto& b) {
                    // Extract the numeric part of the joint name for comparison
                    int a_num = std::stoi(a.first.substr(6, a.first.find_last_of('_') - 6));
                    int b_num = std::stoi(b.first.substr(6, b.first.find_last_of('_') - 6));
                    return a_num < b_num;
                });

        YAML::Emitter out;
        out << YAML::BeginMap;
        
        out << YAML::Key << "name" << YAML::Value << YAML::BeginSeq;
        for (const auto& jp : joint_positions) {
            out << jp.first;
        }
        out << YAML::EndSeq;

        out << YAML::Key << "position" << YAML::Value << YAML::BeginSeq;
        for (const auto& jp : joint_positions) {
            out << jp.second;
        }
        out << YAML::EndSeq;

        out << YAML::EndMap;

        std::string pkg_path = ament_index_cpp::get_package_share_directory("allegro_hand_controllers");
        std::string file_path = pkg_path + "/pose/pose_moveit.yaml";
        std::ofstream fout(file_path);
        fout << out.c_str();
        fout.close();

        std_msgs::msg::String msg_out;
        msg_out.data = "moveit";
        joint_cmd_pub_->publish(msg_out);

        should_save_joint_states_ = false;
        RCLCPP_INFO(this->get_logger(), "Joint states saved successfully.");
    }

    void executeTrajectoryStatusCallback(const action_msgs::msg::GoalStatusArray::SharedPtr msg)
    {
        for (const auto& status : msg->status_list) {
            if (status.status == action_msgs::msg::GoalStatus::STATUS_SUCCEEDED) {
                RCLCPP_INFO(this->get_logger(), "Execution succeeded, preparing to save joint states.");
                should_save_joint_states_ = true;
                return;
            }
        }
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr joint_cmd_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<action_msgs::msg::GoalStatusArray>::SharedPtr execute_trajectory_status_sub_;
    bool should_save_joint_states_ = false;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SaveJointStateNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
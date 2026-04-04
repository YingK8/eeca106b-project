#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sstream>

using namespace std;
using namespace rclcpp;

class ChangeJointStateNode : public rclcpp::Node
{
public:
    ChangeJointStateNode() : Node("allegro_hand_sim2real")
    {
        joint_cmd_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/allegroHand/joint_cmd", 1); 
        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/allegroHand_sim/joint_states", 1, bind(&ChangeJointStateNode::changeJointStates, this, placeholders::_1));

    }

private:
    void changeJointStates(const sensor_msgs::msg::JointState::SharedPtr msg)
    {

        // Create a vector of pairs (joint_name, position)
        vector<pair<string, double>> joint_positions;
        for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i) {
            joint_positions.push_back(make_pair(msg->name[i], msg->position[i]));
        }

        // Sort the vector based on joint names
        sort(joint_positions.begin(), joint_positions.end(),
                [](const auto& a, const auto& b) {
                    // Extract the numeric part of the joint name for comparison
                    int a_num = stoi(a.first.substr(6, a.first.find_last_of('_') - 6));
                    int b_num = stoi(b.first.substr(6, b.first.find_last_of('_') - 6));
                    return a_num < b_num;
                });

        auto joint_msg = sensor_msgs::msg::JointState();
        joint_msg.header.stamp = this->now();   
        joint_msg.name.resize(joint_positions.size());
        joint_msg.position.resize(joint_positions.size());
    
        for (size_t i = 0; i < joint_positions.size(); ++i) {
            joint_msg.name[i]     = joint_positions[i].first;
            joint_msg.position[i] = joint_positions[i].second;
        }
    
        joint_cmd_pub_->publish(joint_msg);

      //  RCLCPP_INFO(this->get_logger(), "Joint states changed successfully.");
    }

    Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_pub_;
    Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;

};

int main(int argc, char **argv)
{
    init(argc, argv);
    auto node = std::make_shared<ChangeJointStateNode>();
    spin(node);
    shutdown();
    return 0;
}
#include "mainwindow.h"

#include <QApplication>
#include <rclcpp/rclcpp.hpp>
int main(int argc, char *argv[])
{
    // ROS2 초기화
    rclcpp::init(argc, argv);

    QApplication a(argc, argv);

    // MainWindow에 ROS2 노드 전달
    auto node = std::make_shared<rclcpp::Node>("allegro_hand_gui_node");
    MainWindow w(nullptr);
    w.setNode(node); // 노드를 MainWindow에 설정
    w.show();

    // Qt 애플리케이션 실행
    int result = a.exec();

    // ROS2 종료
    rclcpp::shutdown();
    return result;
}

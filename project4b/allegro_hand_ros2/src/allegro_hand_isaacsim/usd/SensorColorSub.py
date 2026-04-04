import carb
import omni
import omni.usd

import rclpy
from rclpy.node import Node
from rclpy.utilities import get_default_context
from std_msgs.msg import Int32MultiArray

from omni.kit.scripting import BehaviorScript
from pxr import UsdShade, Gf, Sdf


ctx = get_default_context()  
if not ctx.ok():
    rclpy.init()
    ctx.on_shutdown(rclpy.shutdown)

class SensorColorBehavior(BehaviorScript):
    def on_init(self):
        """스크립트가 Prim에 붙을 때 단 한 번 호출"""
        self._node = None
        self._sub = None
        self._latest_data = None

    def on_play(self):
        self._node = rclpy.create_node('sensor_color_node')  
        self._sub = self._node.create_subscription(
            Int32MultiArray,
            'allegroHand_0/tactile_sensors',
            self._ros_callback,
            10
        )
        carb.log_info("SensorColorBehavior: ROS2 subscriber created")

    def _ros_callback(self, msg: Int32MultiArray):
        """토픽 콜백: 최신 데이터 버퍼링"""
        self._latest_data = msg.data

    def on_update(self, current_time: float, delta_time: float):
        """매 프레임 호출: ROS 콜백 스핀, 색상 계산, USD에 쓰기"""
        if self._node:
            rclpy.spin_once(self._node, timeout_sec=0.0)

        data = self._latest_data
        if not data or len(data) < 4:
            return

        rgb_vals = []
        for i in range(4):
            raw = float(data[i])
            value = raw * (255.0 / 500.0)
            if value >= 500.0:
                value = 500.0

            # 구간별 r,g,b 계산
            if value <= 64.0:
                r, g, b = 0.0, value / 64.0, 1.0
            elif value <= 128.0:
                r, g, b = 0.0, 1.0, 1.0 - ((value - 64.0) / 64.0)
            elif value <= 192.0:
                r, g, b = (value - 128.0) / 64.0, 1.0, 0.0
            elif value <= 255.0:
                r, g, b = 1.0, 1.0 - ((value - 192.0) / 64.0), 0.0
            else:
                r, g, b = 1.0, 0.0, 0.0

            rgb_vals.append((r, g, b))


        stage = omni.usd.get_context().get_stage()
        for idx, (r, g, b) in enumerate(rgb_vals, start=1):
            prim_path = f"/World/allegro_hand_right/Looks/Fingertip{idx}/Shader"
            shader = UsdShade.Shader.Get(stage, Sdf.Path(prim_path))  
            if not shader:
                carb.log_warn(f"Shader not found: {prim_path}")
                continue
            
            inp = shader.GetInput("diffuse_tint")
            if not inp:
                inp = shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f)  
            inp.Set(Gf.Vec3f(r, g, b))  




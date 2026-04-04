from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import kornia
import numpy as np
import rclpy
import torch
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from perseus_detector.detector.models import KeypointCNN
from rclpy.node import Node


UNIT_CUBE_KEYPOINTS = np.array(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=np.float32,
)

KUBRIC_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
}

CAMERA_INTRINSICS_BY_SERIAL = {
    "481F2EAF": "c299_1080p_intrinsics_camera13.yaml",
    "4040DCAF": "c299_1080p_intrinsics_camera14.yaml",
}


# NOTE: these are hardcoded transformations specific to the CAD mount built for 106B Project 4b.
# They take into account the offset between the AR tag and the wrist frame on the hand.
T_H_ar = np.eye(4) # tilted hand frame to ar marker frame
T_H_ar[:3, 3] = np.array([0.097869, -0.0451, -0.0295])
T_W_H = np.eye(4)
theta = np.deg2rad(20)  # 20 deg tilt from horizontal
T_W_H[:3, :3] = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta),  np.cos(theta)]
])

def _rotmat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    # returns (x, y, z, w).
    return Rotation.from_matrix(rot).as_quat()


class LogitechCamera:
    def __init__(
        self,
        camera_index: int,
        device_path: str,
        width: int,
        height: int,
        fps: int,
        fourcc: str,
        autofocus: bool,
        use_v4l2_ctl: bool,
        logger,
    ) -> None:
        self._logger = logger
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp = 0.0
        self._stop_event = threading.Event()

        if use_v4l2_ctl: # should not be used
            self._logger.warn("Using V4L2, make sure this is intended!")
            self._configure_device(device_path, width, height, fps, fourcc)

        capture_target = device_path if device_path else camera_index
        self._cap = cv2.VideoCapture(capture_target, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera target {capture_target}")

        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not autofocus:
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        fourcc_value = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4))
        self._logger.info(
            f"Camera ready: {actual_w}x{actual_h} @ {actual_fps:.2f} FPS, FOURCC={actual_fourcc}"
        )

        self.width = actual_w
        self.height = actual_h

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _configure_device(self, device_path: str, width: int, height: int, fps: int, fourcc: str) -> None:
        if not shutil.which("v4l2-ctl"):
            self._logger.warn("v4l2-ctl not found, skipping pre-configuration.")
            return
        cmd = [
            "v4l2-ctl",
            "-d",
            device_path,
            f"--set-fmt-video=width={width},height={height},pixelformat={fourcc}",
            f"--set-parm={fps}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            self._logger.warn("v4l2-ctl failed, continuing with OpenCV settings.")
            if result.stderr.strip():
                self._logger.warn(result.stderr.strip())

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.002)
                continue
            ts = time.time()
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_timestamp = ts

    def read_latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self._frame_lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_timestamp

    def close(self) -> None:
        self._stop_event.set()
        if self._reader.is_alive():
            self._reader.join(timeout=1.0)
        self._cap.release()


class CubeEstimatorNode(Node):
    def __init__(self) -> None:
        super().__init__("cube_estimator")
        torch.set_float32_matmul_precision("high")

        self.declare_parameter("camera_index", 0)
        self.declare_parameter("camera_device", "/dev/video0")
        self.declare_parameter("camera_width", 1920)
        self.declare_parameter("camera_height", 1080)
        self.declare_parameter("camera_fps", 30)
        self.declare_parameter("camera_fourcc", "MJPG")
        self.declare_parameter("camera_autofocus", False)
        # By default, do not use v4l2-ctl pre-configuration.
        self.declare_parameter("use_v4l2_ctl", False)

        self.declare_parameter("intrinsics_yaml", "auto")
        self.declare_parameter("weights_path", self._default_weights_path())
        self.declare_parameter("cube_size_m", 0.07)
        self.declare_parameter("inference_input_size", 256)
        self.declare_parameter("inference_rate_hz", 30.0)
        self.declare_parameter("pose_topic", "/cube_pose")
        self.declare_parameter("world_frame_id", "aruco_world")
        self.declare_parameter("apply_kubric_to_z_up", True)
        self.declare_parameter("enable_smoother", False)

        self.declare_parameter("use_marker_init", True)
        self.declare_parameter("marker_dict", "DICT_5X5_50")
        self.declare_parameter("marker_id", 0)
        self.declare_parameter("marker_size_m", 0.05)
        self.declare_parameter("marker_timeout_sec", 15.0)
        self.declare_parameter("cache_extrinsics", True)
        self.declare_parameter("use_cached_extrinsics", False)
        self.declare_parameter("extrinsics_cache_path", self._default_extrinsics_cache_path())
        self.declare_parameter("allow_cache_fallback", True)

        camera_index = int(self.get_parameter("camera_index").value)
        camera_device = str(self.get_parameter("camera_device").value)
        camera_width = int(self.get_parameter("camera_width").value)
        camera_height = int(self.get_parameter("camera_height").value)
        camera_fps = int(self.get_parameter("camera_fps").value)
        camera_fourcc = str(self.get_parameter("camera_fourcc").value)
        camera_autofocus = bool(self.get_parameter("camera_autofocus").value)
        use_v4l2_ctl = bool(self.get_parameter("use_v4l2_ctl").value)

        self.inference_size = int(self.get_parameter("inference_input_size").value)
        self.apply_kubric_to_z_up = bool(self.get_parameter("apply_kubric_to_z_up").value)
        self.world_frame_id = str(self.get_parameter("world_frame_id").value)

        if bool(self.get_parameter("enable_smoother").value):
            raise RuntimeError("Smoother is currently not supported.")

        intrinsics_path = self._resolve_intrinsics_path(
            camera_device=camera_device,
            intrinsics_yaml=str(self.get_parameter("intrinsics_yaml").value),
        )
        weights_path = Path(str(self.get_parameter("weights_path").value)).expanduser()

        self.camera_matrix, self.dist_coeffs = self._load_intrinsics(intrinsics_path)
        self.get_logger().info(f"Loaded intrinsics from {intrinsics_path}")

        self.camera = LogitechCamera(
            camera_index=camera_index,
            device_path=camera_device,
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            fourcc=camera_fourcc,
            autofocus=camera_autofocus,
            use_v4l2_ctl=use_v4l2_ctl,
            logger=self.get_logger(),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")
        self.model = self._load_model(weights_path)

        cube_size_m = float(self.get_parameter("cube_size_m").value)
        self.object_keypoints = UNIT_CUBE_KEYPOINTS * (cube_size_m / 2.0)

        self.R_world_camera = np.eye(3, dtype=np.float64)
        self.t_world_camera = np.zeros(3, dtype=np.float64)
        self._initialize_world_from_marker()

        pose_topic = str(self.get_parameter("pose_topic").value)
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)

        self._last_camera_ts = 0.0
        self._last_fps_window = time.monotonic()
        self._fps_count = 0

        rate_hz = float(self.get_parameter("inference_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate_hz, self._on_timer)
        self.get_logger().info(f"Publishing cube pose to {pose_topic}")

    def _intrinsics_search_paths(self) -> list[Path]:
        here = Path(__file__).resolve()
        candidates = [here.parents[1] / "assets" / "calibration" / "c299_1080p_intrinsics.yaml"]

        # If running from a workspace, prefer the source tree assets path.
        for parent in here.parents:
            if parent.name == "cube_rotation_ws":
                candidates.append(
                    parent / "src" / "cro_estimation" / "assets" / "calibration" / "c299_1080p_intrinsics.yaml"
                )
                break

        return candidates

    def _default_intrinsics_path(self) -> str:
        candidates = self._intrinsics_search_paths()
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])

    def _calibration_dir(self) -> Path:
        for candidate in self._intrinsics_search_paths():
            calibration_dir = candidate.parent
            if calibration_dir.exists():
                return calibration_dir
        return self._intrinsics_search_paths()[0].parent

    def _get_camera_serial(self, device_path: str) -> str:
        if not device_path:
            raise RuntimeError("Camera auto-detection requires a non-empty camera_device path.")

        if not shutil.which("udevadm"):
            raise RuntimeError("udevadm not found; cannot auto-detect camera intrinsics from device serial.")

        cmd = ["udevadm", "info", "--query=all", f"--name={device_path}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"Failed to query udev for camera device {device_path}: {stderr or 'unknown error'}"
            )

        serial_short = None
        serial_full = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("E: ID_SERIAL_SHORT="):
                serial_short = line.split("=", 1)[1].strip()
            elif line.startswith("E: ID_SERIAL="):
                serial_full = line.split("=", 1)[1].strip()

        if serial_short:
            return serial_short
        if serial_full:
            return serial_full

        raise RuntimeError(f"Could not find ID_SERIAL_SHORT or ID_SERIAL for camera device {device_path}.")

    def _resolve_intrinsics_path(self, camera_device: str, intrinsics_yaml: str) -> Path:
        intrinsics_yaml = intrinsics_yaml.strip()
        if intrinsics_yaml and intrinsics_yaml.lower() != "auto":
            return Path(intrinsics_yaml).expanduser()

        serial = self._get_camera_serial(camera_device)
        intrinsics_filename = CAMERA_INTRINSICS_BY_SERIAL.get(serial)
        if intrinsics_filename is None:
            supported_serials = ", ".join(sorted(CAMERA_INTRINSICS_BY_SERIAL.keys()))
            raise RuntimeError(
                f"Unsupported camera serial '{serial}' for device {camera_device}. "
                f"Supported serials: {supported_serials}"
            )

        intrinsics_path = self._calibration_dir() / intrinsics_filename
        if not intrinsics_path.exists():
            raise RuntimeError(
                f"Resolved intrinsics file for serial '{serial}' does not exist: {intrinsics_path}"
            )
        return intrinsics_path

    def _default_extrinsics_cache_path(self) -> str:
        here = Path(__file__).resolve()
        candidates = []

        # Prefer workspace assets path if available (avoid install/site-packages paths).
        for base in [Path.cwd(), here]:
            for parent in base.parents:
                if parent.name == "cube_rotation_ws":
                    candidates.append(
                        parent / "src" / "cro_estimation" / "assets" / "calibration" / "cam_to_world.yaml"
                    )
                    break

        source_default = here.parents[1] / "assets" / "calibration" / "cam_to_world.yaml"
        if "site-packages" not in {p.name for p in source_default.parents}:
            candidates.append(source_default)

        # Fallback to a user-writable location if no workspace path is found.
        candidates.append(Path("~/.ros/cro_estimation/calibration/cam_to_world.yaml").expanduser())

        for candidate in candidates:
            if candidate.parent.exists():
                return str(candidate)
        return str(candidates[0])

    def _default_weights_path(self) -> str:
        here = Path(__file__).resolve()
        candidates = [here.parents[1] / "assets" / "ckpts" / "perseus.pth"]

        # If running from an installed package, derive share path from __file__.
        for parent in here.parents:
            if parent.name == "site-packages":
                install_prefix = parent.parents[3]  # .../install/cro_estimation
                candidates.append(
                    install_prefix / "share" / "cro_estimation" / "assets" / "ckpts" / "perseus.pth"
                )
                break

        # If running from a workspace, use the source tree path.
        for parent in here.parents:
            if parent.name == "cube_rotation_ws":
                candidates.append(parent / "src" / "cro_estimation" / "assets" / "ckpts" / "perseus.pth")
                break

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])

    def _load_intrinsics(self, intrinsics_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        fs = cv2.FileStorage(str(intrinsics_path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"Could not open intrinsics file: {intrinsics_path}")
        camera_matrix = fs.getNode("camera_matrix").mat()
        dist_coeffs = fs.getNode("dist_coeffs").mat()
        fs.release()
        if camera_matrix is None or dist_coeffs is None:
            raise RuntimeError(f"Missing camera_matrix/dist_coeffs in intrinsics file: {intrinsics_path}")
        return camera_matrix.astype(np.float64), dist_coeffs.reshape(-1, 1).astype(np.float64)

    def _load_model(self, weights_path: Path) -> torch.nn.Module:
        if not weights_path.exists():
            raise RuntimeError(f"Perseus weights not found: {weights_path}")
        model = KeypointCNN()
        try:
            state_dict = torch.load(weights_path, weights_only=True, map_location=self.device)
        except TypeError:
            state_dict = torch.load(weights_path, map_location=self.device)
        for key in list(state_dict.keys()):
            if key.startswith("module."):
                state_dict[key.replace("module.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.get_logger().info(f"Loaded Perseus model from {weights_path}")
        return model

    def _initialize_world_from_marker(self) -> None:
        use_marker_init = bool(self.get_parameter("use_marker_init").value)
        if not use_marker_init:
            self.get_logger().warn("Marker initialization disabled; publishing in camera frame.")
            return

        cache_path = Path(str(self.get_parameter("extrinsics_cache_path").value)).expanduser()
        use_cached = bool(self.get_parameter("use_cached_extrinsics").value)
        allow_cache_fallback = bool(self.get_parameter("allow_cache_fallback").value)

        if use_cached:
            cached = self._load_extrinsics_cache(cache_path)
            if cached is None:
                raise RuntimeError(f"Cached extrinsics requested but not found at {cache_path}")
            self.R_world_camera, self.t_world_camera = cached
            self.get_logger().info(f"Loaded cached camera->world transform from {cache_path}")
            return

        detected = self._detect_marker_transform()
        if detected is not None:
            self.R_world_camera, self.t_world_camera = detected
            if bool(self.get_parameter("cache_extrinsics").value):
                self._save_extrinsics_cache(cache_path, self.R_world_camera, self.t_world_camera)
                self.get_logger().info(f"Saved camera->world transform cache to {cache_path}")
            return

        if allow_cache_fallback:
            cached = self._load_extrinsics_cache(cache_path)
            if cached is not None:
                self.R_world_camera, self.t_world_camera = cached
                self.get_logger().warn(
                    f"Marker detection failed; using cached camera->world transform from {cache_path}"
                )
                return

        raise RuntimeError("Failed to initialize camera->world transform from marker and no cache available.")

    def _detect_marker_transform(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        marker_dict_name = str(self.get_parameter("marker_dict").value)
        if marker_dict_name not in ARUCO_DICT_MAP:
            raise RuntimeError(f"Unsupported marker_dict '{marker_dict_name}'")

        marker_id = int(self.get_parameter("marker_id").value)
        marker_size_m = float(self.get_parameter("marker_size_m").value)
        timeout_sec = float(self.get_parameter("marker_timeout_sec").value)

        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[marker_dict_name])
        if hasattr(cv2.aruco, "DetectorParameters"):
            parameters = cv2.aruco.DetectorParameters()
        else:
            parameters = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            detect_fn = detector.detectMarkers
        else:
            detect_fn = lambda image: cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)  # noqa: E731

        half = marker_size_m / 2.0
        marker_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )

        self.get_logger().info(
            f"Detecting marker id={marker_id} ({marker_dict_name}) for up to {timeout_sec:.1f}s..."
        )
        start = time.time()
        while rclpy.ok() and (time.time() - start) < timeout_sec:
            frame, _ = self.camera.read_latest()
            if frame is None:
                time.sleep(0.01)
                continue

            corners, ids, _ = detect_fn(frame)
            if ids is None:
                continue

            ids_flat = ids.flatten().tolist()
            if marker_id not in ids_flat:
                continue

            idx = ids_flat.index(marker_id)
            image_points = corners[idx].reshape(4, 2).astype(np.float32)

            pnp_flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
            ok, rvec, tvec = cv2.solvePnP(
                marker_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=pnp_flag,
            )
            if not ok:
                continue

            r_marker_camera, _ = cv2.Rodrigues(rvec)
            r_world_camera = r_marker_camera.T
            t_world_camera = -r_world_camera @ tvec.reshape(3)
            self.get_logger().info("Marker-based camera->world transform initialized.")
            return r_world_camera.astype(np.float64), t_world_camera.astype(np.float64)

        self.get_logger().error("Marker detection timed out.")
        return None

    def _save_extrinsics_cache(self, cache_path: Path, rot: np.ndarray, trans: np.ndarray) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fs = cv2.FileStorage(str(cache_path), cv2.FILE_STORAGE_WRITE)
        fs.write("rotation_matrix", rot)
        fs.write("translation", trans.reshape(3, 1))
        fs.write("timestamp", float(time.time()))
        fs.write("marker_id", int(self.get_parameter("marker_id").value))
        fs.release()

    def _load_extrinsics_cache(self, cache_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not cache_path.exists():
            return None
        fs = cv2.FileStorage(str(cache_path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            return None
        rot = fs.getNode("rotation_matrix").mat()
        trans = fs.getNode("translation").mat()
        fs.release()
        if rot is None or trans is None:
            return None
        return rot.astype(np.float64), trans.reshape(3).astype(np.float64)

    def _prepare_model_input(self, frame_bgr: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        h, w = frame_bgr.shape[:2]
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        crop = frame_bgr[y0 : y0 + side, x0 : x0 + side]
        resized = cv2.resize(crop, (self.inference_size, self.inference_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor, (x0, y0, side)

    def _estimate_cube_pose(self, frame_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        model_input, (x0, y0, side) = self._prepare_model_input(frame_bgr)
        with torch.no_grad():
            raw = self.model(model_input).reshape(-1, self.model.n_keypoints, 2)
            keypoints = kornia.geometry.denormalize_pixel_coordinates(
                raw, self.inference_size, self.inference_size
            )[0].cpu().numpy()

        scale = side / float(self.inference_size)
        image_points = keypoints * scale + np.array([x0, y0], dtype=np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            self.object_keypoints,
            image_points.astype(np.float32),
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            return None

        ok, rvec, tvec = cv2.solvePnP(
            self.object_keypoints,
            image_points.astype(np.float32),
            self.camera_matrix,
            self.dist_coeffs,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        r_camera_cube, _ = cv2.Rodrigues(rvec)
        if self.apply_kubric_to_z_up:
            r_camera_cube = r_camera_cube @ KUBRIC_TO_Z_UP

        t_camera_cube = tvec.reshape(3)
        r_world_cube = self.R_world_camera @ r_camera_cube
        t_world_cube = self.R_world_camera @ t_camera_cube + self.t_world_camera
        return r_world_cube, t_world_cube

    def _on_timer(self) -> None:
        frame, camera_ts = self.camera.read_latest()
        if frame is None:
            return
        if camera_ts == self._last_camera_ts:
            return
        self._last_camera_ts = camera_ts

        estimate = self._estimate_cube_pose(frame)
        if estimate is None:
            return

        rot, trans = estimate        

        T_ar_cube = np.eye(4)
        T_ar_cube[:3, :3] = rot
        T_ar_cube[:3, 3] = trans

        T_W_cube = T_W_H @ T_H_ar @ T_ar_cube
        rot, trans = T_W_cube[:3, :3], T_W_cube[:3, 3]

        quat_xyzw = _rotmat_to_quat_xyzw(rot)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame_id
        msg.pose.position.x = float(trans[0])
        msg.pose.position.y = float(trans[1])
        msg.pose.position.z = float(trans[2])
        msg.pose.orientation.x = float(quat_xyzw[0])
        msg.pose.orientation.y = float(quat_xyzw[1])
        msg.pose.orientation.z = float(quat_xyzw[2])
        msg.pose.orientation.w = float(quat_xyzw[3])
        self.pose_pub.publish(msg)

        self._fps_count += 1
        now = time.monotonic()
        if now - self._last_fps_window >= 2.0:
            fps = self._fps_count / (now - self._last_fps_window)
            self.get_logger().debug(f"Estimator FPS: {fps:.1f}")
            self._fps_count = 0
            self._last_fps_window = now

    def destroy_node(self) -> bool:
        if hasattr(self, "camera") and self.camera is not None:
            self.camera.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CubeEstimatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

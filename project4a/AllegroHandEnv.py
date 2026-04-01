import numpy as np
import dm_control
import mujoco as mj

class AllegroHandEnv:
    def __init__(self, physics: dm_control.mjcf.physics.Physics, 
                 q_h_slice: slice, 
                 object_name: str, 
                 num_fingers=4):
        self.physics = physics
        self.q_h_slice = q_h_slice
        self.num_fingers = num_fingers
        self.object_name = object_name

    def set_configuration(self, q_h: np.array):
        self.physics.data.qpos[self.q_h_slice] = q_h
        self.physics.forward()

    def get_body_positions(self, body_names: list[str]):
        """
        Input: list of fingertip names in the XML
        Returns: (num_fingers x 3) np.array containing 
        finger positions in workspace coordinates
        """
        pos_array = np.zeros((len(body_names), 3))
        for i in range(len(body_names)):
            body_id = self.physics.model.body(body_names[i]).id
            body_pos = self.physics.data.body(body_id).xpos
            pos_array[i] = body_pos
        return pos_array

    def get_contact_normals_and_positions(self, contact: mj._structs._MjContactList):
        """
        Input: contact data structure that contains MuJoCo contact information
        Returns the normal vector and positions for each part of the hand that's in contact with the ball
        """
        geom_id_pairs = contact.geom
        model_ptr = self.physics.model.ptr
        contact_struct = self.physics.data.ptr.contact

        # Indices of only the contacts between the object and any part of the hand
        indices = [
            i
            for i, pair in enumerate(geom_id_pairs)
            if "table/table_geom" not in
               [mj.mj_id2name(model_ptr, mj.mjtObj.mjOBJ_GEOM, gid) for gid in pair]
            and ((mj.mj_id2name(model_ptr, mj.mjtObj.mjOBJ_GEOM, pair[0]) == self.object_name)
            or (mj.mj_id2name(model_ptr, mj.mjtObj.mjOBJ_GEOM, pair[1]) == self.object_name))
        ]

        if not indices:
            # No valid contacts
            return np.zeros((0, 9)), np.zeros((0, 3))

        normals_list = []
        for i in indices:
            frame = contact_struct.frame[i]
            arr = np.array(frame)
            if arr.shape == ():
                print(f"Warning: contact_struct.frame[{i}] is scalar, replacing with zeros.")
                arr = np.zeros(9)
            elif arr.size != 9:
                print(f"Warning: contact_struct.frame[{i}] has size {arr.size}, replacing with zeros.")
                arr = np.zeros(9)
            normals_list.append(arr)
        contact_normals = np.vstack(normals_list) if normals_list else np.zeros((0, 9))

        positions_list = []
        for i in indices:
            pos = contact_struct.pos[i]
            arr = np.array(pos)
            if arr.shape == ():
                arr = np.zeros(3)
            elif arr.size != 3:
                arr = np.zeros(3)
            positions_list.append(arr)
        contact_positions = np.vstack(positions_list) if positions_list else np.zeros((0, 3))

        # Negate normal vectors so they point towards the object, only if valid 2D
        # Only negate if valid 2D array
        if contact_normals.ndim == 2 and contact_normals.shape[0] > 0 and contact_normals.shape[1] >= 3:
            contact_normals[:, :3] *= -1
        return contact_normals, contact_positions

class AllegroHandEnvSphere(AllegroHandEnv):
    def __init__(self, physics: dm_control.mjcf.physics.Physics, 
                 sphere_center: int, 
                 sphere_radius: int, 
                 q_h_slice: slice, 
                 object_name: str):
        super().__init__(physics, q_h_slice, object_name)
        self.physics = physics
        self.sphere_center = sphere_center
        self.sphere_radius = sphere_radius
        self.q_h_slice = q_h_slice
        self.num_fingers = 4
    
    def sphere_surface_distance(self, pos: np.array, center: np.array, radius: int):
        """
        Returns the distance from pos to the surface of a sphere with a specified
        radius and center
        """
        d = np.linalg.norm(pos - center) - radius
        return d

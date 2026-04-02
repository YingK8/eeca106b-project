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

    def get_fingertip_jacobian(self, tip_name: str) -> np.ndarray:
        """
        Returns the 3xN Jacobian for the given fingertip body, mapping hand joint velocities to fingertip Cartesian velocity.
        N = number of hand joints (typically 16 for Allegro hand).
        """
        # Get the body id for the fingertip
        body_id = self.physics.model.body(tip_name).id
        # Number of DoFs in the model
        n_dof = self.physics.model.nv
        # Jacobian buffer: (3, n_dof)
        jacp = np.zeros((3, n_dof))
        # Compute translational Jacobian for the body CoM
        mj.mj_jacBodyCom(self.physics.model.ptr, self.physics.data.ptr, jacp, None, body_id)
        # Only keep columns for hand joints
        if hasattr(self, 'q_h_slice'):
            jacp_hand = jacp[:, self.q_h_slice]
        else:
            jacp_hand = jacp
        return jacp_hand

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

        contact_normals = np.array([contact_struct.frame[i] for i in indices])
        contact_positions = np.array([contact_struct.pos[i] for i in indices])
        # Negate normal vectors so they point towards the object
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

import kinpy as kp
import numpy as np

from typing import List


class KinematicsSolver:
    def __init__(self, urdf_path: str, eef_link: str):
        with open(urdf_path, "rb") as file:
            urdf_data = file.read()
        self.chain = kp.build_serial_chain_from_urdf(urdf_data, eef_link)
        print(self.chain)

    def fk(self, joint_angles: np.ndarray) -> np.ndarray:
        """forward_kinematics"""
        transformation_matrix = self.chain.forward_kinematics(joint_angles)
        return transformation_matrix

    def ik(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        initial_state: np.ndarray,
    ) -> np.ndarray:
        """
        inverse_kinematics
        NOTE: initial_state is the initial joint angles for the solver
        """
        # Target position and orientation (if provided)
        target = kp.Transform(
            rot=target_orientation,
            pos=target_position,
        )
        # Calculate inverse kinematics
        joint_angles = self.chain.inverse_kinematics(target, initial_state)
        return joint_angles

    def joint_names(self) -> List[str]:
        """Get a list of joint names"""
        return self.chain.get_joint_parameter_names()

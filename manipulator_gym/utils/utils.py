import numpy as np
import math


def rotationMatrixToEulerAngles(R):
    """
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def eulerAnglesToRotationMatrix(roll, pitch, yaw):
    """
    Convert euler angles to rotation matrix
    """
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)],
        ]
    )
    R_y = np.array(
        [
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)],
        ]
    )
    R_z = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    return np.dot(R_z, np.dot(R_y, R_x))


def convert_img(data):
    """
    This is used to convert ros img since cv_bridge is giving
    errors on nvidia jetson
    """
    dtype = np.dtype("uint8")  # Hardcode to 8 bits...
    dtype = dtype.newbyteorder(">" if data.is_bigendian else "<")
    return np.ndarray(shape=(data.height, data.width, 3), dtype=dtype, buffer=data.data)


class WorkspaceBoundary:
    """
    This class defines the workspace boundary of the robot.
    TODO (YL)
    """

    @staticmethod
    def make_from_rectangular_boundary(lower_bound, upper_bound):
        """
        This function creates a workspace boundary from a
        rectangular boundary.
        """
        raise NotImplementedError

    @staticmethod
    def make_from_sphere_boundary(lower_bound, upper_bound):
        """
        This function creates a workspace boundary from a
        spherical boundary.
        """
        raise NotImplementedError

    def check_if_within_boundary(self, pose):
        """
        This function checks if the given pose is within the boundary.
        """
        raise NotImplementedError

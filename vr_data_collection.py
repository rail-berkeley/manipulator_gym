
import numpy as np
from absl import app, flags, logging

import sys
import cv2
import time

from pyquaternion import Quaternion

from manipulator_gym.manipulator_env import ManipulatorEnv
from manipulator_gym.interfaces.base_interface import ManipulatorInterface
from manipulator_gym.interfaces.interface_service import ActionClientInterface
import manipulator_gym.utils.transformation_utils as tr
from manipulator_gym.utils.gym_wrappers import ResizeObsImageWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("rlds_output", None, "Path to log the data.")
flags.DEFINE_string("ds_name", "test_vrmani_env", "Name of the dataset.")
flags.DEFINE_string("ip", "localhost", "IP address of the robot server.")
flags.DEFINE_bool("show_img", False, "Whether to visualize the images or not.")
flags.DEFINE_bool("test", False, "Whether to test the data collection or not.")
flags.DEFINE_bool("resize_img", False, "Whether to resize the images or not.")

FLAGS(sys.argv)


# factor of oculus linear movement to robot linear movement
LINEAR_ACTION_FACTOR = 1.1
# how much to clip the linear action
CLIP_LINEAR_ACTION = 0.02
# how much to rotate every step
ANGULAR_ACTION_PER_STEP = 0.06

TIME_STEP = 0.1


class TeleopDataCollector:
    def __init__(self, oculus_reader, env, log_dir=None) -> None:
        self.ref_point_set = False
        self.gripper_open = False
        self.reference_vr_transform = None
        self.reference_robot_transform = None

        self.a_button_press = False
        self.a_button_down = False
        self.b_button_press = False
        self.b_button_down = False
        self.RTr_button_press = False
        self.RTr_button_down = False
        self.RJ_button_press = False
        self.RJ_button_down = False

        self.oculus_reader = oculus_reader
        self.env = env

        self.one_traj_data = np.array([])
        self.traj_num = 0
        self.log_dir = log_dir  # this will also track if the data collection is enabled or not
        self.current_language_text = None

        # if log_dir is not None, then we will log the data as RLDS
        # Uses: https://github.com/rail-berkeley/oxe_envlogger
        if log_dir is not None:
            import tensorflow_datasets as tfds
            from oxe_envlogger.envlogger import OXEEnvLogger

            step_metadata_info = {'language_text': tfds.features.Text(
                doc="Language embedding for the episode.")
            }
            self.env = OXEEnvLogger(
                env,
                dataset_name="test_vrmani_env",
                directory=log_dir,
                max_episodes_per_file=1,
                step_metadata_info=step_metadata_info,
            )

    def oculus_to_robot(self, current_vr_transform):
        z_rot = tr.RpToTrans(Quaternion(
            axis=[0, 0, 1], angle=-np.pi / 2).rotation_matrix)
        x_rot = tr.RpToTrans(Quaternion(
            axis=[1, 0, 0], angle=np.pi / 2).rotation_matrix)
        current_vr_transform = z_rot.dot(x_rot).dot(current_vr_transform)
        return current_vr_transform

    def teleop_data_collection(self):
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
        if "r" in transformations:
            vr_transform = transformations['r']

        # Button Handling
        if len(buttons.keys()) <= 0:
            return

        if buttons["A"] and not self.a_button_down:
            self.a_button_down = True

        if self.a_button_down and not buttons["A"]:
            self.a_button_press = True
            self.a_button_down = False

        if buttons["B"] and not self.b_button_down:
            self.b_button_down = True

        if self.b_button_down and not buttons["B"]:
            self.b_button_press = True
            self.b_button_down = False

        if buttons["RTr"] and not self.RTr_button_down:
            self.RTr_button_down = True

        if self.RTr_button_down and not buttons["RTr"]:
            self.RTr_button_press = True
            self.RTr_button_down = False

        if buttons["RJ"] and not self.RJ_button_down:
            self.RJ_button_down = True

        if self.RJ_button_down and not buttons["RJ"]:
            self.RJ_button_press = True
            self.RJ_button_down = False

        # Controlling the arm with VR (interacting through the env)
        if (self.RJ_button_press):
            self.env.close()
            print("Stopping data collection")
            return True

        if (self.b_button_press):
            if len(self.one_traj_data) > 0:
                # np.save('traj_' + str(self.traj_num), self.one_traj_data)
                self.traj_num += 1
                self.one_traj_data = np.array([])
            print("Starting a Trajectory")

            # call this to get the task text from user during data collection logging mode
            if self.log_dir is not None:
                # NOTE: provide input task to the language_text field by getting the task
                if FLAGS.test:
                    task_text = "dummy task text"  # test mode
                else:
                    task_text = input("Enter task text: ") or self.current_language_text

                self.current_language_text = task_text
                self.env.set_step_metadata(
                    {"language_text": self.current_language_text})
            self.env.reset()

            self.ref_point_set = True
            self.gripper_open = True # Default open?

            self.reference_vr_transform = self.oculus_to_robot(vr_transform)
            self.initial_vr_offset = tr.RpToTrans(
                np.eye(3), self.reference_vr_transform[:3, 3])
            self.reference_vr_transform = tr.TransInv(
                self.initial_vr_offset).dot(self.reference_vr_transform)

            self.b_button_press = False

        # Handling reference positions and starting and stopping VR using the A button
        if self.a_button_press and not self.ref_point_set:
            print("Reference point set, motion VR enabled")
            self.ref_point_set = True

            self.reference_vr_transform = self.oculus_to_robot(vr_transform)
            self.initial_vr_offset = tr.RpToTrans(
                np.eye(3), self.reference_vr_transform[:3, 3])
            self.reference_vr_transform = tr.TransInv(
                self.initial_vr_offset).dot(self.reference_vr_transform)

            self.a_button_press = False

        if self.a_button_press and self.ref_point_set:
            print("Reference point deactivated, motion VR disabled")
            self.ref_point_set = False
            self.reference_vr_transform = None
            self.initial_vr_offset = None
            self.a_button_press = False

        # Performing a step in the env if ref_point and VR enabled motion has been activated
        # Copy the most recent T_yb transform into a temporary variable
        # Calulating relative changes and putting into action
        if self.gripper_open:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1])
        else:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])

        """Will start collecting data if the reference point is set"""
        if self.ref_point_set:
            current_vr_transform = self.oculus_to_robot(vr_transform)
            delta_vr_transform = current_vr_transform.dot(
                tr.TransInv(self.reference_vr_transform))
            M_delta, v_delta = tr.TransToRp(delta_vr_transform)

            action[:3] = v_delta*LINEAR_ACTION_FACTOR
            self.reference_vr_transform = self.oculus_to_robot(vr_transform)

            # arm_eul =  ang.rotationMatrixToEulerAngles(ref_orientation_arm)
            # arm_eul[0] -= vr_eul[1]
            # arm_eul[1] -= vr_eul[0]
            # arm_eul[2] -= vr_eul[2]
            # T_yb[:3,:3] = ang.eulerAnglesToRotationMatrix(arm_eul)
            # self.update_T_yb()

        # handling gripper open/close
        if (self.RTr_button_press):
            if self.gripper_open:
                self.gripper_open = False
                action[6] = 1
            else:
                self.gripper_open = True
                action[6] = 0
            self.RTr_button_press = False

        # Use button for orientation
        if buttons['rightJS'][0] < -0.6:
            action[3] = -ANGULAR_ACTION_PER_STEP
        if buttons['rightJS'][0] > 0.6:
            action[3] = ANGULAR_ACTION_PER_STEP

        if buttons['rightJS'][1] < -0.6:
            action[4] = -ANGULAR_ACTION_PER_STEP
        if buttons['rightJS'][1] > 0.6:
            action[4] = ANGULAR_ACTION_PER_STEP

        # Applying relative changes by stepping the environment
        # Define the range you want to bound the values to
        # Use np.clip() to bound the values within the specified range
        action[:3] = np.clip(action[:3], -CLIP_LINEAR_ACTION, CLIP_LINEAR_ACTION)

        print("gripper state: ", action[6])
        print(action)

        assert len(action) == 7
        if self.ref_point_set:
            self.one_traj_data = np.append(
                self.one_traj_data, {"action": action})

            if self.log_dir is not None:
                self.env.set_step_metadata(
                    {"language_text": self.current_language_text})

            # cast action to float32 TODO: better and more robust way in oxeenvlogger
            action = action.astype(np.float32)
            self.env.step(action)

        time.sleep(TIME_STEP)
        return False


class DummyOculusReader:
    """This is a dummy oculus reader to mimic the oculus reader for testing purposes"""

    def __init__(self, num_steps=10, num_of_episodes=3):
        self.num_steps = num_steps
        self.num_of_episodes = num_of_episodes
        self.current_step = self.num_steps  # to trigger the reset
        self.current_episode = 0
        self._end_of_logging = False  # some bad hack to end the logging

    def get_transformations_and_buttons(self):
        """
        This will mimic the oculus reader and return the transformations and buttons
        call the reset every start and end the episode after max steps is reached,
        then call end the reader when the num of episodes is reached
        """
        default_val = {
            "r": np.eye(4)
        }, {
            "A": False,
            "B": False,
            "RTr": False,
            "RJ": False,
            "rightJS": [0, 0]
        }

        if self.current_step == self.num_steps:
            print("mimic starting new trajectory")
            self.current_step = 0
            default_val[1]["B"] = True
            self.current_episode += 1
            return default_val

        if self.current_episode == self.num_of_episodes:
            print("mimic end of logging")
            if not self._end_of_logging:
                default_val[1]["RJ"] = True
            self._end_of_logging = True
            return default_val

        self.current_step += 1
        return default_val


if __name__ == '__main__':
    if FLAGS.test:
        env = ManipulatorEnv(manipulator_interface=ManipulatorInterface())
        reader = DummyOculusReader(num_of_episodes=40)
    else:
        from oculus_reader.reader import OculusReader
        env = ManipulatorEnv(
            manipulator_interface=ActionClientInterface(host=FLAGS.ip))
        reader = OculusReader()

    # this will dictate the size of the image that will be logged
    if FLAGS.resize_img:
        env = ResizeObsImageWrapper(
            env,
            resize_size={"image_primary": (256, 256), "image_wrist": (128, 128)}
        )

    data_collector = TeleopDataCollector(
        oculus_reader=reader,
        env=env,
        log_dir=f"{FLAGS.rlds_output}/{FLAGS.ds_name}/0.1.0"
    )

    stopDataCollection = False
    while not stopDataCollection:
        if FLAGS.show_img:
            obs = data_collector.env.obs()
            img = obs["image_primary"]
            cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(10)
            img = obs["image_wrist"]
            cv2.imshow("image_wrist", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(10)

        stopDataCollection = data_collector.teleop_data_collection()

import numpy as np
import time
from typing import Optional, List, Any
import logging

from agentlace.action import ActionClient, ActionServer, ActionConfig
from agentlace.internal.utils import mat_to_jpeg, jpeg_to_mat

from manipulator_gym.interfaces.base_interface import ManipulatorInterface
from typing import Optional
import cv2


# These describes the ports and API keys used in the agentlace server
DefaultActionConfig = ActionConfig(
    port_number=5556,
    action_keys=[
        "reset",
        "step_action",
        "configure",
        "move_eef",
        "move_gripper",
        "custom_fn",
    ],
    observation_keys=["eef_pose", "gripper_state", "primary_img", "wrist_img"],
    broadcast_port=5556 + 1,
)


class ActionClientInterface(ManipulatorInterface):
    """Action Client interface with agentlace."""

    def __init__(
        self, host: str = "localhost", port: int = 5556, obs_timeout: float = 0.05
    ):
        """
        Initialize the action client interface.
        Args:
            host: the host of the server
            port: the port number of the server
            obs_timeout: the timeout for getting the observation
        """
        _config = DefaultActionConfig
        _config.port_number = port
        _config.broadcast_port = port + 1
        self._client = ActionClient(host, _config)
        self.timeout = obs_timeout
        self.last_get_obs_time = 0
        self._update_full_obs()
        # Assuming the environment parameters and image size are predefined
        print("initializing action client interface")

    @property
    def eef_pose(self) -> np.ndarray:
        self._update_full_obs()
        return self.latest_obs["eef_pose"]

    @property
    def gripper_state(self) -> float:
        self._update_full_obs()
        return self.latest_obs["gripper_state"]

    @property
    def primary_img(self) -> np.ndarray:
        # NOTE: we convert img to jpeg for lower data transfer
        self._update_full_obs()
        return jpeg_to_mat(self.latest_obs["primary_img"])

    @property
    def wrist_img(self) -> Optional[np.ndarray]:
        # NOTE: we convert img to jpeg for lower data transfer
        self._update_full_obs()
        if "wrist_img" in self.latest_obs:
            return jpeg_to_mat(self.latest_obs["wrist_img"])
        return None  # wrist_img is not available

    def step_action(self, action: np.ndarray) -> bool:
        res = self._client.act("step_action", action)
        if res and "status" in res and res["status"]:
            return True
        return False

    def move_eef(self, pose: np.ndarray) -> bool:
        res = self._client.act("move_eef", pose)
        if res and "status" in res and res["status"]:
            return True

    def reset(self, **kwargs) -> bool:
        res = self._client.act("reset", kwargs)
        if res and "status" in res and res["status"]:
            return True
        return False

    def configure(self, **kwargs) -> bool:
        res = self._client.act("configure", kwargs)
        if res and "status" in res and res["status"]:
            return True
        return False

    def custom_fn(self, fn_name: str, **kwargs) -> Any:
        """
        Experimental function.
        This is a generic method to call any custom function
        in the interface.
        """
        res = self._client.act("custom_fn", {"fn_name": fn_name, "kwargs": kwargs})
        if res and "status" in res and res["status"]:
            return res.get("res_payload", None)
        return res

    def _update_full_obs(self):
        """This avoids calling the obs() method multiple times."""
        if time.time() - self.last_get_obs_time > self.timeout:
            latest_obs = self._client.obs()
            while latest_obs is None:
                print("waiting for observation")
                latest_obs = self._client.obs()
                time.sleep(0.5)
            self.latest_obs = latest_obs
            self.last_get_obs_time = time.time()


##############################################################################


class ManipulatorInterfaceServer:
    """
    Interface Server with runs the interface on the robot with agentlace.
    """

    def __init__(
        self,
        manipulator_interface: ManipulatorInterface,
        port: int = 5556,
        resize_img: Optional[List[int]] = None,
    ):
        """
        Provide the manipulator interface to the server.
        args:
            manipulator_interface: the interface to the robot
            port: the port number to run the server
            resize_img: resize the img before sending to the client for
                        lower data transfer, default is None
        """
        super().__init__()
        self._manipulator_interface = manipulator_interface
        _config = DefaultActionConfig
        _config.port_number = port
        _config.broadcast_port = port + 1
        print(f"initializing manipulator interface server with port: {port}")

        self.__server = ActionServer(
            _config,
            obs_callback=self.__observe,
            act_callback=self.__action,
            # hide all logs
            log_level=logging.CRITICAL,
        )
        self._resize_img = resize_img

    def start(self, threaded: bool = False):
        """
        This starts the server. Default is blocking.
        """
        self.__server.start(threaded)

    def stop(self):
        """Stop the server."""
        self.__server.stop()

    def __observe(self, types: list) -> dict:
        # we will default return the full observation
        # NOTE: we resize and convert img to jpeg for lower data transfer
        def resize_img_fn(img):
            if self._resize_img is not None:
                img = cv2.resize(img, tuple(self._resize_img))
            return img

        obs = {
            "eef_pose": self._manipulator_interface.eef_pose,
            "gripper_state": self._manipulator_interface.gripper_state,
            "primary_img": mat_to_jpeg(
                resize_img_fn(self._manipulator_interface.primary_img)
            ),
        }
        # Add wrist image to the observation if available
        if self._manipulator_interface.wrist_img is not None:
            obs["wrist_img"] = mat_to_jpeg(
                resize_img_fn(self._manipulator_interface.wrist_img)
            )
        return obs

    def __action(self, type: str, req_payload) -> dict:
        if type == "reset":
            status = self._manipulator_interface.reset(**req_payload)
        elif type == "configure":
            status = self._manipulator_interface.configure(**req_payload)
        elif type == "step_action":
            status = self._manipulator_interface.step_action(req_payload)
        elif type == "move_eef":
            status = self._manipulator_interface.move_eef(req_payload)
        elif type == "move_gripper":
            status = self._manipulator_interface.move_gripper(req_payload)
        elif type == "custom_fn":
            # **Call a custom_fn or method of the interface
            fn_name = req_payload["fn_name"]
            if hasattr(self._manipulator_interface, fn_name):
                res_payload = getattr(self._manipulator_interface, fn_name)(
                    **req_payload["kwargs"]
                )
                return {"status": True, "res_payload": res_payload}

            print(f"Method {fn_name} not found in the interface")
            return {"status": False, "error": "Method not found"}
        else:
            raise ValueError(f"Invalid action type: {type}")
        print(f"Action: {type} status: {status}")
        return {"status": status}

"""
This script to eval OpenVLA model on bridge data robot setup.
"""
from PIL import Image
import threading
import os
import time
import numpy as np
from typing import Any, Dict, Optional
from absl import app, flags, logging
import cv2
from manipulator_gym.utils.gym_wrappers import (
    ConvertState2Proprio,
    ResizeObsImageWrapper,
)
from manipulator_gym.manipulator_env import ManipulatorEnv
from manipulator_gym.interfaces.interface_service import ActionClientInterface

FLAGS = flags.FLAGS
flags.DEFINE_string("ip", "localhost", "IP address of the robot server.")
flags.DEFINE_integer("port", 5556, "Port of the manipulator server.")
flags.DEFINE_bool("show_img", False, "Whether to visualize the images or not.")
flags.DEFINE_string(
    "text_cond", 'put the eggplant on the orange plate', "Language prompt for the task."
)
flags.DEFINE_string("lora_adapter_dir", None, "Path to the LORA adapter directory.")
flags.DEFINE_string(
    "dataset_stats",
    None,
    "Path to the dataset stats json file, default to brige_orig.",
)

np.set_printoptions(precision=2, suppress=True)
device = "cuda:0"

def get_single_img(obs):
    img = obs["image_primary"]
    return img[-1] if img.ndim == 4 else img

class OpenVLAPolicy():
    
    def __init__(self, lora_adapter_dir=None, dataset_stats_path=None, device="cuda:0"):
        self.lora_adapter_dir = lora_adapter_dir
        self.dataset_stats_path = dataset_stats_path
        self.device = device
        self.agent = self.create_agent()
    
    def create_agent(self):
        import peft
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor

        # Load Processor & VLA
        processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True
        )
        base_vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            #attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # For fine-tuned VLA policies, need to add the Lora adapters and new dataset stats
        if self.lora_adapter_dir is not None:
            assert (
                self.dataset_stats_path is not None
            ), "fine-tuned VLA usually requires custom dataset stats"
            adapter_dir = self.config["lora_adapter_dir"]
            print(f"Loading LORA Adapter from: {adapter_dir}")
            vla = peft.PeftModel.from_pretrained(base_vla, adapter_dir)
            vla = (
                vla.merge_and_unload()
            )  # this merges the adapter into the model for faster inference
        else:
            vla = base_vla
        vla = vla.to(self.device)

        # Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if self.dataset_stats_path is not None:
            dataset_stats_path = self.dataset_stats_path
            assert (
                "dataset_statistics.json" in dataset_stats_path
            ), "Please provide the correct dataset statistics file."
            path = os.path.expanduser(dataset_stats_path)
            print(f"Loading custom dataset statistics .json from: {path}")

            with open(path, "r") as f:
                import json

                vla.norm_stats = json.load(f)

                # assume only one key in the .json file and gets the key
                dataset_names = vla.norm_stats.keys()
                assert (
                    len(dataset_names) == 1
                ), "Only one dataset name should be in the .json file."
                unnorm_key = list(dataset_names)[0]
        else:
            unnorm_key = "bridge_orig"

        print(f"Un-normalization key: {unnorm_key}")

        # Construct a callable function to predict actions
        def _call_action_fn(obs_dict, language_instruction):
            image = obs_dict["image_primary"]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_cond = Image.fromarray(image)

            prompt = f"In: What action should the robot take to {language_instruction}?\nOut:"
            inputs = processor(prompt, image_cond).to(self.device, dtype=torch.bfloat16)

            # Predict Action (7-DoF; un-normalize for BridgeData V2)
            action = vla.predict_action(
                **inputs, unnorm_key=unnorm_key, do_sample=False
            )
            assert (
                len(action) == 7
            ), f"Action size should be in x, y, z, rx, ry, rz, gripper"

            return action

        self._call_action_fn = _call_action_fn

    def __call__(self, obs_dict, language_instruction):
        assert "image_primary" in obs_dict.keys()
        assert isinstance(obs_dict["image_primary"], np.ndarray)
        assert len(obs_dict["image_primary"].shape) == 3
        return self._call_action_fn(obs_dict, language_instruction)

    def reset(self):
        pass

def main(_):
    interface = ActionClientInterface(host=FLAGS.ip, port=FLAGS.port)
    env = ManipulatorEnv(
        manipulator_interface=interface,
    )  # default doesn't use wrist cam

    env = ConvertState2Proprio(env)
    env = ResizeObsImageWrapper(
        env, resize_size={"image_primary": (256, 256)}
    )

    # Load Processor & VLA
    eval_policy = OpenVLAPolicy(lora_adapter_dir=FLAGS.lora_adapter_dir, dataset_stats_path=FLAGS.dataset_stats)

    def run_continuous_img_primary(manipulator_interface):
        """Continuously run manipulator_interface.img_primary in a loop."""
        while True:
            try:
                _ = manipulator_interface.primary_img
            except Exception as e:
                print(f"Error in continuous img_primary thread: {e}")
                break

    def start_img_primary_thread(manipulator_interface):
        """Start a thread that continuously runs img_primary."""
        img_primary_thread = threading.Thread(
            target=run_continuous_img_primary,
            args=(manipulator_interface,),
            daemon=True  # This ensures the thread will be terminated when the main program exits
        )
        img_primary_thread.start()
        return img_primary_thread
    img_primary_thread = start_img_primary_thread(interface)
    try:
        # running rollouts
        for _ in range(100):
            obs, info = env.reset()
            eval_policy.reset()
            episode_return = 0.0
            for i in range(250):
                start_time = time.time()

                # if FLAGS.show_img:
                #     show_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                #     cv2.imshow("image", show_img)
                #     # capture "r" key and reset
                #     if cv2.waitKey(10) & 0xFF == ord("r"):
                #         break

                actions = eval_policy(obs, FLAGS.text_cond)

                print("--- VLA inference took %s seconds ---" % (time.time() - start_time))
                print(f" Step {i}: performing action: {actions}")

                # step env -- info contains full "chunk" of observations for logging
                # obs only contains observation for final step of chunk
                obs, reward, done, trunc, info = env.step(actions)
                episode_return += reward

                if done:
                    break

            print(f"Episode return: {episode_return}")
    except KeyboardInterrupt:
        env.reset()
        quit()


if __name__ == "__main__":
    app.run(main)

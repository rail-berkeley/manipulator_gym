"""
This script to eval OpenVLA model on bridge data robot setup.
"""

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import os
import torch
import time
import peft
import numpy as np
from absl import app, flags, logging
import cv2

from manipulator_gym.manipulator_env import ManipulatorEnv
from manipulator_gym.interfaces.interface_service import ActionClientInterface
from manipulator_gym.utils.gym_wrappers import ClipActionBoxBoundary

FLAGS = flags.FLAGS
flags.DEFINE_string("ip", "localhost", "IP address of the robot server.")
flags.DEFINE_integer("port", 5556, "Port of the manipulator server.")
flags.DEFINE_bool("show_img", False, "Whether to visualize the images or not.")
flags.DEFINE_string(
    "text_cond", "put the banana on the plate", "Language prompt for the task."
)
flags.DEFINE_string("lora_adapter_dir", None, "Path to the LORA adapter directory.")
flags.DEFINE_bool("clip_actions", False, "Clip actions to 0.02")
flags.DEFINE_string(
    "dataset_stats",
    "bridge_orig",
    "Path to the dataset stats json file, default to brige_orig.",
)
# Example lora_adapter_dir: "adapter-tmp/openvla-7b+serl_demos+b4+lr-2e-05+lora-r32+dropout-0.0+q-4bit/"


# np decimal printout to 2 decimal places
np.set_printoptions(precision=2, suppress=True)
device = "cuda:0"


def main(_):
    env = ManipulatorEnv(
        manipulator_interface=ActionClientInterface(host=FLAGS.ip, port=FLAGS.port),
        # manipulator_interface=ManipulatorInterface(), # for testing
    )  # default doesn't use wrist cam
    # NOTE: using the kitchen sink setup boundary of https://github.com/simpler-env/SimplerEnv
    env = ClipActionBoxBoundary(
        env, workspace_boundary=[[-float('inf'), -float('inf'), -float('inf')], [float('inf'), float('inf'), float('inf')]]
    )

    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True
    )
    base_vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    if FLAGS.lora_adapter_dir is not None:
        # Load LORA Adapter after finetuning
        print(f"Loading LORA Adapter from: {FLAGS.lora_adapter_dir}")
        vla = peft.PeftModel.from_pretrained(base_vla, FLAGS.lora_adapter_dir)
        vla = (
            vla.merge_and_unload()
        )  # this merges the adapter into the model for faster inference
    else:
        vla = base_vla

    # Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
    unnorm_key = FLAGS.dataset_stats
    if ".json" in FLAGS.dataset_stats:
        assert (
            "dataset_statistics.json" in FLAGS.dataset_stats
        ), "Please provide the correct dataset statistics file."
        path = os.path.expanduser(FLAGS.dataset_stats)
        print(f"Loading custom dataset statistics .json from: {path}")
        import json

        with open(path, "r") as f:
            vla.norm_stats = json.load(f)

            # assume only one key in the .json file and gets the key
            dataset_names = vla.norm_stats.keys()
            assert (
                len(dataset_names) == 1
            ), "Only one dataset name should be in the .json file."
            unnorm_key = list(dataset_names)[0]

    print(f"Un-normalization key: {unnorm_key}")

    # format the prompt
    prompt = f"In: What action should the robot take to {FLAGS.text_cond}?\nOut:"

    # running rollouts
    for _ in range(100):
        obs, info = env.reset()
        episode_return = 0.0
        for i in range(100):
            start_time = time.time()

            image = obs["image_primary"]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB

            if FLAGS.show_img:
                cv2.imshow("image", image)
                # capture "r" key and reset
                if cv2.waitKey(10) & 0xFF == ord("r"):
                    break

            image_cond = Image.fromarray(image)
            inputs = processor(prompt, image_cond).to(device, dtype=torch.bfloat16)

            # Predict Action (7-DoF; un-normalize for BridgeData V2)
            action = vla.predict_action(
                **inputs, unnorm_key=unnorm_key, do_sample=False
            )
            assert (
                len(action) == 7
            ), f"Action size should be in x, y, z, rx, ry, rz, gripper"

            print("--- VLA inference took %s seconds ---" % (time.time() - start_time))
            print(f" Step {i}: performing action: {action}")

            if FLAGS.clip_actions:
                action[:6] = np.clip(action[:6], -0.02, 0.02)
                print(f"Clipped action: {action}")

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(action)
            episode_return += reward

            if done:
                break

        print(f"Episode return: {episode_return}")


if __name__ == "__main__":
    app.run(main)

"""
This is a script modified from
https://github.com/octo-models/octo/blob/main/examples/03_eval_finetuned.py
"""

from absl import app, flags, logging
import gym
import jax
import numpy as np
import cv2

from manipulator_gym.manipulator_env import ManipulatorEnv, StateEncoding
from manipulator_gym.interfaces.interface_service import ActionClientInterface
from manipulator_gym.interfaces.base_interface import ManipulatorInterface
from manipulator_gym.utils.gym_wrappers import (
    ConvertState2Proprio,
    ResizeObsImageWrapper,
    ClipActionBoxBoundary,
)

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, RHCWrapper, \
    NormalizeProprio, TemporalEnsembleWrapper

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to Octo checkpoint directory.")
flags.DEFINE_string("ip", "localhost", "IP address of the robot server.")
flags.DEFINE_bool("show_img", False, "Whether to visualize the images or not.")
flags.DEFINE_string("text_cond", "put the banana on the plate", "Language prompt for the task.")


def main(_):
    # load finetuned model
    logging.info("Loading finetuned model...")
    if not FLAGS.checkpoint_path:
        model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
    else:
        model = OctoModel.load_pretrained(FLAGS.checkpoint_path)

    # make gym environment
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_0": ...
    #     "image_1": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #     "goal": {
    #       "image_0": ...
    #       "image_1": ...
    #     }
    #   }
    ##################################################################################################################
    env = ManipulatorEnv(
        manipulator_interface=ActionClientInterface(host=FLAGS.ip),
        # manipulator_interface=ManipulatorInterface(), # for testing
        state_encoding=StateEncoding.POS_EULER,
        use_wrist_cam=True,
    )
    env = ClipActionBoxBoundary(env, workspace_boundary=[[0.0, -0.5, -0.5], [0.6, 0.5, 0.5]])
    env = ConvertState2Proprio(env)
    env = ResizeObsImageWrapper(env, resize_size={"image_primary": (256, 256), "image_wrist": (128, 128)})

    # # add wrappers for history and "receding horizon control", i.e. action chunking
    # env = HistoryWrapper(env, horizon=2)
    env = HistoryWrapper(env, horizon=1)
    env = TemporalEnsembleWrapper(env, 4)
    # env = RHCWrapper(env, exec_horizon=4)

    # wrap env to handle action/proprio normalization -- match normalization type to the one used during finetuning
    # this wrapper can only be used when there is proprio metadata (dataset_statistics['proprio'])
    # else use dataset_statistics['action'] in model.sample_actions()
    # env = NormalizeProprio(
    #     env,
    #     model.dataset_statistics["bridge_dataset"],
    # )

    # running rollouts
    for _ in range(100):
        obs, info = env.reset()

        # Create task specification --> use model utility to create task dict with correct entries
        language_instruction = [FLAGS.text_cond]
        task = model.create_tasks(texts=language_instruction)
        # task = model.create_tasks(goals={"image_primary": img})   # for goal-conditioned

        episode_return = 0.0
        for i in range(250):
            if FLAGS.show_img:
                img = obs["image_primary"][0]
                cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # img2 = obs["image_wrist"][0]
                # cv2.imshow("image_wrist", cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
                # capture "r" key and reset
                if cv2.waitKey(10) & 0xFF == ord("r"):
                    break

            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            actions = model.sample_actions(
                jax.tree_map(lambda x: x[None], obs), 
                task,
                # NOTE: we are using bridge_dataset's statistics for default normalization
                unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
                rng=jax.random.PRNGKey(0)
            )
            actions = actions[0]
            print("performing action: ",actions)
            print(f"Step {i} with action size of {len(actions)}")
            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)
            episode_return += reward

        print(f"Episode return: {episode_return}")

if __name__ == "__main__":
    app.run(main)

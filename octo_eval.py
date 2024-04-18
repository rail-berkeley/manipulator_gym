"""
This is a script modified from
https://github.com/octo-models/octo/blob/main/examples/03_eval_finetuned.py
"""

from absl import app, flags, logging
import gym
import jax
import numpy as np
import wandb
import cv2

from manipulator_gym.manipulator_env import ManipulatorEnv, StateEncoding
from manipulator_gym.interfaces.interface_service import ActionClientInterface

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, RHCWrapper, UnnormalizeActionProprio

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to Octo checkpoint directory.")
flags.DEFINE_string("ip", "localhost", "IP address of the robot server.")
flags.DEFINE_bool("show_img", False, "Whether to visualize the images or not.")
flags.DEFINE_string("text_cond", "put the banana on the plate", "Language prompt for the task.")


class ConvertState2Propio(gym.Wrapper):
    """
    convert dict key 'state' to 'proprio' to comply to bridge_dataset stats
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict()
        # convert in obs space
        for key, value in self.env.observation_space.spaces.items():
            if key == "state":
                self.observation_space["proprio"] = value
            else:
                self.observation_space[key] = value

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        obs["proprio"] = obs["state"]
        obs.pop("state")
        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["proprio"] = obs["state"]
        obs.pop("state")
        return obs, info


def main(_):
    # load finetuned model
    logging.info("Loading finetuned model...")
    if not FLAGS.checkpoint_path:
        model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
        # model = OctoModel.load_pretrained("./oier_models")
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
        state_encoding=StateEncoding.POS_EULER,
        use_wrist_cam=True,
    )
    env = ConvertState2Propio(env)
    # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=4)
    # NOTE: we are using bridge_dataset's statistics for default normalization
    # wrap env to handle action/proprio normalization -- match normalization type to the one used during finetuning
    env = UnnormalizeActionProprio(
        env, model.dataset_statistics["bridge_dataset"], normalization_type="normal"
    )
    # running rollouts
    for _ in range(3):
        one_traj_data = np.array([])
        obs, info = env.reset()
        # create task specification --> use model utility to create task dict with correct entries
        language_instruction = [FLAGS.text_cond]
        task = model.create_tasks(texts=language_instruction)
        # task = model.create_tasks(goals={"image_primary": img})   # for goal-conditioned

        episode_return = 0.0
        for i in range(150):
            if FLAGS.show_img:
                img = obs["image_primary"][0]
                cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                cv2.waitKey(10)

            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            actions = model.sample_actions(
                jax.tree_map(lambda x: x[None], obs), task, rng=jax.random.PRNGKey(0)
            )
            actions = actions[0]
            # actions = actions[:, :-1] # TODO(YL) why needed?
            # actions = actions * 2
            print("performing action: ",actions)
            print(f"Step {i} with action size of {len(actions)}")
            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            one_traj_data = np.append(one_traj_data, {"action" : actions})
            obs, reward, done, trunc, info = env.step(actions)
            episode_return += reward

        # np.save('octo_eval_traj' + str(i), one_traj_data)
        print(f"Episode return: {episode_return}")

if __name__ == "__main__":
    app.run(main)

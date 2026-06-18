import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
GYM_ENV_DIR = os.path.join(PROJECT_ROOT, "gym_env")
if GYM_ENV_DIR not in sys.path:
    sys.path.insert(0, GYM_ENV_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from PyQt5 import QtCore
from stable_baselines3 import TD3, SAC, PPO
import numpy as np
import gym_env
import gym
import math
import argparse
import cv2
from tqdm import tqdm

if __package__:
    from .config_loader import read_required_config
else:
    from config_loader import read_required_config

if __package__:
    from .sequence_gpide import is_sequence_gpide_enabled, load_sequence_agent
else:
    from sequence_gpide import is_sequence_gpide_enabled, load_sequence_agent

if __package__:
    from .cpo_agent import CPOAgent, PPOLagrangianAgent
else:
    from cpo_agent import CPOAgent, PPOLagrangianAgent


def rule_based_policy(obs):
    '''
    custom linear policy
    used for LGMD compare
    '''
    action = 0
    # 将obs从1~-1转换成0~1
    obs = np.squeeze(obs, axis=0)

    for i in range(5):
        obs[i] = obs[i]/2 + 0.5

    # obs_weight_depth = np.array([1.0, 3.0, 5.0, -3.0, -1.0, 3.0])
    obs_weight = np.array([1.0, 3.0, 3.0, -3.0, -1.0, 3.0])
    action = obs * obs_weight

    action_sum = np.sum(action)

    if action_sum > math.radians(40):
        action_sum = math.radians(40)
    elif action_sum < -math.radians(40):
        action_sum = -math.radians(40)

    return np.array([action_sum])


class EvaluateThread(QtCore.QThread):
    # signals
    def __init__(self, eval_path, config, model_file, eval_ep_num, eval_env=None, eval_dynamics=None):
        super(EvaluateThread, self).__init__()
        print("init training thread")

        # config
        self.cfg, self.config_file = read_required_config(config)

        # change eval_env and eval_dynamics if is not None
        if eval_env is not None:
            self.cfg.set('options', 'env_name', eval_env)

        if eval_env == 'NH_center':
            self.cfg.set('environment', 'accept_radius', str(1))

        if eval_dynamics is not None:
            self.cfg.set('options', 'dynamic_name', eval_dynamics)


        base_env = gym.make('airsim-env-v0')
        base_env.set_config(self.cfg)
        if is_sequence_gpide_enabled(self.cfg):
            self.env = base_env
            if getattr(self.env, "generate_q_map", False):
                self.env.generate_q_map = False
        else:
            self.env = base_env

        self.eval_path = eval_path
        self.model_file = model_file
        self.eval_ep_num = eval_ep_num
        self.eval_env = self.cfg.get('options', 'env_name')
        self.eval_dynamics = self.cfg.get('options', 'dynamic_name')

    def terminate(self):
        print('Evaluation terminated')

    def run(self):
        # self.run_rule_policy()
        return self.run_drl_model()

    def run_drl_model(self):
        print('start evaluation')
        algo = self.cfg.get('options', 'algo')
        algo_key = algo.strip().upper().replace('_', '-')
        sequence_gpide = is_sequence_gpide_enabled(self.cfg)
        if sequence_gpide:
            model = load_sequence_agent(self.model_file, self.cfg, self.env)
        elif algo == 'TD3':
            model = TD3.load(self.model_file, env=self.env)
        elif algo == 'SAC':
            model = SAC.load(self.model_file, env=self.env)
        elif algo == 'PPO':
            model = PPO.load(self.model_file, env=self.env)
        elif algo_key == 'CPO':
            model = CPOAgent.load(self.model_file, self.env, self.cfg)
        elif algo_key == 'PPO-LAGRANGIAN':
            model = PPOLagrangianAgent.load(self.model_file, self.env, self.cfg)
        else:
            raise Exception('algo set error {}'.format(algo))
        self.env.model = model

        obs = self.env.reset()
        if sequence_gpide:
            model.reset_history(obs)
        episode_num = 0
        time_step = 0
        reward_sum = np.array([.0])
        episode_successes = []
        episode_crashes = []
        traj_list_all = []
        action_list_all = []
        state_list_all = []
        obs_list_all = []

        traj_list = []
        action_list = []
        state_raw_list = []
        step_num_list = []
        obs_list = []
        cv2.waitKey(1)

        while episode_num < self.eval_ep_num:
            if sequence_gpide:
                unscaled_action = model.select_action(deterministic=True)
            else:
                unscaled_action, _ = model.predict(obs, deterministic=True)
            time_step += 1

            new_obs, reward, done, info, = self.env.step(unscaled_action)
            if sequence_gpide:
                model.observe(unscaled_action, reward, new_obs)
            pose = self.env.dynamic_model.get_position()
            traj_list.append(pose)
            action_list.append(unscaled_action)
            state_raw_list.append(self.env.dynamic_model.state_raw)
            obs_list.append(obs)

            obs = new_obs
            reward_sum[-1] += reward

            if done:
                episode_num += 1
                maybe_is_success = info.get('is_success')
                maybe_is_crash = info.get('is_crash')
                print('episode: ', episode_num, ' reward:', reward_sum[-1],
                      'success:', maybe_is_success)
                episode_successes.append(float(maybe_is_success))
                episode_crashes.append(float(maybe_is_crash))
                reward_sum = np.append(reward_sum, .0)
                obs = self.env.reset()
                if sequence_gpide:
                    model.reset_history(obs)
                if info.get('is_success'):
                    traj_list.append(1)
                    action_list.append(1)
                    step_num_list.append(info.get('step_num'))
                elif info.get('is_crash'):
                    traj_list.append(2)
                    action_list.append(2)
                else:
                    traj_list.append(3)
                    action_list.append(3)
                # traj_list.append(info)
                traj_list_all.append(traj_list)
                action_list_all.append(action_list)
                state_list_all.append(state_raw_list)
                obs_list_all.append(obs_list)
                traj_list = []
                action_list = []
                state_raw_list = []
                obs_list = []

        # save trajectory data in eval folder
        eval_folder = self.eval_path + '/eval_{}_{}_{}'.format(self.eval_ep_num, self.eval_env, self.eval_dynamics)
        os.makedirs(eval_folder, exist_ok=True)
        np.save(eval_folder + '/traj_eval',
                np.array(traj_list_all, dtype=object))
        np.save(eval_folder + '/action_eval',
                np.array(action_list_all, dtype=object))
        np.save(eval_folder + '/state_eval',
                np.array(state_list_all, dtype=object))
        np.save(eval_folder + '/obs_eval',
                np.array(obs_list_all, dtype=object))

        average_success_step_num = np.mean(step_num_list) if step_num_list else 0.0

        print('Average episode reward: ', reward_sum[:self.eval_ep_num].mean(),
              'Success rate:', np.mean(episode_successes),
              'Crash rate: ', np.mean(episode_crashes),
              'average success step num: ', average_success_step_num)
        
        results = [
            reward_sum[:self.eval_ep_num].mean(),
            np.mean(episode_successes),
            np.mean(episode_crashes),
            average_success_step_num,
        ]
        
        print(results)
        np.save(eval_folder + '/results', np.array(results))
        
        return results

    def run_rule_policy(self):
        obs = self.env.reset()
        episode_num = 0
        time_step = 0
        reward_sum = np.array([.0])
        while episode_num < self.eval_ep_num:
            unscaled_action = rule_based_policy(obs)
            time_step += 1
            new_obs, reward, done, info, = self.env.step(unscaled_action)
            reward_sum[-1] += reward

            obs = new_obs
            if done:
                episode_num += 1
                maybe_is_success = info.get('is_success')
                print('episode: ', episode_num, ' reward:', reward_sum[-1],
                      'success:', maybe_is_success)
                reward_sum = np.append(reward_sum, .0)
                obs = self.env.reset()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained UAV navigation model.")
    parser.add_argument("--eval_path", required=True, help="Training log folder to store evaluation outputs.")
    parser.add_argument("--config", default=None, help="Path to config.ini. Defaults to eval_path/config/config.ini.")
    parser.add_argument("--model_file", default=None, help="Path to model file.")
    parser.add_argument("--eval_ep_num", type=int, default=50)
    parser.add_argument("--eval_env", default=None)
    parser.add_argument("--eval_dynamics", default=None)
    args = parser.parse_args()

    eval_path = args.eval_path
    config_file = args.config or os.path.join(eval_path, 'config', 'config.ini')
    model_file = args.model_file or os.path.join(eval_path, 'models', 'model_sb3.zip')
    eval_ep_num = args.eval_ep_num
    evaluate_thread = EvaluateThread(eval_path, config_file, model_file,
                                     eval_ep_num, args.eval_env, args.eval_dynamics)
    evaluate_thread.run()


def run_eval_multi():
    # run evaluation for multi models
    eval_logs_name = 'Maze'
    eval_logs_path = 'logs_eval/' + eval_logs_name
    eval_ep_num = 50
    eval_env_name = 'NH_center'        # 1-Trees 2-SimpleAvoid 3-NH_center
    eval_dynamic_name = 'SimpleMultirotor'  # 1-SimpleMultirotor or Multirotor

    model_list = []
    for train_name in os.listdir(eval_logs_path):
        for repeat_name in os.listdir(eval_logs_path + '/' + train_name):
            model_path = eval_logs_path + '/' + train_name + '/' + repeat_name
            model_list.append(model_path)
            # print(model_path)

    # evaluate model according to model path
    eval_num = len(model_list)
    results_list = []

    for i in tqdm(range(eval_num)):
        eval_path = model_list[i]
        config_file = eval_path + '/config/config.ini'
        model_file = eval_path + '/models/model_sb3.zip'

        print(i, eval_path)
        evaluate_thread = EvaluateThread(eval_path, config_file, model_file, eval_ep_num, eval_env_name, eval_dynamic_name)
        results = evaluate_thread.run()
        results_list.append(results)

        del evaluate_thread

    # save all results in a numpy file
    print(results_list)
    np.save('logs_eval/results/eval_{}_{}_{}_{}'.format(eval_ep_num, eval_logs_name, eval_env_name, eval_dynamic_name), np.array(results_list))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')

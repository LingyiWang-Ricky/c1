import datetime
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

import gym
import gym_env
import numpy as np
from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise
from wandb.integration.sb3 import WandbCallback
import wandb
from PyQt5 import QtCore
import argparse
import ast
import torch as th

if __package__:
    from .config_loader import read_required_config
else:
    from config_loader import read_required_config

if __package__:
    from .custom_policy_sb3 import CNN_FC, CNN_GAP, CNN_GAP_BN, No_CNN, CNN_MobileNet, CNN_GAP_new
else:
    from custom_policy_sb3 import CNN_FC, CNN_GAP, CNN_GAP_BN, No_CNN, CNN_MobileNet, CNN_GAP_new

if __package__:
    from .sequence_gpide import build_sequence_agent, is_sequence_gpide_enabled
else:
    from sequence_gpide import build_sequence_agent, is_sequence_gpide_enabled


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_parser():
    parser = argparse.ArgumentParser(
        description="Training thread without plot")
    parser.add_argument(
        '-c',
        '--config',
        help='config file name in configs folder, such as config_default',
        default='config_Trees_SimpleMultirotor_2D')
    parser.add_argument('-n',
                        '--note',
                        help='training objective',
                        default='depth_upper_split_5')

    return parser


class TrainingThread(QtCore.QThread):
    """
    QT thread for policy training
    """

    def __init__(self, config):
        super(TrainingThread, self).__init__()
        print("init training thread")

        # config
        self.cfg, self.config_file = read_required_config(config)

        env_name = self.cfg.get('options', 'env_name')
        self.project_name = env_name

        # make gym environment
        base_env = gym.make('airsim-env-v0')
        base_env.set_config(self.cfg)
        if is_sequence_gpide_enabled(self.cfg):
            self.env = base_env
            if getattr(self.env, "generate_q_map", False):
                print("[GPIDE] sequence agents do not support q_map, disabling generate_q_map.")
                self.env.generate_q_map = False
                self.cfg.set('options', 'generate_q_map', 'False')
        else:
            self.env = base_env

        wandb_name = self.cfg.get(
            'options', 'policy_name') + '-' + self.cfg.get('options', 'algo')
        if self.cfg.get('options', 'dynamic_name') == 'SimpleFixedwing':
            if self.cfg.get('options', 'perception') == "lgmd":
                wandb_name += '-LGMD'
            else:
                wandb_name += '-depth'
            if self.cfg.getfloat('fixedwing', 'pitch_flap_hz') != 0:
                wandb_name += '-Flapping'

        # wandb
        if self.cfg.getboolean('options', 'use_wandb'):
            wandb.init(
                project=self.project_name,
                notes=self.cfg.get('wandb', 'notes'),
                name=self.cfg.get('wandb', 'name'),
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                save_code=True,  # optional
            )

    def terminate(self):
        print('TrainingThread terminated')

    def run(self):
        print("run training thread")

        # ! -----------------------------------init folders-----------------------------------------
        now = datetime.datetime.now()
        now_string = now.strftime('%Y_%m_%d_%H_%M')
        file_path = 'logs/' + self.project_name + '/' + now_string + '_' + self.cfg.get(
            'options', 'dynamic_name') + '_' + self.cfg.get(
                'options', 'policy_name') + '_' + self.cfg.get(
                    'options', 'algo')
        log_path = file_path + '/tb_logs'
        model_path = file_path + '/models'
        config_path = file_path + '/config'
        data_path = file_path + '/data'
        console_path = file_path + '/console'
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(config_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)  # create data path to save q_map
        os.makedirs(console_path, exist_ok=True)

        stdout_log = open(os.path.join(console_path, 'stdout.log'), 'a', encoding='utf-8')
        stderr_log = open(os.path.join(console_path, 'stderr.log'), 'a', encoding='utf-8')
        sys.stdout = Tee(sys.__stdout__, stdout_log)
        sys.stderr = Tee(sys.__stderr__, stderr_log)
        print('training_run_dir={}'.format(file_path), flush=True)
        print('console_log_dir={}'.format(console_path), flush=True)

        # save config file
        with open(os.path.join(config_path, 'config.ini'), 'w') as configfile:
            self.cfg.write(configfile)

        if is_sequence_gpide_enabled(self.cfg):
            algo = self.cfg.get('options', 'algo')
            print('algo: ', algo)
            print('ablation_mode: ', self.cfg.get('options', 'ablation_mode', fallback='legacy'))
            print('temporal_encoder: GPIDE sequence')
            model = build_sequence_agent(self.cfg, self.env)
            self.env.model = model
            self.env.data_path = data_path
            total_timesteps = self.cfg.getint('options', 'total_timesteps')
            checkpoint_interval = self.cfg.getint('DRL', 'checkpoint_interval', fallback=2500)
            max_episodes = self.cfg.getint('options', 'max_episodes', fallback=0)
            min_ent_coef = self.cfg.getfloat('DRL', 'min_ent_coef', fallback=0.0)
            csv_path = os.path.join(data_path, 'training_log.csv')
            episode_csv_path = os.path.join(data_path, 'episode_log.csv')
            metrics_json_path = os.path.join(data_path, 'metrics_summary.json')
            model.learn(
                total_timesteps,
                checkpoint_dir=model_path,
                checkpoint_interval=checkpoint_interval,
                csv_path=csv_path,
                tensorboard_log=log_path,
                max_episodes=max_episodes,
                min_ent_coef=min_ent_coef,
                episode_csv_path=episode_csv_path,
                metrics_json_path=metrics_json_path,
            )
            model_name = 'model_sequence_gpide.pt'
            model.save(os.path.join(model_path, model_name), total_step=model.total_steps)
            print('training finished')
            print('model saved to: {}'.format(model_path))
            print('training data saved to: {}'.format(data_path))
            return

        #! -----------------------------------policy selection-------------------------------------
        feature_num_state = self.env.dynamic_model.state_feature_length
        feature_num_cnn = self.cfg.getint('options', 'cnn_feature_num')
        policy_name = self.cfg.get('options', 'policy_name')

        # feature extraction network
        if self.cfg.get('options', 'activation_function') == 'tanh':
            activation_function = th.nn.Tanh
        else:
            activation_function = th.nn.ReLU
        
        if policy_name == 'mlp':
            policy_base = 'MlpPolicy'
            policy_kwargs = dict(activation_fn=activation_function)
        else:
            policy_base = 'CnnPolicy'
            if policy_name == 'CNN_FC':
                policy_used = CNN_FC
            elif policy_name == 'CNN_GAP':
                policy_used = CNN_GAP_new
            elif policy_name == 'CNN_GAP_BN':
                policy_used = CNN_GAP_BN
            elif policy_name == 'CNN_MobileNet':
                policy_used = CNN_MobileNet
            elif policy_name == 'No_CNN':
                policy_used = No_CNN
            else:
                raise Exception('policy select error: ', policy_name)

            policy_kwargs = dict(
                features_extractor_class=policy_used,
                features_extractor_kwargs=dict(
                    features_dim=feature_num_state + feature_num_cnn,
                    state_feature_dim=feature_num_state),
                activation_fn=activation_function)

        # fully-connected work after feature extraction
        net_arch_list = ast.literal_eval(self.cfg.get("options", "net_arch"))
        policy_kwargs['net_arch'] = net_arch_list

        #! ---------------------------------algorithm selection-------------------------------------
        algo = self.cfg.get('options', 'algo')
        print('algo: ', algo)
        print('ablation_mode: ', self.cfg.get('options', 'ablation_mode', fallback='legacy'))
        if algo == 'PPO':
            model = PPO(
                policy_base,
                self.env,
                # n_steps = 200,
                learning_rate=self.cfg.getfloat('DRL', 'learning_rate'),
                policy_kwargs=policy_kwargs,
                tensorboard_log=log_path,
                seed=0,
                verbose=2)
        elif algo == 'SAC':
            n_actions = self.env.action_space.shape[-1]
            noise_sigma = self.cfg.getfloat(
                'DRL', 'action_noise_sigma') * np.ones(n_actions)
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=noise_sigma)
            model = SAC(
                policy_base,
                self.env,
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
                buffer_size=self.cfg.getint('DRL', 'buffer_size'),
                gamma=self.cfg.getfloat('DRL', 'gamma'),
                learning_starts=self.cfg.getint('DRL', 'learning_starts'),
                learning_rate=self.cfg.getfloat('DRL', 'learning_rate'),
                batch_size=self.cfg.getint('DRL', 'batch_size'),
                train_freq=(self.cfg.getint('DRL', 'train_freq'), 'step'),
                gradient_steps=self.cfg.getint('DRL', 'gradient_steps'),
                tensorboard_log=log_path,
                seed=0,
                verbose=2)
        elif algo == 'TD3':
            # The noise objects for TD3
            n_actions = self.env.action_space.shape[-1]
            noise_sigma = self.cfg.getfloat(
                'DRL', 'action_noise_sigma') * np.ones(n_actions)
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=noise_sigma)
            model = TD3(
                policy_base,
                self.env,
                action_noise=action_noise,
                learning_rate=self.cfg.getfloat('DRL', 'learning_rate'),
                gamma=self.cfg.getfloat('DRL', 'gamma'),
                policy_kwargs=policy_kwargs,
                learning_starts=self.cfg.getint('DRL', 'learning_starts'),
                batch_size=self.cfg.getint('DRL', 'batch_size'),
                train_freq=(self.cfg.getint('DRL', 'train_freq'), 'step'),
                gradient_steps=self.cfg.getint('DRL', 'gradient_steps'),
                buffer_size=self.cfg.getint('DRL', 'buffer_size'),
                tensorboard_log=log_path,
                seed=0,
                verbose=2)
        else:
            raise Exception('Invalid algo name : ', algo)

        # TODO create eval_callback
        # eval_freq = self.cfg.getint('TD3', 'eval_freq')
        # n_eval_episodes = self.cfg.getint('TD3', 'n_eval_episodes')
        # eval_callback = EvalCallback(self.env, best_model_save_path= file_path + '/eval',
        #                      log_path= file_path + '/eval', eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
        #                      deterministic=True, render=False)

        #! -------------------------------------train-----------------------------------------
        print('start training model')
        total_timesteps = self.cfg.getint('options', 'total_timesteps')
        self.env.model = model
        self.env.data_path = data_path

        if self.cfg.getboolean('options', 'use_wandb'):
            # if algo == 'TD3' or algo == 'SAC':
            #     wandb.watch(model.actor, log_freq=100, log="all")  # log gradients
            # elif algo == 'PPO':
            #     wandb.watch(model.policy, log_freq=100, log="all")
            model.learn(
                total_timesteps,
                log_interval=1,
                callback=WandbCallback(
                    model_save_freq=10000,
                    gradient_save_freq=5000,
                    model_save_path=model_path,
                    verbose=2,
                )
            )
        else:
            model.learn(total_timesteps)

        #! ---------------------------model save----------------------------------------------------
        model_name = 'model_sb3'
        model.save(model_path + '/' + model_name)

        print('training finished')
        print('model saved to: {}'.format(model_path))


def main():
    parser = get_parser()
    args = parser.parse_args()

    config_file = 'configs/' + args.config + '.ini'

    print(config_file)

    training_thread = TrainingThread(config_file)
    training_thread.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')

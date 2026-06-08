import gym
from gym import spaces
import airsim
from configparser import NoOptionError, NoSectionError
import keyboard

import torch as th
import numpy as np
import math
import cv2

from .dynamics.multirotor_simple import MultirotorDynamicsSimple
from .dynamics.multirotor_airsim import MultirotorDynamicsAirsim
from .dynamics.fixedwing_simple import FixedwingDynamicsSimple
# from .lgmd.LGMD import LGMD

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal


class AirsimGymEnv(gym.Env, QtCore.QThread):
    # pyqt signal for visualization
    action_signal = pyqtSignal(int, np.ndarray)
    state_signal = pyqtSignal(int, np.ndarray)
    attitude_signal = pyqtSignal(int, np.ndarray, np.ndarray)
    reward_signal = pyqtSignal(int, float, float)
    pose_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    lgmd_signal = pyqtSignal(float, float, np.ndarray)

    def __init__(self) -> None:
        super().__init__()
        np.set_printoptions(formatter={'float': '{: 4.2f}'.format},
                            suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)
        print("init airsim-gym-env.")
        self.model = None
        self.data_path = None
        self.lgmd = None

    def set_config(self, cfg):
        """get config from .ini file
        """
        self.cfg = cfg
        self.env_name = cfg.get('options', 'env_name')
        self.dynamic_name = cfg.get('options', 'dynamic_name')
        self.keyboard_debug = cfg.getboolean('options', 'keyboard_debug')
        self.generate_q_map = cfg.getboolean('options', 'generate_q_map')
        self.print_train_info = cfg.getboolean('options', 'print_train_info', fallback=True)
        self.perception_type = cfg.get('options', 'perception')
        self.goal_curriculum_enabled = cfg.getboolean('curriculum', 'enabled', fallback=False)
        self._goal_curriculum_base_rect = None
        self._goal_curriculum_base_distance = None
        self._goal_curriculum_base_z_offset_range = None
        self._goal_curriculum_base_z_offset_min = None

        # create LGMD agent
        if self.perception_type == 'lgmd':
            self.lgmd = LGMD(type='origin',  p_threshold=50, s_threshold=0, Ki=2, i_layer_size=3, activate_coeff=1, use_on_off=True)
            self.split_out_last = np.array([0, 0, 0, 0, 0])

        print('Environment: ', self.env_name, "Dynamics: ", self.dynamic_name,
              'Perception: ', self.perception_type)

        # set dynamics
        if self.dynamic_name == 'SimpleFixedwing':
            self.dynamic_model = FixedwingDynamicsSimple(cfg)
        elif self.dynamic_name == 'SimpleMultirotor':
            self.dynamic_model = MultirotorDynamicsSimple(cfg)
        elif self.dynamic_name == 'Multirotor':
            self.dynamic_model = MultirotorDynamicsAirsim(cfg)
        else:
            raise Exception("Invalid dynamic_name!", self.dynamic_name)

        # set start and goal position according to different environment
        if self.env_name == 'NH_center':
            start_position = [0, 0, 5]
            goal_rect = [-128, -128, 128, 128]  # rectangular goal pose
            goal_distance = 90
            self.dynamic_model.set_start(
                start_position, random_angle=math.pi*2)
            self.dynamic_model.set_goal(random_angle=math.pi*2, rect=goal_rect)
            self._goal_curriculum_base_rect = list(goal_rect)
            self.work_space_x = [-140, 140]
            self.work_space_y = [-140, 140]
            self.work_space_z = [0.5, 20]
            self.max_episode_steps = 1000
        elif self.env_name == 'NH_tree':
            start_position = [110, 180, 5]
            goal_distance = 90
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model.set_goal(distance=90, random_angle=0)
            self.work_space_x = [start_position[0],
                                 start_position[0] + goal_distance + 10]
            self.work_space_y = [
                start_position[1] - 30, start_position[1] + 30]
            self.work_space_z = [0.5, 10]
            self.max_episode_steps = 400
        elif self.env_name == 'City':
            start_position = [40, -30, 40]
            goal_position = [280, -200, 40]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 350]
            self.work_space_y = [-300, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 400
        elif self.env_name in ('City_400', 'City_400_400'):
            start_position = [0, 0, 50]
            goal_rect = [-220, -220, 220, 220]
            self.dynamic_model.set_start(start_position, random_angle=math.pi*2)
            if hasattr(self.dynamic_model, '_set_goal_pose_single'):
                # Fixed-wing dynamics randomize the start/goal rectangle at reset.
                self.dynamic_model._set_goal_pose_single([200, -200, 50])
            else:
                # Multirotor dynamics randomize the goal on the City_400 boundary.
                self.dynamic_model.set_goal(rect=goal_rect, random_angle=math.pi*2)
                self._goal_curriculum_base_rect = list(goal_rect)
            self.work_space_x = [-220, 220]
            self.work_space_y = [-220, 220]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 800
        elif self.env_name == 'Tree_200':
            # note: the start and end points will be covered by
            # update_start_and_goal_pose_random function
            start_position = [0, 0, 8]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 100]
            self.work_space_y = [-100, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 600
        elif self.env_name == 'SimpleAvoid':
            start_position = [0, 0, 5]
            goal_distance = 50
            simple_avoid_angle_min_pi = self._cfg_getfloat(
                'environment', 'simple_avoid_angle_min_pi', 0.0)
            simple_avoid_angle_max_pi = self._cfg_getfloat(
                'environment', 'simple_avoid_angle_max_pi',
                self._cfg_getfloat('environment', 'simple_avoid_random_angle_pi', 2.0))
            simple_avoid_angle_offset = math.pi * simple_avoid_angle_min_pi
            simple_avoid_random_angle = math.pi * max(
                simple_avoid_angle_max_pi - simple_avoid_angle_min_pi, 0.0)
            self.dynamic_model.set_start(
                start_position, random_angle=simple_avoid_random_angle,
                angle_offset=simple_avoid_angle_offset)
            self.dynamic_model.set_goal(
                distance=goal_distance, random_angle=simple_avoid_random_angle,
                angle_offset=simple_avoid_angle_offset)
            self.work_space_x = [
                start_position[0] - goal_distance - 10, start_position[0] + goal_distance + 10]
            self.work_space_y = [
                start_position[1] - goal_distance - 10, start_position[1] + goal_distance + 10]
            self.work_space_z = [0.5, 50]
            self.max_episode_steps = 400
        elif self.env_name == 'Forest':
            start_position = [0, 0, 10]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 100]
            self.work_space_y = [-100, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 300
        elif self.env_name == 'Trees':
            start_position = [0, 0, 5]
            goal_distance = 70
            self.dynamic_model.set_start(
                start_position, random_angle=math.pi*2)
            self.dynamic_model.set_goal(
                distance=goal_distance, random_angle=math.pi*2)
            self.work_space_x = [
                start_position[0] - goal_distance - 10, start_position[0] + goal_distance + 10]
            self.work_space_y = [
                start_position[1] - goal_distance - 10, start_position[1] + goal_distance + 10]
            self.work_space_z = [0.5, 50]
            self.max_episode_steps = 500
        else:
            raise Exception("Invalid env_name!", self.env_name)

        self.client = self.dynamic_model.client
        self.state_feature_length = self.dynamic_model.state_feature_length
        self.cnn_feature_length = self.cfg.getint('options', 'cnn_feature_num')
        self.vector_feature_length = 5

        # training state
        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = 0

        # other settings
        self.crash_distance = cfg.getint('environment', 'crash_distance')
        self.accept_radius = cfg.getint('environment', 'accept_radius')

        self.max_depth_meters = cfg.getint('environment', 'max_depth_meters')
        self.screen_height = cfg.getint('environment', 'screen_height')
        self.screen_width = cfg.getint('environment', 'screen_width')
        self.image_retry_count = cfg.getint('environment', 'image_retry_count', fallback=3)
        self.last_depth_image = None
        self.last_gray_image = None

        self.trajectory_list = []

        # observation space vector or image
        if self.perception_type == 'vector' or self.perception_type == 'lgmd':
            self.observation_space = spaces.Box(low=0, high=1,
                                                shape=(1,
                                                       self.vector_feature_length + self.state_feature_length),
                                                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.screen_height,
                                                       self.screen_width, 2),
                                                dtype=np.uint8)

        self.action_space = self.dynamic_model.action_space
        self.max_episode_steps = cfg.getint(
            'environment', 'max_episode_steps', fallback=self.max_episode_steps)
        self._goal_curriculum_base_distance = getattr(self.dynamic_model, 'goal_distance', None)
        if hasattr(self.dynamic_model, 'goal_z_offset_range'):
            self._goal_curriculum_base_z_offset_range = self.dynamic_model.goal_z_offset_range
            self._goal_curriculum_base_z_offset_min = self.dynamic_model.goal_z_offset_min

        self.reward_type = None
        try:
            self.reward_type = cfg.get('options', 'reward_type')
            print('Reward type: ', self.reward_type)
        except NoOptionError:
            self.reward_type = None

    def reset(self):
        # reset state.  Apply optional curriculum before dynamics.reset(),
        # because reset() samples the next goal pose from the current goal
        # rectangle/distance and vertical offset settings.
        self._apply_goal_curriculum()
        self.dynamic_model.reset()
        self._current_step_info = None
        self._position_before_action = None
        self._goal_capture_info = {}
        self._obstacle_shield_info = {}
        self._obstacle_avoid_turn_sign = 0.0
        self._obstacle_avoid_turn_steps = 0
        self._boundary_shield_info = {}

        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        initial_distance = self.get_distance_to_goal_3d() if getattr(self.dynamic_model, 'navigation_3d', False) \
            else self.dynamic_model.get_distance_to_goal_2d()
        self.dynamic_model.goal_distance = max(float(initial_distance), 1e-6)
        self.previous_distance_from_des_point = float(initial_distance)
        self.previous_vertical_error = abs(float(self.dynamic_model.goal_position[2] - self.dynamic_model.get_position()[2])) \
            if getattr(self.dynamic_model, 'navigation_3d', False) else 0.0
        self.best_distance_to_goal = float(initial_distance)
        self.no_progress_steps = 0
        self._last_stuck_check_step = -1

        self.trajectory_list = []

        obs = self.get_obs()

        return obs


    def _apply_goal_curriculum(self):
        """Gradually expand goal difficulty for sparse AirSim maps.

        NH_center 3D starts with random yaw, random goal direction and altitude
        changes in a cluttered map.  Jumping immediately to the full 256 m goal
        square makes early replay dominated by crash/outside/no-progress samples.
        This optional curriculum shrinks the goal rectangle/distance and vertical
        offset at the beginning, then restores the configured final task after
        the requested number of episodes.
        """
        if not getattr(self, 'goal_curriculum_enabled', False):
            return

        episodes = max(1, self.cfg.getint('curriculum', 'episodes', fallback=500))
        start_ratio = float(np.clip(
            self._cfg_getfloat('curriculum', 'start_ratio', 0.35), 0.0, 1.0))
        progress = float(np.clip(getattr(self, 'episode_num', 0) / episodes, 0.0, 1.0))
        ratio = start_ratio + (1.0 - start_ratio) * progress

        if self._goal_curriculum_base_rect is not None and getattr(self.dynamic_model, 'goal_rect', None) is not None:
            rect = np.asarray(self._goal_curriculum_base_rect, dtype=np.float32)
            min_extent = self._cfg_getfloat('curriculum', 'min_goal_extent', 35.0)
            scaled = rect * ratio
            for idx, sign in enumerate(np.sign(rect)):
                if sign != 0:
                    scaled[idx] = sign * max(abs(float(scaled[idx])), min_extent)
            self.dynamic_model.goal_rect = scaled.astype(float).tolist()

        if self._goal_curriculum_base_distance is not None and getattr(self.dynamic_model, 'goal_rect', None) is None:
            min_distance = self._cfg_getfloat('curriculum', 'min_goal_distance', 35.0)
            self.dynamic_model.goal_distance = max(
                min_distance, float(self._goal_curriculum_base_distance) * ratio)

        if getattr(self.dynamic_model, 'navigation_3d', False) and \
                self._goal_curriculum_base_z_offset_range is not None:
            z_start_ratio = float(np.clip(
                self._cfg_getfloat('curriculum', 'start_z_ratio', start_ratio), 0.0, 1.0))
            z_ratio = z_start_ratio + (1.0 - z_start_ratio) * progress
            min_z_range = self._cfg_getfloat('curriculum', 'min_goal_z_offset_range', 0.5)
            final_z_range = float(self._goal_curriculum_base_z_offset_range)
            if final_z_range > 0:
                self.dynamic_model.goal_z_offset_range = max(min_z_range, final_z_range * z_ratio)
                base_z_min = float(self._goal_curriculum_base_z_offset_min or 0.0)
                self.dynamic_model.goal_z_offset_min = min(
                    base_z_min, self.dynamic_model.goal_z_offset_range)

    def step(self, action):
        self._position_before_action = self.dynamic_model.get_position()
        action = self.apply_goal_capture_shield(action)
        action = self.apply_obstacle_shield(action)
        action = self.apply_boundary_shield(action)
        # set action
        if self.dynamic_name == 'SimpleFixedwing':
            # add step to calculate pitch flap deg Fixed wing only
            self.dynamic_model.set_action(action, self.step_num)
        else:
            self.dynamic_model.set_action(action)

        position_ue4 = self.dynamic_model.get_position()
        self.trajectory_list.append(position_ue4)

        # get new obs
        obs = self.get_obs()
        info = self.get_done_info()
        done = info['done']
        self._current_step_info = info
        self._add_constraint_info(info, action)
        if done:
            print(info)

        # ----------------compute reward---------------------------
        if self.dynamic_name == 'SimpleFixedwing':
            # reward = self.compute_reward_fixedwing(done, action)
            reward = self.compute_reward_final_fixedwing(done, action)
        elif self.reward_type == 'reward_with_action':
            reward = self.compute_reward_with_action(done, action)
        elif self.reward_type == 'reward_new':
            reward = self.compute_reward_multirotor_new(done, action)
        elif self.reward_type == 'reward_lqr':
            reward = self.compute_reward_lqr(done, action)
        elif self.reward_type == 'reward_final':
            reward = self.compute_reward_final(done, action)
        else:
            reward = self.compute_reward(done, action)

        self.cumulated_episode_reward += reward

        # ----------------print info---------------------------
        if self.print_train_info:
            self.print_train_info_airsim(action, obs, reward, info)

        if self.cfg.get('options', 'dynamic_name') == 'SimpleFixedwing':
            self.set_pyqt_signal_fixedwing(action, reward, done)
        else:
            self.set_pyqt_signal_multirotor(action, reward)

        if self.keyboard_debug:
            action_copy = np.copy(action)
            action_copy[-1] = math.degrees(action_copy[-1])
            state_copy = np.copy(self.dynamic_model.state_raw)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print(
                '=============================================================================')
            print('episode', self.episode_num, 'step',
                  self.step_num, 'total step', self.total_step)
            print('action', action_copy)
            print('state', state_copy)
            print('state_norm', self.dynamic_model.state_norm)
            print('reward {:.3f} {:.3f}'.format(
                reward, self.cumulated_episode_reward))
            print('done', done)
            keyboard.wait('a')

        if self.generate_q_map and (self.cfg.get('options', 'algo') == 'TD3' or self.cfg.get('options', 'algo') == 'SAC'):
            if self.model is not None:
                with th.no_grad():
                    # get q-value for td3
                    obs_copy = obs.copy()
                    if self.perception_type != 'vector':
                        obs_copy = obs_copy.swapaxes(0, 1)
                        obs_copy = obs_copy.swapaxes(0, 2)
                    q_value_current = self.model.critic(th.from_numpy(obs_copy[tuple(
                        [None])]).float().cuda(), th.from_numpy(action[None]).float().cuda())
                    q_1 = q_value_current[0].cpu().numpy()[0]
                    q_2 = q_value_current[1].cpu().numpy()[0]

                    q_value = min(q_1, q_2)[0]

                    self.visual_log_q_value(q_value, action, reward)

        self.step_num += 1
        self.total_step += 1
        self._current_step_info = None

        return obs, reward, done, info

# ! -------------------------get obs------------------------------------------
    def get_obs(self):
        if self.perception_type == 'vector':
            obs = self.get_obs_vector()
        elif self.perception_type == 'lgmd':
            obs = self.get_obs_lgmd()
        else:
            obs = self.get_obs_image()

        return obs

    def get_obs_image(self):
        # Normal mode: get depth image then transfer to matrix with state
        # 1. get current depth image and transfer to 0-255  0-20m 255-0m
        image = self.get_depth_image()  # 0-6550400.0 float 32
        image_resize = cv2.resize(image, (self.screen_width,
                                          self.screen_height))
        self.min_distance_to_obstacles = image.min()
        # switch 0 and 255
        image_scaled = np.clip(
            image_resize, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        # 2. get current state (relative_pose, velocity)
        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = self.dynamic_model._get_state_feature()
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        # 3. generate image with state
        image_with_state = np.array([image_uint8, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)

        return image_with_state

    @staticmethod
    def _valid_image_response(response):
        return response is not None and response.width > 0 and response.height > 0

    def _sim_get_images_with_retry(self, requests, min_responses=1):
        """Call AirSim image API with bounded retries.

        AirSim can occasionally return empty images or raise an RPCError such as
        ``bad cast`` during long training runs.  Keep the request payload as a
        list on every attempt and return ``None`` after the configured retry
        budget so callers can fall back to the last valid frame instead of
        crashing the trainer.
        """
        attempts = max(1, int(getattr(self, 'image_retry_count', 3)))
        last_error = None
        for attempt in range(attempts):
            try:
                responses = self.client.simGetImages(requests)
            except Exception as exc:
                last_error = exc
                responses = None

            if responses and len(responses) >= min_responses and self._valid_image_response(responses[0]):
                return responses
            print("get_image_fail... attempt {}/{}".format(attempt + 1, attempts))

        if last_error is not None:
            print("get_image_fail fallback after AirSim RPC error: {}".format(last_error))
        return None

    def _fallback_depth_image(self):
        if self.last_depth_image is not None:
            return self.last_depth_image.copy()
        return np.full((self.screen_height, self.screen_width), self.max_depth_meters, dtype=np.float32)

    def _fallback_gray_image(self):
        if self.last_gray_image is not None:
            return self.last_gray_image.copy()
        return np.zeros((self.screen_height, self.screen_width), dtype=np.uint8)

    def get_depth_gray_image(self):
        # get depth and rgb image
        # scene vision image in png format
        requests = [
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        ]
        responses = self._sim_get_images_with_retry(requests, min_responses=2)
        if responses is None or not self._valid_image_response(responses[1]):
            return self._fallback_depth_image(), self._fallback_gray_image()

        try:
            # get depth image
            depth_img = airsim.list_to_2d_float_array(
                responses[0].image_data_float,
                responses[0].width, responses[0].height)
            depth_meter = (depth_img * 100).astype(np.float32)

            # get gray image
            img_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
            # reshape array to 3 channel image array H X W X 3
            img_rgb = img_1d.reshape(responses[1].height, responses[1].width, 3)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        except Exception as exc:
            print("get_image_fail fallback after image decode error: {}".format(exc))
            return self._fallback_depth_image(), self._fallback_gray_image()

        self.last_depth_image = depth_meter.copy()
        self.last_gray_image = img_gray.copy()
        return depth_meter, img_gray

    def get_depth_image(self):
        requests = [airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)]
        responses = self._sim_get_images_with_retry(requests, min_responses=1)
        if responses is None:
            return self._fallback_depth_image()

        try:
            depth_img = airsim.list_to_2d_float_array(
                responses[0].image_data_float, responses[0].width,
                responses[0].height)
            depth_meter = (depth_img * 100).astype(np.float32)
        except Exception as exc:
            print("get_image_fail fallback after depth decode error: {}".format(exc))
            return self._fallback_depth_image()

        self.last_depth_image = depth_meter.copy()
        return depth_meter

    def get_obs_vector(self):

        image = self.get_depth_image()  # 0-6550400.0 float 32
        self.min_distance_to_obstacles = image.min()

        image_scaled = np.clip(image, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        image_obs = image_uint8
        split_row = 1
        split_col = self.vector_feature_length

        v_split_list = np.vsplit(image_obs, split_row)

        split_final = []
        for i in range(split_row):
            h_split_list = np.hsplit(v_split_list[i], split_col)
            for j in range(split_col):
                split_final.append(h_split_list[j].max())

        img_feature = np.array(split_final) / 255.0

        state_feature = self.dynamic_model._get_state_feature() / 255

        feature_all = np.concatenate((img_feature, state_feature), axis=0)

        self.feature_all = feature_all

        feature_all = np.reshape(feature_all, (1, len(feature_all)))

        return feature_all

    def get_obs_lgmd(self):
        # get depth and gray image
        depth_meter, img_gray = self.get_depth_gray_image()
        self.min_distance_to_obstacles = depth_meter.min()

        self.lgmd.update(img_gray)

        split_col_num = self.vector_feature_length
        s_layer = self.lgmd.s_layer  # (192, 320)
        s_layer_split = np.hsplit(s_layer, split_col_num)  # (192, 109)

        lgmd_out_list = []
        activate_coeff = 0.5
        for i in range(split_col_num):
            s_layer_activated_sum = abs(np.sum(s_layer_split[i]))
            Kf = -(s_layer_activated_sum * activate_coeff) / (192*64)  # 0 - 1
            a = np.exp(Kf)
            lgmd_out_norm = (1 / (1 + a) - 0.5) * 2
            lgmd_out_list.append(lgmd_out_norm)

        # show iamges
        heatmapshow = None
        heatmapshow = cv2.normalize(s_layer, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        cv2.imshow('gray image', img_gray)
        cv2.imshow('depth image', np.clip(depth_meter, 0, 255)/255)
        cv2.imshow('s-layer', heatmapshow)
        cv2.waitKey(1)

        # update LGMD
        split_final = np.array(lgmd_out_list)
        
        filter_coeff = 0.8
        split_final_filter = filter_coeff * split_final + (1-filter_coeff) * self.split_out_last
        self.split_out_last = split_final_filter

        img_feature = np.array(split_final_filter)

        state_feature = self.dynamic_model._get_state_feature() / 255

        feature_all = np.concatenate((img_feature, state_feature), axis=0)

        self.feature_all = feature_all

        feature_all = np.reshape(feature_all, (1, len(feature_all)))

        return feature_all
# ! ---------------------calculate rewards-------------------------------------

    def compute_reward(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance * \
                500  # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # add yaw_rate cost
            yaw_speed_cost = 0.1 * \
                abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = 0.1 * \
                    ((abs(action[1]) / self.dynamic_model.v_z_max)**2)
                z_err_cost = 0.05 * \
                    ((abs(
                        self.dynamic_model.state_raw[1]) / self.dynamic_model.max_vertical_difference)**2)
                action_cost += (v_z_cost + z_err_cost)

            action_cost += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.1 * abs(yaw_error / 180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_final(self, done, action):
        reward = 0
        reward_reach = self._cfg_getfloat('reward', 'reward_reach', 10)
        reward_crash = self._cfg_getfloat('reward', 'reward_crash', -20)
        reward_outside = self._cfg_getfloat('reward', 'reward_outside', -10)
        reward_timeout = self._cfg_getfloat('reward', 'reward_timeout', 0)
        
        if self.env_name == 'NH_center':
            distance_reward_coef = self._cfg_getfloat('reward', 'distance_reward_coef', 500)
        else:
            distance_reward_coef = self._cfg_getfloat('reward', 'distance_reward_coef', 50)
        pose_penalty_coef = self._cfg_getfloat('reward', 'pose_penalty_coef', 0.1)
        obstacle_penalty_coef = self._cfg_getfloat('reward', 'obstacle_penalty_coef', 0.2)
        action_penalty_coef = self._cfg_getfloat('reward', 'action_penalty_coef', 0.1)
        yaw_penalty_coef = self._cfg_getfloat('reward', 'yaw_penalty_coef', 0.5)
        step_penalty = self._cfg_getfloat('reward', 'step_penalty', 0.0)
        no_progress_penalty = self._cfg_getfloat('reward', 'no_progress_penalty', 0.0)
        progress_epsilon = self._cfg_getfloat('reward', 'progress_epsilon', 0.01)
        reverse_progress_penalty_coef = self._cfg_getfloat('reward', 'reverse_progress_penalty_coef', 0.0)
        min_forward_speed = self._cfg_getfloat('reward', 'min_forward_speed', 0.0)
        low_speed_penalty_coef = self._cfg_getfloat('reward', 'low_speed_penalty_coef', 0.0)
        heading_alignment_coef = self._cfg_getfloat('reward', 'heading_alignment_coef', 0.0)
        heading_error_penalty_coef = self._cfg_getfloat('reward', 'heading_error_penalty_coef', 0.0)
        boundary_penalty_coef = self._cfg_getfloat('reward', 'boundary_penalty_coef', 0.0)
        boundary_safe_margin = self._cfg_getfloat('reward', 'boundary_safe_margin', 5.0)
        path_penalty_coef = self._cfg_getfloat('reward', 'path_penalty_coef', 0.0)
        vertical_reward_coef = self._cfg_getfloat('reward', 'vertical_reward_coef', 0.0)
        vertical_error_penalty_coef = self._cfg_getfloat('reward', 'vertical_error_penalty_coef', 0.0)

        if not done:
            # 1 - goal reward.  Also penalize no-progress steps so a policy cannot
            # avoid collisions by circling or crawling until max_episode_steps.
            distance_now = self.get_distance_to_goal_3d()
            distance_progress = self.previous_distance_from_des_point - distance_now
            reward_distance = distance_reward_coef * distance_progress / \
                self.dynamic_model.goal_distance   # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            # 2 - Position punishment
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x = current_pose[0]
            y = current_pose[1]
            z = current_pose[2]
            x_g = goal_pose[0]
            y_g = goal_pose[1]
            z_g = goal_pose[2]

            punishment_xy = np.clip(self.getDis(
                x, y, 0, 0, x_g, y_g) / 10, 0, 1)
            punishment_z = 0.5 * np.clip(abs(z - z_g)/5, 0, 1)

            punishment_pose = punishment_xy + punishment_z

            if self.min_distance_to_obstacles < 10:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 5, 0, 1)
            else:
                punishment_obs = 0

            punishment_action = 0

            # add yaw_rate cost
            yaw_speed_cost = abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = ((abs(action[1]) / self.dynamic_model.v_z_max)**2)
                z_err_cost = (
                    (abs(self.dynamic_model.state_raw[1]) / self.dynamic_model.max_vertical_difference)**2)
                punishment_action += (v_z_cost + z_err_cost)

            punishment_action += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = abs(yaw_error / 90)
            heading_alignment = max(0.0, math.cos(math.radians(yaw_error)))
            heading_reward = heading_alignment_coef * heading_alignment
            heading_error_penalty = heading_error_penalty_coef * abs(yaw_error / 180.0)
            progress_penalty = no_progress_penalty if distance_progress < progress_epsilon else 0.0
            reverse_progress_penalty = reverse_progress_penalty_coef * max(0.0, -distance_progress)
            forward_speed = float(action[0]) if len(np.asarray(action).reshape(-1)) > 0 else 0.0
            low_speed_penalty = low_speed_penalty_coef * max(0.0, min_forward_speed - forward_speed)
            boundary_margin = self.get_workspace_margin(include_z=False)
            z_boundary_margin = self.get_workspace_z_margin() if self.dynamic_model.navigation_3d else float('inf')
            z_boundary_safe_margin = self._cfg_getfloat('reward', 'z_boundary_safe_margin', 2.0)
            boundary_cost = 1.0 - np.clip(boundary_margin / max(boundary_safe_margin, 1e-6), 0.0, 1.0)
            z_boundary_cost = 0.0
            if self.dynamic_model.navigation_3d:
                z_boundary_cost = 1.0 - np.clip(z_boundary_margin / max(z_boundary_safe_margin, 1e-6), 0.0, 1.0)
            boundary_penalty = boundary_penalty_coef * max(boundary_cost, z_boundary_cost)
            path_distance = self.getDis(x, y, self.dynamic_model.start_position[0], self.dynamic_model.start_position[1], x_g, y_g)
            path_penalty = path_penalty_coef * np.clip(path_distance / 10.0, 0.0, 1.0)
            vertical_error = abs(float(z - z_g)) if self.dynamic_model.navigation_3d else 0.0
            vertical_progress = 0.0
            vertical_error_penalty = 0.0
            if self.dynamic_model.navigation_3d:
                vertical_progress = float(getattr(self, 'previous_vertical_error', vertical_error) - vertical_error)
                self.previous_vertical_error = vertical_error
                vertical_error_penalty = vertical_error_penalty_coef * np.clip(
                    vertical_error / max(float(self.dynamic_model.max_vertical_difference), 1e-6), 0.0, 1.0)

            reward = reward_distance + vertical_reward_coef * vertical_progress + heading_reward - \
                pose_penalty_coef * punishment_pose - obstacle_penalty_coef * punishment_obs - \
                action_penalty_coef * punishment_action - yaw_penalty_coef * yaw_error_cost - \
                step_penalty - progress_penalty - reverse_progress_penalty - low_speed_penalty - \
                heading_error_penalty - boundary_penalty - path_penalty - vertical_error_penalty

            if isinstance(getattr(self, '_current_step_info', None), dict):
                self._current_step_info.update({
                    'distance_progress': float(distance_progress),
                    'progress_penalty': float(progress_penalty),
                    'reverse_progress_penalty': float(reverse_progress_penalty),
                    'low_speed_penalty': float(low_speed_penalty),
                    'heading_alignment': float(heading_alignment),
                    'heading_reward': float(heading_reward),
                    'heading_error_penalty': float(heading_error_penalty),
                    'boundary_margin': float(boundary_margin),
                    'boundary_cost': float(boundary_cost),
                    'z_boundary_margin': float(z_boundary_margin),
                    'z_boundary_cost': float(z_boundary_cost),
                    'boundary_penalty': float(boundary_penalty),
                    'path_distance': float(path_distance),
                    'path_penalty': float(path_penalty),
                    'current_z': float(z),
                    'goal_z': float(z_g),
                    'vertical_error': float(vertical_error),
                    'vertical_progress': float(vertical_progress),
                    'vertical_error_penalty': float(vertical_error_penalty),
                })
        else:
            terminal_info = getattr(self, '_current_step_info', None) or self.get_done_info()
            if terminal_info.get('is_crash'):
                reward = reward_crash
            elif terminal_info.get('is_not_in_workspace'):
                reward = reward_outside
            elif terminal_info.get('is_success'):
                reward = reward_reach
            elif terminal_info.get('is_stuck'):
                reward = self._cfg_getfloat('reward', 'reward_stuck', reward_timeout)
            elif terminal_info.get('is_timeout') or terminal_info.get('is_max_steps'):
                reward = reward_timeout

        return reward

    def compute_reward_final_fixedwing(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        if not done:
            # 1 - goal reward
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = 300 * (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance   # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            # 2 - Position punishment
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x = current_pose[0]
            y = current_pose[1]
            x_g = goal_pose[0]
            y_g = goal_pose[1]

            punishment_xy = np.clip(self.getDis(
                x, y, 0, 0, x_g, y_g) / 50, 0, 1)
            # punishment_z = 0.5 * np.clip(abs(z - z_g)/5, 0, 1)

            punishment_pose = punishment_xy

            if self.min_distance_to_obstacles < 20:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 15, 0, 1)
            else:
                punishment_obs = 0

            # action cost
            punishment_action = abs(action[0]) / self.dynamic_model.roll_rate_max

            yaw_error = self.dynamic_model.state_raw[1]
            yaw_error_cost = abs(yaw_error / 90)

            reward = reward_distance - 0.1 * punishment_pose - 0.5 * \
                punishment_obs - 0.1 * punishment_action - 0.1 * yaw_error_cost
            # reward = reward

            # print("r_dist: {:.2f} p_pose: {:.2f} p_obs: {:.2f} p_action: {:.2f}, p_yaw_e: {:.2f}".format(reward_distance, punishment_pose, punishment_obs, punishment_action, yaw_error_cost))
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_test(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -100
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance * \
                100  # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # add yaw_rate cost
            yaw_speed_cost = 0.1 * \
                abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = 0.1 * abs(action[1]) / self.dynamic_model.v_z_max
                z_err_cost = 0.05 * \
                    abs(self.dynamic_model.state_raw[1]) / \
                    self.dynamic_model.max_vertical_difference
                action_cost += (v_z_cost + z_err_cost)

            action_cost += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.1 * abs(yaw_error / 180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_fixedwing(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -50
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance * \
                300  # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            # 只有action cost和obs cost
            # 由于没有速度控制，所以前面那个也取消了
            # action_cost = 0
            # obs_cost = 0

            # relative_yaw_cost = abs(
            #     (self.dynamic_model.state_norm[0]/255-0.5) * 2)
            # action_cost = abs(action[0]) / self.dynamic_model.roll_rate_max

            # obs_punish_distance = 15
            # if self.min_distance_to_obstacles < obs_punish_distance:
            #     obs_cost = 1 - (self.min_distance_to_obstacles -
            #                     self.crash_distance) / (obs_punish_distance -
            #                                             self.crash_distance)
            #     obs_cost = 0.5 * obs_cost ** 2
            # reward = reward_distance - (2 * relative_yaw_cost + 0.5 * action_cost + obs_cost)

            action_cost = abs(action[0]) / self.dynamic_model.roll_rate_max

            yaw_error_deg = self.dynamic_model.state_raw[1]
            yaw_error_cost = 0.1 * abs(yaw_error_deg / 180)

            reward = reward_distance - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                yaw_error_deg = self.dynamic_model.state_raw[1]
                reward = reward_reach * (1 - abs(yaw_error_deg / 180))
                # reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_multirotor_new(self, done, action):
        reward = 0
        reward_reach = 100
        reward_crash = -100
        reward_outside = 0

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point -
                               distance_now) / self.dynamic_model.goal_distance * 5
            self.previous_distance_from_des_point = distance_now

            state_cost = 0
            action_cost = 0
            obs_cost = 0

            yaw_error_deg = self.dynamic_model.state_raw[1]

            relative_yaw_cost = abs(yaw_error_deg/180)
            action_cost = abs(action[1]) / self.dynamic_model.yaw_rate_max_rad

            obs_punish_dist = 5
            if self.min_distance_to_obstacles < obs_punish_dist:
                obs_cost = 1 - (self.min_distance_to_obstacles -
                                self.crash_distance) / (obs_punish_dist - self.crash_distance)
                obs_cost = 0.5 * obs_cost ** 2
            reward = - (2 * relative_yaw_cost + 0.5 * action_cost)
        else:
            if self.is_in_desired_pose():
                # 到达之后根据yaw偏差对reward进行scale
                reward = reward_reach * \
                    (1 - abs(self.dynamic_model.state_norm[1]))
                # reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_with_action(self, done, action):
        reward = 0
        reward_reach = 50
        reward_crash = -50
        reward_outside = -10

        step_cost = 0.01  # 10 for max 1000 steps

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance * \
                10  # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # add action cost
            # speed 0-8  cruise speed is 4, punish for too fast and too slow
            v_xy_cost = 0.02 * abs(action[0]-5) / 4
            yaw_rate_cost = 0.02 * \
                abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad
            if self.dynamic_model.navigation_3d:
                v_z_cost = 0.02 * abs(action[1]) / self.dynamic_model.v_z_max
                action_cost += v_z_cost
            action_cost += (v_xy_cost + yaw_rate_cost)

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.05 * abs(yaw_error/180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_lqr(self, done, action):
        # 模仿matlab提供的mix reward的思想设计
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = 0

        if not done:
            action_cost = 0
            # add yaw_rate cost
            yaw_speed_cost = 0.2 * \
                ((action[-1] / self.dynamic_model.yaw_rate_max_rad) ** 2)

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = 0.1 * ((action[1] / self.dynamic_model.v_z_max)**2)
                z_err_cost = 0.1 * \
                    ((self.dynamic_model.state_raw[1] /
                      self.dynamic_model.max_vertical_difference)**2)
                action_cost += (v_z_cost + z_err_cost)

            action_cost += yaw_speed_cost

            yaw_error_clip = min(
                max(-60, self.dynamic_model.state_raw[2]), 60) / 60
            yaw_error_cost = 1.0 * (yaw_error_clip**2)

            reward = - (action_cost + yaw_error_cost)

            # print('r: {:.2f} y_r: {:.2f} y_e: {:.2f} z_r: {:.2f} z_e: {:.2f}'.format(reward, yaw_speed_cost, yaw_error_cost, v_z_cost, z_err_cost))
        else:
            if self.is_in_desired_pose():
                yaw_error_clip = min(
                    max(-30, self.dynamic_model.state_raw[2]), 30) / 30
                reward = reward_reach * (1 - yaw_error_clip**2)
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

# ! ------------------ is done-----------------------------------------------

    def is_done(self):
        return self.get_done_info()['done']

    def get_done_info(self):
        is_success = self.is_in_desired_pose()
        is_crash = self.is_crashed()
        is_not_in_workspace = self.is_not_inside_workspace()
        episode_step = self.step_num + 1
        current_distance = float(self.get_distance_to_goal_3d())
        is_stuck = self._update_stuck_status(current_distance, episode_step)
        # This is a max-episode/search-step termination, not wall-clock runtime.
        is_max_steps = episode_step >= self.max_episode_steps
        is_timeout = is_max_steps  # Backward-compatible alias used by older logs/configs.
        done = is_crash or is_not_in_workspace or is_success or is_max_steps or is_stuck
        if is_crash:
            done_reason = 'crash'
        elif is_not_in_workspace:
            done_reason = 'outside'
        elif is_success:
            done_reason = 'success'
        elif is_stuck:
            done_reason = 'stuck'
        elif is_max_steps:
            done_reason = 'max_steps'
        else:
            done_reason = ''
        return {
            'is_success': is_success,
            'is_crash': is_crash,
            'is_not_in_workspace': is_not_in_workspace,
            'is_timeout': is_timeout,
            'is_max_steps': is_max_steps,
            'is_stuck': is_stuck,
            'done_reason': done_reason,
            'done': done,
            'step_num': episode_step,
            'episode_step': episode_step,
            'max_episode_steps': self.max_episode_steps,
            'no_progress_steps': int(getattr(self, 'no_progress_steps', 0)),
            'best_distance_to_goal': float(getattr(self, 'best_distance_to_goal', current_distance)),
            'current_distance_to_goal': current_distance,
        }

    def _update_stuck_status(self, current_distance, episode_step):
        """Detect policies that spend many steps without reducing goal distance.

        Long 1000-step episodes where the UAV circles or hovers provide weak SAC
        feedback and slow down training.  This configurable early termination marks
        those episodes as ``stuck`` so the policy receives an immediate terminal
        penalty instead of wasting the full max_episode_steps budget.
        """
        enabled = self.cfg.getboolean('safety', 'stuck_check_enabled', fallback=False)
        if not enabled:
            return False

        if getattr(self, '_last_stuck_check_step', -1) == episode_step:
            return bool(getattr(self, '_last_is_stuck', False))
        self._last_stuck_check_step = episode_step

        max_no_progress_steps = self.cfg.getint('safety', 'max_no_progress_steps', fallback=200)
        warmup_steps = self.cfg.getint('safety', 'stuck_warmup_steps', fallback=50)
        min_progress = self._cfg_getfloat('safety', 'stuck_min_progress', 0.5)
        best_distance = float(getattr(self, 'best_distance_to_goal', current_distance))
        if current_distance < best_distance - min_progress:
            self.best_distance_to_goal = float(current_distance)
            self.no_progress_steps = 0
        elif episode_step > warmup_steps:
            self.no_progress_steps = int(getattr(self, 'no_progress_steps', 0)) + 1

        is_stuck = int(getattr(self, 'no_progress_steps', 0)) >= max(1, max_no_progress_steps)
        self._last_is_stuck = bool(is_stuck)
        return bool(is_stuck)

    def _add_constraint_info(self, info, action):
        """Attach safety costs used by GPIDE/FOCOPS sequence SAC.

        The sequence agent reads ``info['constraint_cost']`` as the cost target
        for its FOCOPS-inspired cost critics.  Keep this independent from the
        scalar reward so reward shaping and safety constraints can be tuned
        separately.  Crash receives a terminal cost even if the obstacle distance
        was not measurable on that step.
        """
        if info is None:
            return

        safe_distance = self._cfg_getfloat('constraint', 'safe_distance', self.crash_distance + 3.0)
        crash_cost = self._cfg_getfloat('constraint', 'crash_cost', 1.0)
        outside_cost = self._cfg_getfloat('constraint', 'outside_cost', 0.5)
        max_steps_cost = self._cfg_getfloat('constraint', 'max_steps_cost', 0.3)
        stuck_cost = self._cfg_getfloat('constraint', 'stuck_cost', max_steps_cost)
        action_cost_coef = self._cfg_getfloat('constraint', 'action_cost_coef', 0.05)
        yaw_cost_coef = self._cfg_getfloat('constraint', 'yaw_cost_coef', 0.05)
        boundary_safe_margin = self._cfg_getfloat('constraint', 'boundary_safe_margin', 5.0)
        z_boundary_safe_margin = self._cfg_getfloat('constraint', 'z_boundary_safe_margin', 2.0)
        boundary_cost_coef = self._cfg_getfloat('constraint', 'boundary_cost_coef', 0.3)

        distance_margin = max(safe_distance - self.crash_distance, 1e-6)
        min_distance = float(getattr(self, 'min_distance_to_obstacles', safe_distance))
        if np.isfinite(min_distance):
            obstacle_cost = 1.0 - np.clip((min_distance - self.crash_distance) / distance_margin, 0.0, 1.0)
        else:
            obstacle_cost = 0.0

        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size:
            # Do not treat forward speed itself as unsafe, otherwise the safety
            # critic can learn to prefer crawling or circling.  In 2D we use yaw
            # effort; in 3D include vertical effort as well.
            action_span = np.maximum(np.asarray(self.action_space.high, dtype=np.float32).reshape(-1), 1e-6)
            controlled = [abs(action_arr[-1]) / action_span[-1]]
            if getattr(self.dynamic_model, 'navigation_3d', False) and action_arr.size > 2:
                controlled.append(abs(action_arr[1]) / action_span[1])
            action_cost = float(np.clip(np.mean(controlled), 0.0, 1.0))
        else:
            action_cost = 0.0

        yaw_error = 0.0
        if hasattr(self.dynamic_model, 'state_raw') and len(self.dynamic_model.state_raw) > 2:
            yaw_error = float(abs(self.dynamic_model.state_raw[2]) / 180.0)
        yaw_error_cost = float(np.clip(yaw_error, 0.0, 1.0))
        boundary_margin = self.get_workspace_margin(include_z=False)
        z_boundary_margin = self.get_workspace_z_margin() if getattr(self.dynamic_model, 'navigation_3d', False) else float('inf')
        boundary_cost = float(1.0 - np.clip(boundary_margin / max(boundary_safe_margin, 1e-6), 0.0, 1.0))
        z_boundary_cost = 0.0
        if getattr(self.dynamic_model, 'navigation_3d', False):
            z_boundary_cost = float(1.0 - np.clip(z_boundary_margin / max(z_boundary_safe_margin, 1e-6), 0.0, 1.0))
        combined_boundary_cost = max(boundary_cost, z_boundary_cost)

        terminal_cost = 0.0
        if info.get('is_crash'):
            terminal_cost = crash_cost
            obstacle_cost = max(float(obstacle_cost), crash_cost)
        elif info.get('is_not_in_workspace'):
            terminal_cost = outside_cost
        elif info.get('is_stuck'):
            terminal_cost = stuck_cost
        elif info.get('is_max_steps') or info.get('is_timeout'):
            terminal_cost = max_steps_cost

        constraint_cost = max(
            terminal_cost,
            float(obstacle_cost) + action_cost_coef * action_cost + yaw_cost_coef * yaw_error_cost +
            boundary_cost_coef * combined_boundary_cost,
        )
        info.update({
            'constraint_cost': float(np.clip(constraint_cost, 0.0, max(1.0, crash_cost))),
            'obstacle_cost': float(np.clip(obstacle_cost, 0.0, max(1.0, crash_cost))),
            'action_cost': action_cost,
            'yaw_error_cost': yaw_error_cost,
            'executed_action': np.asarray(action, dtype=np.float32).reshape(-1).tolist(),
            'boundary_margin': float(boundary_margin),
            'boundary_cost': boundary_cost,
            'z_boundary_margin': float(z_boundary_margin),
            'z_boundary_cost': z_boundary_cost,
            'distance_to_goal': float(self.dynamic_model.goal_distance),
            'current_distance_to_goal': float(self.get_distance_to_goal_3d()),
            'relative_yaw_deg': float(self.dynamic_model.state_raw[2]) if hasattr(self.dynamic_model, 'state_raw') and len(self.dynamic_model.state_raw) > 2 else 0.0,
            'current_z': float(self.dynamic_model.get_position()[2]),
            'goal_z': float(self.dynamic_model.goal_position[2]),
            'vertical_error': float(abs(self.dynamic_model.get_position()[2] - self.dynamic_model.goal_position[2])),
            'min_distance_to_obstacles': min_distance,
            'no_progress_steps': int(getattr(self, 'no_progress_steps', 0)),
            'best_distance_to_goal': float(getattr(self, 'best_distance_to_goal', self.get_distance_to_goal_3d())),
            **getattr(self, '_goal_capture_info', {}),
            **getattr(self, '_obstacle_shield_info', {}),
            **getattr(self, '_boundary_shield_info', {}),
        })


    def apply_goal_capture_shield(self, action):
        """Prioritize capture when the vehicle is already near the goal.

        SimpleMultirotor cannot command a zero forward speed, so a policy that is
        close to the target can still fly through the acceptance area between two
        simulator ticks or be turned away by obstacle/boundary shields.  This
        optional local controller only activates inside a small capture radius:
        it slows to minimum speed, points yaw toward the goal, and biases vertical
        speed toward the goal altitude.
        """
        self._goal_capture_info = {'goal_capture_active': 0}
        enabled = self.cfg.getboolean('safety', 'goal_capture_shield_enabled', fallback=False)
        action_arr = np.asarray(action, dtype=np.float32).copy()
        if not enabled or self.dynamic_name != 'SimpleMultirotor' or action_arr.size < 2:
            return action_arr

        current_position = self.dynamic_model.get_position()
        goal_position = self.dynamic_model.goal_position
        dx = float(goal_position[0] - current_position[0])
        dy = float(goal_position[1] - current_position[1])
        horizontal_distance = math.hypot(dx, dy)
        capture_radius = self._cfg_getfloat('safety', 'goal_capture_radius', self.accept_radius * 2.0)
        if horizontal_distance > capture_radius:
            return action_arr

        desired_yaw = math.atan2(dy, dx)
        yaw_error = desired_yaw - self.dynamic_model.get_attitude()[2]
        if yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        yaw_gain = self._cfg_getfloat('safety', 'goal_capture_yaw_gain', 2.0)
        action_arr[0] = float(self.dynamic_model.v_xy_min)
        action_arr[-1] = np.clip(
            yaw_gain * yaw_error,
            -self.dynamic_model.yaw_rate_max_rad,
            self.dynamic_model.yaw_rate_max_rad,
        )

        if getattr(self.dynamic_model, 'navigation_3d', False) and action_arr.size >= 3:
            z_error = float(goal_position[2] - current_position[2])
            z_gain = self._cfg_getfloat('safety', 'goal_capture_z_gain', 0.8)
            action_arr[1] = np.clip(
                z_gain * z_error,
                -self.dynamic_model.v_z_max,
                self.dynamic_model.v_z_max,
            )
        else:
            z_error = 0.0

        self._goal_capture_info = {
            'goal_capture_active': 1,
            'goal_capture_horizontal_distance': float(horizontal_distance),
            'goal_capture_yaw_error_deg': float(math.degrees(yaw_error)),
            'goal_capture_z_error': float(z_error),
        }
        return action_arr.astype(np.float32)

    def apply_obstacle_shield(self, action):
        """Bias exploration away from close frontal depth obstacles.

        The SAC policy only sees compact state/depth features and can spend early
        replay crashing before it learns reliable avoidance.  This optional guard
        uses the previous depth frame to slow down and add a turn toward the
        clearer image side when a frontal obstacle is already close.  It is
        disabled by default and only activates when ``safety.obstacle_shield_enabled``
        is set, so existing 2D configs keep their learned behavior.
        """
        enabled = self.cfg.getboolean('safety', 'obstacle_shield_enabled', fallback=False)
        action_arr = np.asarray(action, dtype=np.float32).copy()
        self._obstacle_shield_info = {'obstacle_shield_active': 0}
        if not enabled or self.dynamic_name != 'SimpleMultirotor' or action_arr.size < 2:
            return action_arr
        if getattr(self, '_goal_capture_info', {}).get('goal_capture_active'):
            return action_arr

        depth_image = getattr(self, 'last_depth_image', None)
        if depth_image is None:
            return action_arr

        depth = np.asarray(depth_image, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            return action_arr

        finite_depth = depth[np.isfinite(depth)]
        if finite_depth.size == 0:
            return action_arr

        threshold_cfg = self._cfg_getfloat('safety', 'front_obstacle_threshold', 0.35)
        if threshold_cfg <= 1.0:
            threshold_m = threshold_cfg * float(self.max_depth_meters)
        else:
            threshold_m = threshold_cfg

        h, w = depth.shape
        row0 = int(h * 0.25)
        row1 = int(h * 0.85)
        col0 = int(w * 0.25)
        col1 = int(w * 0.75)
        front = depth[row0:row1, col0:col1]
        left = depth[row0:row1, :max(col0, 1)]
        right = depth[row0:row1, min(col1, w - 1):]

        front_min = float(np.nanmin(front)) if front.size else float(np.nanmin(finite_depth))
        if not np.isfinite(front_min) or front_min >= threshold_m:
            self._obstacle_avoid_turn_sign = 0.0
            self._obstacle_avoid_turn_steps = 0
            self._obstacle_shield_info = {
                'obstacle_shield_active': 0,
                'obstacle_shield_front_min': front_min,
                'obstacle_shield_threshold_m': float(threshold_m),
            }
            return action_arr

        left_clearance = float(np.nanpercentile(left, 30)) if left.size else front_min
        right_clearance = float(np.nanpercentile(right, 30)) if right.size else front_min
        raw_side_sign = 1.0 if left_clearance >= right_clearance else -1.0

        # Hold the same avoidance direction for a few steps.  In dense NH_center
        # streets a single depth frame can alternate between left/right as the
        # camera passes tree trunks; without hysteresis the shield dithers and the
        # vehicle keeps flying into the obstacle corridor.
        hold_steps = self.cfg.getint('safety', 'obstacle_shield_turn_hold_steps', fallback=8)
        held_steps = int(getattr(self, '_obstacle_avoid_turn_steps', 0))
        held_sign = float(getattr(self, '_obstacle_avoid_turn_sign', raw_side_sign))
        if held_steps > 0:
            side_sign = held_sign
            self._obstacle_avoid_turn_steps = held_steps - 1
        else:
            side_sign = raw_side_sign
            self._obstacle_avoid_turn_sign = side_sign
            self._obstacle_avoid_turn_steps = max(0, hold_steps - 1)

        emergency_cfg = self._cfg_getfloat('safety', 'obstacle_emergency_threshold', 0.22)
        emergency_m = emergency_cfg * float(self.max_depth_meters) if emergency_cfg <= 1.0 else emergency_cfg
        emergency_active = front_min <= emergency_m
        proximity = 1.0 - np.clip((front_min - emergency_m) / max(threshold_m - emergency_m, 1e-6), 0.0, 1.0)

        min_shield_speed = self._cfg_getfloat('safety', 'obstacle_shield_min_speed', float(self.dynamic_model.v_xy_min))
        emergency_speed = self._cfg_getfloat('safety', 'obstacle_emergency_speed', min_shield_speed)
        speed_scale = self._cfg_getfloat('safety', 'speed_scale', 0.7)
        scaled_policy_speed = float(action_arr[0]) * speed_scale
        target_speed = min_shield_speed + (1.0 - proximity) * max(0.0, scaled_policy_speed - min_shield_speed)
        if emergency_active:
            target_speed = min(target_speed, emergency_speed)
        action_arr[0] = np.clip(target_speed, self.dynamic_model.v_xy_min, self.dynamic_model.v_xy_max)

        yaw_bias = self._cfg_getfloat('safety', 'yaw_rate_bias', 1.0)
        emergency_yaw_bias = self._cfg_getfloat('safety', 'obstacle_emergency_yaw_rate_bias', yaw_bias)
        goal_turn_bias = self._cfg_getfloat('safety', 'goal_turn_bias', 0.0)
        goal_yaw = 0.0
        if hasattr(self.dynamic_model, 'state_raw') and len(self.dynamic_model.state_raw) > 2:
            goal_yaw = float(np.clip(math.radians(self.dynamic_model.state_raw[2]), -1.0, 1.0))
        yaw_cmd = side_sign * (emergency_yaw_bias if emergency_active else yaw_bias) + goal_turn_bias * goal_yaw * (1.0 - proximity)
        action_arr[-1] = np.clip(
            yaw_cmd,
            -self.dynamic_model.yaw_rate_max_rad,
            self.dynamic_model.yaw_rate_max_rad,
        )

        if getattr(self.dynamic_model, 'navigation_3d', False) and action_arr.size >= 3:
            vertical_goal_bias = self._cfg_getfloat('safety', 'vertical_goal_bias', 0.0)
            vertical_error = float(self.dynamic_model.goal_position[2] - self.dynamic_model.get_position()[2])
            z_damping = self._cfg_getfloat('safety', 'obstacle_shield_z_damping', 0.5)
            emergency_z_damping = self._cfg_getfloat('safety', 'obstacle_emergency_z_damping', z_damping)
            damping = emergency_z_damping if emergency_active else z_damping
            action_arr[1] = np.clip(
                damping * action_arr[1] + vertical_goal_bias * np.clip(vertical_error, -1.0, 1.0) * (1.0 - proximity),
                -self.dynamic_model.v_z_max,
                self.dynamic_model.v_z_max,
            )

        self._obstacle_shield_info = {
            'obstacle_shield_active': 1,
            'obstacle_shield_emergency_active': int(emergency_active),
            'obstacle_shield_front_min': front_min,
            'obstacle_shield_threshold_m': float(threshold_m),
            'obstacle_shield_emergency_threshold_m': float(emergency_m),
            'obstacle_shield_proximity': float(proximity),
            'obstacle_shield_left_clearance': left_clearance,
            'obstacle_shield_right_clearance': right_clearance,
            'obstacle_shield_turn_sign': float(side_sign),
            'obstacle_shield_raw_turn_sign': float(raw_side_sign),
        }
        return action_arr.astype(np.float32)

    def apply_boundary_shield(self, action):
        """Limit outward actions near workspace boundaries.

        Reward penalties alone often arrive too late: early exploration may cross
        the boundary before the policy learns the cost.  This light-weight guard
        only activates close to the workspace edge and when the commanded forward
        direction points outward.
        """
        self._boundary_shield_info = {}
        enabled = self.cfg.getboolean('safety', 'boundary_shield_enabled', fallback=False)
        if not enabled or self.dynamic_name != 'SimpleMultirotor':
            return action

        action_arr = np.asarray(action, dtype=np.float32).copy()
        if action_arr.size < 2:
            return action_arr
        if getattr(self, '_goal_capture_info', {}).get('goal_capture_active'):
            self._boundary_shield_info = {'boundary_shield_active': 0}
            return action_arr

        position = self.dynamic_model.get_position()
        margin = self.get_workspace_margin(include_z=False)
        z_margin = self.get_workspace_z_margin() if getattr(self.dynamic_model, 'navigation_3d', False) else float('inf')
        shield_margin = self._cfg_getfloat('safety', 'boundary_shield_margin', 10.0)
        z_shield_margin = self._cfg_getfloat('safety', 'boundary_shield_z_margin', 2.0)
        xy_near_boundary = margin < shield_margin
        z_near_boundary = z_margin < z_shield_margin
        if not xy_near_boundary and not z_near_boundary:
            self._boundary_shield_info = {
                'boundary_shield_active': 0,
                'boundary_shield_margin': float(margin),
                'boundary_shield_z_margin': float(z_margin),
            }
            return action_arr

        yaw = self.dynamic_model.get_attitude()[2]
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        outward_score = 0.0
        if xy_near_boundary:
            if position[0] - self.work_space_x[0] < shield_margin:
                outward_score = max(outward_score, -forward_x)
            if self.work_space_x[1] - position[0] < shield_margin:
                outward_score = max(outward_score, forward_x)
            if position[1] - self.work_space_y[0] < shield_margin:
                outward_score = max(outward_score, -forward_y)
            if self.work_space_y[1] - position[1] < shield_margin:
                outward_score = max(outward_score, forward_y)

        z_outward_score = 0.0
        if getattr(self.dynamic_model, 'navigation_3d', False) and action_arr.size >= 3 and z_near_boundary:
            z_speed_span = max(float(getattr(self.dynamic_model, 'v_z_max', 1.0)), 1e-6)
            if position[2] - self.work_space_z[0] < z_shield_margin and action_arr[1] < 0.0:
                z_outward_score = max(z_outward_score, float(-action_arr[1]) / z_speed_span)
            if self.work_space_z[1] - position[2] < z_shield_margin and action_arr[1] > 0.0:
                z_outward_score = max(z_outward_score, float(action_arr[1]) / z_speed_span)

        if outward_score <= 0.0 and z_outward_score <= 0.0:
            self._boundary_shield_info = {
                'boundary_shield_active': 0,
                'boundary_shield_margin': float(margin),
                'boundary_outward_score': float(outward_score),
                'boundary_z_outward_score': float(z_outward_score),
                'boundary_shield_z_margin': float(z_margin),
            }
            return action_arr

        center_x = 0.5 * (self.work_space_x[0] + self.work_space_x[1])
        center_y = 0.5 * (self.work_space_y[0] + self.work_space_y[1])
        desired_yaw = math.atan2(center_y - position[1], center_x - position[0])
        yaw_error = desired_yaw - yaw
        if yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        shield_speed = self._cfg_getfloat('safety', 'boundary_shield_speed', 1.0)
        yaw_gain = self._cfg_getfloat('safety', 'boundary_shield_yaw_gain', 1.5)
        action_arr[0] = min(float(action_arr[0]), max(float(self.dynamic_model.v_xy_min), shield_speed))
        if outward_score > 0.0:
            action_arr[-1] = np.clip(
                yaw_gain * yaw_error,
                -self.dynamic_model.yaw_rate_max_rad,
                self.dynamic_model.yaw_rate_max_rad,
            )
        if z_outward_score > 0.0 and getattr(self.dynamic_model, 'navigation_3d', False) and action_arr.size >= 3:
            z_center = 0.5 * (self.work_space_z[0] + self.work_space_z[1])
            z_direction = np.sign(z_center - position[2])
            z_shield_speed = self._cfg_getfloat('safety', 'boundary_shield_z_speed', 0.5)
            action_arr[1] = np.clip(
                z_direction * z_shield_speed,
                -self.dynamic_model.v_z_max,
                self.dynamic_model.v_z_max,
            )
        self._boundary_shield_info = {
            'boundary_shield_active': 1,
            'boundary_shield_margin': float(margin),
            'boundary_outward_score': float(outward_score),
            'boundary_z_outward_score': float(z_outward_score),
            'boundary_shield_z_margin': float(z_margin),
            'boundary_shield_yaw_error_deg': float(math.degrees(yaw_error)),
        }
        return action_arr.astype(np.float32)

    def _cfg_getfloat(self, section, option, default):
        try:
            return self.cfg.getfloat(section, option)
        except (NoOptionError, NoSectionError, ValueError):
            return default

    def get_workspace_margin(self, include_z=None):
        """Return distance to the nearest relevant workspace boundary.

        SimpleMultirotor 2D policies keep altitude fixed, so using the z-axis
        clearance for boundary reward/cost shaping makes the agent look close to
        a boundary everywhere when the nominal flight height is below the xy
        safety margin.  3D training has the same issue when the vertical
        workspace is much narrower than x/y: use x/y margin by default for
        reward, constraints and yaw shielding, and handle z with a separate,
        smaller margin.
        """
        current_position = self.dynamic_model.get_position()
        margins = [
            current_position[0] - self.work_space_x[0],
            self.work_space_x[1] - current_position[0],
            current_position[1] - self.work_space_y[0],
            self.work_space_y[1] - current_position[1],
        ]
        if include_z is None:
            include_z = False
        if include_z:
            margins.extend([
                current_position[2] - self.work_space_z[0],
                self.work_space_z[1] - current_position[2],
            ])
        return float(min(margins))

    def get_workspace_z_margin(self):
        """Return distance to the nearest vertical workspace boundary."""
        current_z = self.dynamic_model.get_position()[2]
        return float(min(
            current_z - self.work_space_z[0],
            self.work_space_z[1] - current_z,
        ))

    def is_not_inside_workspace(self):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_not_inside = False
        current_position = self.dynamic_model.get_position()

        if current_position[0] < self.work_space_x[0] or current_position[0] > self.work_space_x[1] or \
            current_position[1] < self.work_space_y[0] or current_position[1] > self.work_space_y[1] or \
                current_position[2] < self.work_space_z[0] or current_position[2] > self.work_space_z[1]:
            is_not_inside = True

        return is_not_inside

    def is_in_desired_pose(self):
        """Check goal capture using point and last-step segment distance.

        Using only the current 3D point can miss close goals when the
        SimpleMultirotor moves through the acceptance region between two 0.2 s
        ticks, especially because it always has a positive forward-speed lower
        bound.  Treat horizontal and vertical tolerances separately for 3D, then
        also check the swept segment from the previous position to the current
        position.
        """
        current_position = np.asarray(self.dynamic_model.get_position(), dtype=np.float32)
        goal_position = np.asarray(self.dynamic_model.goal_position, dtype=np.float32)
        point_success, point_metrics = self._is_goal_capture_position(current_position, goal_position)
        segment_success, segment_metrics = self._is_goal_capture_segment(goal_position)
        self._goal_capture_status = {
            **point_metrics,
            **segment_metrics,
            'goal_capture_success_point': int(point_success),
            'goal_capture_success_segment': int(segment_success),
        }
        return bool(point_success or segment_success)

    def _is_goal_capture_position(self, position, goal_position):
        horizontal_distance = float(np.linalg.norm(position[:2] - goal_position[:2]))
        vertical_error = float(abs(position[2] - goal_position[2]))
        distance_3d = float(np.linalg.norm(position - goal_position))
        if getattr(self.dynamic_model, 'navigation_3d', False):
            accept_z_radius = self._cfg_getfloat('environment', 'accept_z_radius', self.accept_radius)
            success = horizontal_distance <= self.accept_radius and vertical_error <= accept_z_radius
        else:
            accept_z_radius = float('inf')
            success = horizontal_distance <= self.accept_radius
        return bool(success), {
            'goal_horizontal_distance': horizontal_distance,
            'goal_vertical_error': vertical_error,
            'goal_distance_3d': distance_3d,
            'accept_z_radius': float(accept_z_radius),
        }

    def _is_goal_capture_segment(self, goal_position):
        previous_position = getattr(self, '_position_before_action', None)
        if previous_position is None:
            return False, {'goal_segment_distance': float('inf')}

        previous = np.asarray(previous_position, dtype=np.float32)
        current = np.asarray(self.dynamic_model.get_position(), dtype=np.float32)
        delta_xy = current[:2] - previous[:2]
        segment_len_sq = float(np.dot(delta_xy, delta_xy))
        if segment_len_sq <= 1e-9:
            closest = current
        else:
            t = float(np.clip(np.dot(goal_position[:2] - previous[:2], delta_xy) / segment_len_sq, 0.0, 1.0))
            closest = previous + t * (current - previous)

        horizontal_distance = float(np.linalg.norm(closest[:2] - goal_position[:2]))
        vertical_error = float(abs(closest[2] - goal_position[2]))
        if getattr(self.dynamic_model, 'navigation_3d', False):
            accept_z_radius = self._cfg_getfloat('environment', 'accept_z_radius', self.accept_radius)
            success = horizontal_distance <= self.accept_radius and vertical_error <= accept_z_radius
        else:
            success = horizontal_distance <= self.accept_radius
        return bool(success), {
            'goal_segment_distance': horizontal_distance,
            'goal_segment_vertical_error': vertical_error,
        }

    def is_crashed(self):
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided or self.min_distance_to_obstacles < self.crash_distance:
            is_crashed = True

        return is_crashed

# ! ----------- useful functions-------------------------------------------
    def get_distance_to_goal_3d(self):
        current_pose = self.dynamic_model.get_position()
        goal_pose = self.dynamic_model.goal_position
        relative_pose_x = current_pose[0] - goal_pose[0]
        relative_pose_y = current_pose[1] - goal_pose[1]
        relative_pose_z = current_pose[2] - goal_pose[2]

        return math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2) + pow(relative_pose_z, 2))

    def getDis(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):
        '''
        Get distance between Point and Line
        Used to calculate position punishment
        '''
        a = lineY2-lineY1
        b = lineX1-lineX2
        c = lineX2*lineY1-lineX1*lineY2
        dis = (math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b, 0.5))

        return dis
# ! -----------used for plot or show states------------------------------------------------------------------

    def print_train_info_airsim(self, action, obs, reward, info):
        if not self.print_train_info:
            return
        # if self.perception_type == 'split' or self.perception_type == 'lgmd':
        #     feature_all = self.feature_all
        # elif self.perception_type == 'vector':
        #     feature_all = self.feature_all
        # else:
        #     if self.cfg.get('options', 'algo') == 'TD3' or self.cfg.get('options', 'algo') == 'SAC':
        #         feature_all = self.model.actor.features_extractor.feature_all
        #     elif self.cfg.get('options', 'algo') == 'PPO':
        #         feature_all = self.model.policy.features_extractor.feature_all

        # self.client.simPrintLogMessage('feature_all: ', str(feature_all))

        msg_train_info = "EP: {} Step: {} Total_step: {}".format(
            self.episode_num, self.step_num, self.total_step)

        self.client.simPrintLogMessage('Train: ', msg_train_info)
        self.client.simPrintLogMessage('Action: ', str(action))
        self.client.simPrintLogMessage('reward: ', "{:4.4f} total: {:4.4f}".format(
            reward, self.cumulated_episode_reward))
        self.client.simPrintLogMessage('Info: ', str(info))
        self.client.simPrintLogMessage(
            'Feature_norm: ', str(self.dynamic_model.state_norm))
        self.client.simPrintLogMessage(
            'Feature_raw: ', str(self.dynamic_model.state_raw))
        self.client.simPrintLogMessage(
            'Min_depth: ', str(self.min_distance_to_obstacles))

    def set_pyqt_signal_fixedwing(self, action, reward, done):
        """
        emit signals for pyqt plot
        """
        step = int(self.total_step)
        # action: v_xy, v_z, roll

        action_plot = np.array([10, 0, math.degrees(action[0])])

        state = self.dynamic_model.state_raw  # distance, relative yaw, roll

        # state out 6: d_xy, d_z, yaw_error, v_xy, v_z, roll
        # state in  3: d_xy, yaw_error, roll
        state_output = np.array([state[0], 0, state[1], 10, 0, state[2]])

        self.action_signal.emit(step, action_plot)
        self.state_signal.emit(step, state_output)

        # other values
        self.attitude_signal.emit(step, np.asarray(self.dynamic_model.get_attitude(
        )), np.asarray(self.dynamic_model.get_attitude_cmd()))
        self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.pose_signal.emit(np.asarray(self.dynamic_model.goal_position), np.asarray(
            self.dynamic_model.start_position), np.asarray(self.dynamic_model.get_position()), np.asarray(self.trajectory_list))

        # lgmd_signal = pyqtSignal(float, float, np.ndarray)  min_dist, lgmd_out, lgmd_split
        self.lgmd_signal.emit(self.min_distance_to_obstacles, 0,  self.feature_all[:-1])

    def set_pyqt_signal_multirotor(self, action, reward):
        step = int(self.total_step)

        # transfer 2D state and action to 3D
        state = self.dynamic_model.state_raw
        if self.dynamic_model.navigation_3d:
            action_output = action
            state_output = state
        else:
            action_output = np.array([action[0], 0, action[1]])
            state_output = np.array([state[0], 0, state[2], state[3], 0, state[5]])

        self.action_signal.emit(step, action_output)
        self.state_signal.emit(step, state_output)

        # other values
        self.attitude_signal.emit(step, np.asarray(self.dynamic_model.get_attitude(
        )), np.asarray(self.dynamic_model.get_attitude_cmd()))
        self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.pose_signal.emit(np.asarray(self.dynamic_model.goal_position), np.asarray(
            self.dynamic_model.start_position), np.asarray(self.dynamic_model.get_position()), np.asarray(self.trajectory_list))

    def visual_log_q_value(self, q_value, action, reward):
        '''
        Create grid map (map_size = work_space)
        Log Q value and the best action in grid map
        At any grid position, record:
        1. Q value
        2. action 0
        3. action 1
        4. steps
        5. reward
        Save image every 10k steps
        Used only for 2D explanation
        '''

        # create init array if not exist
        map_size_x = self.work_space_x[1] - self.work_space_x[0]
        map_size_y = self.work_space_y[1] - self.work_space_y[0]
        if not hasattr(self, 'q_value_map'):
            self.q_value_map = np.full((9, map_size_x+1, map_size_y+1), np.nan)

        # record info
        position = self.dynamic_model.get_position()
        pose_x = position[0]
        pose_y = position[1]

        index_x = int(np.round(pose_x) + self.work_space_x[1])
        index_y = int(np.round(pose_y) + self.work_space_y[1])

        # check if index valid
        if index_x in range(0, map_size_x) and index_y in range(0, map_size_y):
            self.q_value_map[0, index_x, index_y] = q_value
            self.q_value_map[1, index_x, index_y] = action[0]
            self.q_value_map[2, index_x, index_y] = action[-1]
            self.q_value_map[3, index_x, index_y] = self.total_step
            self.q_value_map[4, index_x, index_y] = reward
            self.q_value_map[5, index_x, index_y] = q_value
            self.q_value_map[6, index_x, index_y] = action[0]
            self.q_value_map[7, index_x, index_y] = action[-1]
            self.q_value_map[8, index_x, index_y] = reward
        else:
            print(
                'Error: X:{} and Y:{} is outside of range 0~mapsize (visual_log_q_value)')

        # save array every record_step steps
        record_step = self.cfg.getint('options', 'q_map_save_steps')
        if (self.total_step+1) % record_step == 0:
            if self.data_path is not None:
                np.save(
                    self.data_path + '/q_value_map_{}'.format(self.total_step+1), self.q_value_map)
                # refresh 5 6 7 8 to record period data
                self.q_value_map[5, :, :] = np.nan
                self.q_value_map[6, :, :] = np.nan
                self.q_value_map[7, :, :] = np.nan
                self.q_value_map[8, :, :] = np.nan

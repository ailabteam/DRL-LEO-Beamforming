# satellite_env.py
# Phiên bản 6 - Final, Multi-Scenario (Đã sửa lỗi kiểu dữ liệu và rà soát)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

import config as cfg
from satellite import Satellite
from user import Users
from channel import calculate_fspl_db, calculate_elevation_angle, calculate_atmospheric_loss_db

class SatelliteEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, scenario="baseline"):
        super().__init__()
        
        if scenario not in ["baseline", "handover", "qos"]:
            raise ValueError(f"Kịch bản '{scenario}' không hợp lệ. Chỉ chấp nhận: 'baseline', 'handover', 'qos'.")
        
        self.scenario = scenario
        print(f"Khởi tạo môi trường SatelliteEnv (Scenario: {self.scenario.upper()})...")
        
        # --- 1. Định nghĩa Spaces ---
        self.action_space = spaces.Discrete(cfg.NUM_USERS)
        
        base_features = 2 + 3 * cfg.NUM_USERS
        qos_features = 2 + 4 * cfg.NUM_USERS
        num_state_features = qos_features if self.scenario == "qos" else base_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_state_features,), dtype=np.float32
        )

        # --- 2. Các giá trị không đổi ---
        bandwidth_hz = cfg.TOTAL_BANDWIDTH_MHZ * 1e6
        noise_power_w = cfg.BOLTZMANN_CONSTANT * cfg.SYSTEM_NOISE_TEMPERATURE_K * bandwidth_hz
        self.noise_power_dbw = 10 * np.log10(noise_power_w)
        
        # --- 3. Khởi tạo các thuộc tính ---
        self.satellite = None
        self.users = None
        self.current_step = 0
        self.time_since_last_served = np.zeros(cfg.NUM_USERS, dtype=np.float32)
        self.last_served_user = -1
        
        if self.scenario == "qos":
            self.user_queues_mbits = np.zeros(cfg.NUM_USERS, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.satellite = Satellite(cfg.SAT_INITIAL_POS_KM, cfg.SAT_VELOCITY_KM_S)
        self.users = Users(cfg.NUM_USERS, cfg.USER_DEPLOY_AREA_KM)
        
        self.current_step = 0
        self.time_since_last_served.fill(0)
        self.last_served_user = -1
        
        if self.scenario == "qos":
            self.user_queues_mbits.fill(0)
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self):
        snr_db = self._get_channel_state()
        sat_pos = self.satellite.get_position_km()
        normalized_sat_pos = sat_pos / cfg.SAT_ALTITUDE_KM
        traffic_requests = np.ones(cfg.NUM_USERS)
        normalized_time_since_served = self.time_since_last_served / 100.0
        
        obs_list = [
            normalized_sat_pos[:2],
            snr_db,
            traffic_requests,
            normalized_time_since_served
        ]
        
        if self.scenario == "qos":
            normalized_queues = self.user_queues_mbits / cfg.MAX_QUEUE_SIZE_MBITS
            obs_list.append(normalized_queues)
        
        obs = np.concatenate(obs_list).astype(np.float32)
        
        if obs.shape[0] != self.observation_space.shape[0]:
             raise ValueError(f"Shape of observation {obs.shape} does not match the space {self.observation_space.shape}")
        return obs

    def step(self, action):
        dropped_data_mbits = np.zeros(cfg.NUM_USERS)
        if self.scenario == "qos":
            avg_arrival_per_step = cfg.ARRIVAL_RATE_MBPS * cfg.TIME_STEP_S
            # SỬA LỖI: Poisson cần lam >= 0. Đảm bảo avg_arrival_per_step không âm.
            if avg_arrival_per_step > 0:
                data_arrival_mbits = np.random.poisson(avg_arrival_per_step, size=cfg.NUM_USERS)
                self.user_queues_mbits += data_arrival_mbits
            
            dropped_data_mbits = np.maximum(0, self.user_queues_mbits - cfg.MAX_QUEUE_SIZE_MBITS)
            self.user_queues_mbits = np.minimum(self.user_queues_mbits, cfg.MAX_QUEUE_SIZE_MBITS)

        snr_db_list = self._get_channel_state()
        selected_user_snr_db = snr_db_list[action]
        selected_user_snr_linear = 10**(selected_user_snr_db / 10)
        bandwidth_hz = cfg.TOTAL_BANDWIDTH_MHZ * 1e6
        potential_throughput_bps = bandwidth_hz * math.log2(1 + selected_user_snr_linear)
        potential_throughput_mbps = potential_throughput_bps / 1e6

        actual_throughput_mbps = potential_throughput_mbps
        if self.scenario == "qos":
            # Sửa lỗi: Cần đảm bảo self.user_queues_mbits[action] không âm trước khi chia
            if cfg.TIME_STEP_S > 0:
                max_sendable_mbps = self.user_queues_mbits[action] / cfg.TIME_STEP_S
                actual_throughput_mbps = min(potential_throughput_mbps, max_sendable_mbps)
            else:
                actual_throughput_mbps = 0

        if self.scenario == "qos":
            served_data_mbits = actual_throughput_mbps * cfg.TIME_STEP_S
            self.user_queues_mbits[action] = max(0, self.user_queues_mbits[action] - served_data_mbits)

        # --- PHẦN 4: TÍNH TOÁN REWARD - ĐÃ SỬA LỖI ---
        reward = 0.0 # Khởi tạo là float
        log_throughput_reward = math.log(1 + actual_throughput_mbps)

        if self.scenario == "baseline":
            reward = log_throughput_reward
        
        elif self.scenario == "handover":
            handover_penalty = 0.0
            if self.current_step > 0 and action != self.last_served_user:
                handover_penalty = 10.0
            reward = log_throughput_reward - handover_penalty
        
        elif self.scenario == "qos":
            W_THROUGHPUT = 1.0
            W_DROP = 10.0
            
            drop_penalty = np.sum(dropped_data_mbits)
            # SỬA LỖI: Ép kiểu kết quả cuối cùng thành float của Python
            reward = float(W_THROUGHPUT * log_throughput_reward - W_DROP * drop_penalty)

        self.last_served_user = action
        self.time_since_last_served += 1.0
        self.time_since_last_served[action] = 0.0
        self.satellite.move(cfg.TIME_STEP_S)
        self.current_step += 1
        
        terminated = self.current_step >= int(cfg.SIM_DURATION_S / cfg.TIME_STEP_S)
        observation = self._get_obs()
        info = self._get_info()
        
        # SỬA LỖI: Đảm bảo reward luôn là float
        return observation, float(reward), terminated, False, info

    def _get_channel_state(self):
        sat_pos = self.satellite.get_position_km()
        user_positions = self.users.get_positions_km()
        distances = np.linalg.norm(sat_pos - user_positions, axis=1)
        elevation = calculate_elevation_angle(sat_pos, user_positions)
        fspl_db = calculate_fspl_db(distances, cfg.CENTER_FREQUENCY_GHZ)
        atmos_loss_db = calculate_atmospheric_loss_db(elevation)
        total_loss_db = fspl_db + atmos_loss_db
        power_tx_dbw = 10 * np.log10(cfg.TOTAL_POWER_W)
        power_rx_dbw = (power_tx_dbw + cfg.SAT_ANTENNA_GAIN_DB + cfg.USER_ANTENNA_GAIN_DB - total_loss_db)
        snr_db_list = power_rx_dbw - self.noise_power_dbw
        return snr_db_list

    def _get_info(self):
        # Có thể mở rộng info cho kịch bản QoS để debug
        info_dict = {"snr_db": self._get_channel_state()}
        if self.scenario == "qos":
            info_dict["queues_mbits"] = self.user_queues_mbits
        return info_dict
    
    def render(self): pass
    def close(self): pass

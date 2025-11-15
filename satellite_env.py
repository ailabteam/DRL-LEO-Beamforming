# satellite_env.py (đổi tên từ satellite_env_multiscenario.py nếu cần)
# Phiên bản 4 - Hỗ trợ đa kịch bản (Multi-Scenario)

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

    # Sửa hàm __init__ để chấp nhận tham số 'scenario'
    def __init__(self, scenario="baseline"):
        super().__init__()
        
        # Kiểm tra xem kịch bản có hợp lệ không
        if scenario not in ["baseline", "handover", "qos"]:
            raise ValueError(f"Kịch bản '{scenario}' không hợp lệ. Chỉ chấp nhận: 'baseline', 'handover', 'qos'.")
        
        self.scenario = scenario
        print(f"Khởi tạo môi trường SatelliteEnv (Scenario: {self.scenario.upper()})...")
        
        # --- Định nghĩa Spaces ---
        self.action_space = spaces.Discrete(cfg.NUM_USERS)
        
        # State space vẫn giữ nguyên như phiên bản v3
        num_state_features = 2 + 3 * cfg.NUM_USERS
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_state_features,), dtype=np.float32
        )

        # --- Các giá trị không đổi ---
        bandwidth_hz = cfg.TOTAL_BANDWIDTH_MHZ * 1e6
        noise_power_w = cfg.BOLTZMANN_CONSTANT * cfg.SYSTEM_NOISE_TEMPERATURE_K * bandwidth_hz
        self.noise_power_dbw = 10 * np.log10(noise_power_w)
        
        # --- Khởi tạo các thuộc tính ---
        self.satellite = None
        self.users = None
        self.current_step = 0
        self.time_since_last_served = np.zeros(cfg.NUM_USERS, dtype=np.float32)
        
        # Thuộc tính riêng cho kịch bản handover
        self.last_served_user = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.satellite = Satellite(cfg.SAT_INITIAL_POS_KM, cfg.SAT_VELOCITY_KM_S)
        self.users = Users(cfg.NUM_USERS, cfg.USER_DEPLOY_AREA_KM)
        
        self.current_step = 0
        self.time_since_last_served.fill(0)
        self.last_served_user = -1 # Reset cho mọi kịch bản
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self):
        # ... (Hàm này giữ nguyên, không thay đổi so với phiên bản v3) ...
        snr_db = self._get_channel_state()
        sat_pos = self.satellite.get_position_km()
        normalized_sat_pos = sat_pos / cfg.SAT_ALTITUDE_KM
        traffic_requests = np.ones(cfg.NUM_USERS)
        normalized_time_since_served = self.time_since_last_served / 100.0
        
        obs = np.concatenate([
            normalized_sat_pos[:2],
            snr_db,
            traffic_requests,
            normalized_time_since_served
        ]).astype(np.float32)
        
        if obs.shape != self.observation_space.shape:
             raise ValueError(f"Shape of observation {obs.shape} does not match the space {self.observation_space.shape}")
        return obs

    def step(self, action):
        # Tính toán thông lượng (chung cho tất cả các kịch bản)
        snr_db_list = self._get_channel_state()
        selected_user_snr_db = snr_db_list[action]
        selected_user_snr_linear = 10**(selected_user_snr_db / 10)
        bandwidth_hz = cfg.TOTAL_BANDWIDTH_MHZ * 1e6
        throughput_bps = bandwidth_hz * math.log2(1 + selected_user_snr_linear)
        throughput_mbps = throughput_bps / 1e6
        
        # --- TÍNH TOÁN REWARD DỰA TRÊN KỊCH BẢN ---
        throughput_reward = math.log(1 + throughput_mbps)
        
        if self.scenario == "baseline":
            reward = throughput_reward
        
        elif self.scenario == "handover":
            handover_penalty = 0.0
            HANDOVER_PENALTY_VALUE = 10.0
            
            if self.current_step > 0 and action != self.last_served_user:
                handover_penalty = HANDOVER_PENALTY_VALUE
            
            reward = throughput_reward - handover_penalty
        
        else: # Mặc định hoặc các kịch bản khác
            reward = throughput_reward
            
        # Cập nhật trạng thái chung
        self.last_served_user = action # Cập nhật cho cả 2 kịch bản
        self.time_since_last_served += 1.0
        self.time_since_last_served[action] = 0.0
        self.satellite.move(cfg.TIME_STEP_S)
        self.current_step += 1
        
        terminated = self.current_step >= int(cfg.SIM_DURATION_S / cfg.TIME_STEP_S)
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info

    # --- Các hàm còn lại không thay đổi ---
    def _get_channel_state(self):
        # ... (giữ nguyên)
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
        return {"snr_db": self._get_channel_state()}
        
    def render(self): pass
    def close(self): pass

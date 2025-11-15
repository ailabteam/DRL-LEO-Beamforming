# satellite_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

# Import các thành phần mô phỏng chúng ta đã xây dựng
import config as cfg
from satellite import Satellite
from user import Users
from channel import calculate_fspl_db, calculate_elevation_angle, calculate_atmospheric_loss_db

class SatelliteEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self):
        super().__init__()
        print("Khởi tạo môi trường SatelliteEnv...")
        
        # --- 1. Định nghĩa không gian Hành động (Action Space) ---
        # Hành động: Chọn 1 trong N người dùng để phục vụ
        self.action_space = spaces.Discrete(cfg.NUM_USERS)

        # --- 2. Định nghĩa không gian Quan sát (Observation Space) ---
        # State vector bao gồm:
        # - Vị trí x, y của vệ tinh (chuẩn hóa) (2 features)
        # - SNR (dB) của mỗi người dùng (NUM_USERS features)
        # - Yêu cầu lưu lượng của mỗi người dùng (tạm thời giả định là 1) (NUM_USERS features)
        num_state_features = 2 + 2 * cfg.NUM_USERS
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_state_features,), dtype=np.float32
        )

        # --- 3. Tính toán các giá trị không đổi ---
        # Băng thông tính bằng Hz
        bandwidth_hz = cfg.TOTAL_BANDWIDTH_MHZ * 1e6
        # Công suất nhiễu nhiệt (W)
        noise_power_w = cfg.BOLTZMANN_CONSTANT * cfg.SYSTEM_NOISE_TEMPERATURE_K * bandwidth_hz
        # Chuyển sang dBW
        self.noise_power_dbw = 10 * np.log10(noise_power_w)
        
        # --- 4. Khởi tạo các thành phần mô phỏng ---
        self.satellite = None
        self.users = None
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Khởi tạo lại vệ tinh và người dùng cho episode mới
        self.satellite = Satellite(cfg.SAT_INITIAL_POS_KM, cfg.SAT_VELOCITY_KM_S)
        self.users = Users(cfg.NUM_USERS, cfg.USER_DEPLOY_AREA_KM)
        
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def _get_channel_state(self):
        """
        Hàm helper để tính toán trạng thái kênh cho tất cả người dùng.
        Trả về: snr_db_list (mảng SNR tính bằng dB cho mỗi người dùng)
        """
        sat_pos = self.satellite.get_position_km()
        user_positions = self.users.get_positions_km()
        
        # Tính toán hình học
        distances = np.linalg.norm(sat_pos - user_positions, axis=1)
        elevation = calculate_elevation_angle(sat_pos, user_positions)
        
        # Tính toán suy hao
        fspl_db = calculate_fspl_db(distances, cfg.CENTER_FREQUENCY_GHZ)
        atmos_loss_db = calculate_atmospheric_loss_db(elevation)
        total_loss_db = fspl_db + atmos_loss_db
        
        # Tính công suất tín hiệu nhận được (dBW)
        # P_rx_dbw = P_tx_dbw + G_tx_db + G_rx_db - Loss_db
        power_tx_dbw = 10 * np.log10(cfg.TOTAL_POWER_W)
        power_rx_dbw = (power_tx_dbw + cfg.SAT_ANTENNA_GAIN_DB + 
                        cfg.USER_ANTENNA_GAIN_DB - total_loss_db)
        
        # Tính SNR (dB)
        # SNR_db = P_rx_dbw - N_dbw
        snr_db_list = power_rx_dbw - self.noise_power_dbw
        
        return snr_db_list

    def _get_obs(self):
        """
        Thu thập và trả về observation hiện tại cho agent.
        """
        # Lấy SNR của tất cả người dùng
        snr_db = self._get_channel_state()
        
        # Lấy vị trí vệ tinh và chuẩn hóa (normalize)
        # Chuẩn hóa giúp mạng neural học tốt hơn. Ở đây ta chia cho độ cao.
        sat_pos = self.satellite.get_position_km()
        normalized_sat_pos = sat_pos / cfg.SAT_ALTITUDE_KM
        
        # Giả định yêu cầu lưu lượng của mỗi user là 1 (sẽ nâng cấp sau)
        traffic_requests = np.ones(cfg.NUM_USERS)
        
        # Ghép tất cả lại thành một vector state duy nhất
        obs = np.concatenate([
            normalized_sat_pos[:2], # Chỉ lấy x, y
            snr_db,
            traffic_requests
        ]).astype(np.float32)
        
        return obs

    def _get_info(self):
        """
        Trả về thông tin phụ (để debug hoặc log).
        """
        snr_db = self._get_channel_state()
        return {"snr_db": snr_db}
        
    def step(self, action):
        """
        Thực thi một hành động (chọn người dùng `action` để phục vụ).
        """
        # Lấy trạng thái kênh trước khi hành động
        snr_db_list = self._get_channel_state()
        
        # Lấy SNR của người dùng được chọn bởi agent
        selected_user_snr_db = snr_db_list[action]
        
        # --- 1. Tính toán Reward ---
        # Chuyển SNR từ dB sang dạng tuyến tính (linear scale)
        selected_user_snr_linear = 10**(selected_user_snr_db / 10)
        
        # Tính thông lượng (bps) theo công thức Shannon-Hartley
        # C = B * log2(1 + SNR)
        bandwidth_hz = cfg.TOTAL_BANDWIDTH_MHZ * 1e6
        throughput_bps = bandwidth_hz * math.log2(1 + selected_user_snr_linear)
        
        # Reward = thông lượng tính bằng Mbps
        reward = throughput_bps / 1e6
        
        # --- 2. Cập nhật trạng thái thế giới ---
        self.satellite.move(cfg.TIME_STEP_S)
        self.current_step += 1
        
        # --- 3. Kiểm tra điều kiện kết thúc episode ---
        terminated = self.current_step >= int(cfg.SIM_DURATION_S / cfg.TIME_STEP_S)
        
        # Lấy observation và info mới
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info

    # Các hàm render và close giữ nguyên (chưa cần implement)
    def render(self):
        pass

    def close(self):
        pass

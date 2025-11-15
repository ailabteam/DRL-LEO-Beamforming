# train.py
import os
import sys
from datetime import datetime
from stable_baselines3 import PPO
from satellite_env import SatelliteEnv # Giả sử env đã được cập nhật để hỗ trợ đa kịch bản

def train_agent(scenario, total_timesteps, log_base_path):
    """
    Hàm để huấn luyện agent cho một kịch bản cụ thể.
    """
    print(f"--- Bắt đầu huấn luyện cho Kịch bản: {scenario.upper()} ---")

    # --- 1. Tạo thư mục lưu trữ duy nhất cho lần chạy này ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_base_path, f"{scenario}_{timestamp}")
    model_save_path = os.path.join(log_dir, f"ppo_{scenario}_model")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Thư mục lưu trữ: {log_dir}")

    # --- 2. Khởi tạo môi trường ---
    # Chúng ta sẽ sử dụng môi trường v3 (state có time_since_last_served)
    # và hàm reward log đơn giản cho kịch bản baseline này.
    # Đảm bảo satellite_env.py của bạn đang ở phiên bản đó.
    env = SatelliteEnv() 

    # --- 3. Khởi tạo PPO Agent ---
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cuda")

    # --- 4. Bắt đầu huấn luyện ---
    print(f"\nHuấn luyện với {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    print("\n--- Huấn luyện hoàn tất ---")
    
    # --- 5. Lưu lại model ---
    model.save(model_save_path)
    print(f"Model đã được lưu tại: {model_save_path}.zip")
    print("\nĐể trực quan hóa, chạy lệnh:")
    print(f"tensorboard --logdir {os.path.dirname(log_dir)}") # Chỉ vào thư mục cha

if __name__ == "__main__":
    # Kịch bản mặc định là 'baseline'
    scenario_to_run = "baseline"
    # Số bước huấn luyện
    timesteps_to_run = 100_000 # Giữ ở mức thấp để chạy nhanh kịch bản baseline
    # Đường dẫn thư mục gốc để lưu kết quả
    results_path = os.path.join("results", f"scenario_{scenario_to_run}")
    
    # Chạy hàm huấn luyện
    train_agent(scenario_to_run, timesteps_to_run, results_path)

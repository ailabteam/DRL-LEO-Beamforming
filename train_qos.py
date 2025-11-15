# train_handover.py
import os
from datetime import datetime
from stable_baselines3 import PPO

# Đổi tên file import nếu bạn đã đổi
from satellite_env import SatelliteEnv

def train_agent(scenario, total_timesteps, log_base_path):
    """
    Hàm để huấn luyện agent cho một kịch bản cụ thể.
    """
    print(f"--- Bắt đầu huấn luyện cho Kịch bản: {scenario.upper()} ---")

    # --- 1. Tạo thư mục lưu trữ duy nhất ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_base_path, f"{scenario}_{timestamp}")
    model_save_path = os.path.join(log_dir, f"ppo_{scenario}_model")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Thư mục lưu trữ: {log_dir}")

    # --- 2. Khởi tạo môi trường ---
    # KHỞI TẠO VỚI KỊCH BẢN "handover"
    env = SatelliteEnv(scenario=scenario)

    # --- 3. Khởi tạo PPO Agent ---
    # Chúng ta có thể dùng lại các siêu tham số mặc định
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cuda")

    # --- 4. Bắt đầu huấn luyện ---
    print(f"\nHuấn luyện với {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    print("\n--- Huấn luyện hoàn tất ---")

    # --- 5. Lưu lại model ---
    model.save(model_save_path)
    print(f"Model đã được lưu tại: {model_save_path}.zip")
    print("\nĐể trực quan hóa, chạy lệnh:")
    print(f"tensorboard --logdir {os.path.dirname(log_dir)}")

if __name__ == "__main__":
    scenario_to_run = "qos"
    # Bài toán khó hơn, cần nhiều thời gian học hơn
    timesteps_to_run = 300_000
    results_path = os.path.join("results", f"scenario_{scenario_to_run}")

    train_agent(scenario_to_run, timesteps_to_run, results_path)

# train.py
import os
from datetime import datetime
from stable_baselines3 import PPO
from satellite_env import SatelliteEnv

def main():
    """
    Hàm chính để huấn luyện agent DRL.
    """
    print("--- Bắt đầu quá trình huấn luyện DRL ---")

    # --- 1. Tạo thư mục để lưu trữ kết quả ---
    # Lưu model và log vào một thư mục có timestamp để dễ quản lý
    log_dir = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_save_path = f"{log_dir}/ppo_satellite_model"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Thư mục lưu trữ: {log_dir}")

    # --- 2. Khởi tạo môi trường ---
    env = SatelliteEnv()
    
    # --- 3. Khởi tạo PPO Agent ---
    # "MlpPolicy": Sử dụng mạng Multi-Layer Perceptron (MLP) làm policy network.
    # env: Môi trường mà agent sẽ tương tác.
    # verbose=1: In ra thông tin huấn luyện (reward, loss, etc.).
    # tensorboard_log: Thư mục để lưu log cho TensorBoard (công cụ trực quan hóa).
    # device="cuda": Yêu cầu Stable-Baselines3 sử dụng GPU.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        device="cuda"
    )

    # --- 4. Bắt đầu huấn luyện ---
    # total_timesteps: Tổng số bước (step) mà agent sẽ tương tác với môi trường.
    # 100,000 là một con số nhỏ để chạy thử, quá trình huấn luyện thực sự cần hàng triệu bước.
    total_timesteps = 1_000_000 
    print(f"\nBắt đầu huấn luyện với {total_timesteps} timesteps...")
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True # Hiển thị thanh tiến trình
    )
    
    print("\n--- Huấn luyện hoàn tất ---")
    
    # --- 5. Lưu lại model đã huấn luyện ---
    model.save(model_save_path)
    print(f"Model đã được lưu tại: {model_save_path}.zip")
    
    # --- Hướng dẫn xem kết quả ---
    print("\nĐể trực quan hóa quá trình học, hãy chạy lệnh sau trong terminal:")
    print(f"tensorboard --logdir {log_dir}")

if __name__ == "__main__":
    main()

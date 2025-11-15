# evaluate.py
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from satellite_env import SatelliteEnv

def evaluate_agent(env, model=None, strategy="drl"):
    """
    Chạy một episode và trả về tổng reward.

    Args:
        env (gym.Env): Môi trường đã được khởi tạo.
        model (BaseAlgorithm): Model DRL đã được huấn luyện (nếu có).
        strategy (str): "drl", "random", or "greedy".
    """
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        if strategy == "drl":
            action, _ = model.predict(obs, deterministic=True)
        elif strategy == "random":
            action = env.action_space.sample()
        elif strategy == "greedy":
            # Lấy thông tin SNR từ môi trường
            snr_db = info.get("snr_db", np.zeros(env.action_space.n))
            action = np.argmax(snr_db) # Chọn user có SNR cao nhất

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return total_reward

def main():
    print("--- Bắt đầu quá trình Đánh giá ---")

    # --- 1. Tải model đã huấn luyện ---
    # Hãy đảm bảo đường dẫn này chính xác với file model của bạn
    # Ví dụ: model_path = "logs/20251115_183000/ppo_satellite_model.zip"
    # TẠM THỜI ĐỂ TRỐNG, BẠN SẼ ĐIỀN VÀO SAU
    model_path = "logs/20251115_163013/ppo_satellite_model.zip"

    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        print("Vui lòng kiểm tra lại đường dẫn trong file evaluate.py")
        return

    # --- 2. Khởi tạo môi trường ---
    # Quan trọng: chúng ta sẽ dùng cùng một môi trường cho tất cả các agent
    # để đảm bảo so sánh công bằng.
    eval_env = SatelliteEnv()

    # --- 3. Chạy đánh giá cho từng chiến lược ---
    # Để đảm bảo các agent đối mặt với cùng kịch bản (vị trí user),
    # chúng ta sẽ reset môi trường với cùng một seed trước mỗi lần chạy.
    common_seed = 42

    print("\nĐánh giá DRL Agent (PPO)...")
    eval_env.reset(seed=common_seed)
    drl_reward = evaluate_agent(eval_env, model, strategy="drl")

    print("Đánh giá Random Agent...")
    eval_env.reset(seed=common_seed)
    random_reward = evaluate_agent(eval_env, strategy="random")

    print("Đánh giá Greedy (Max-SNR) Agent...")
    eval_env.reset(seed=common_seed)
    greedy_reward = evaluate_agent(eval_env, strategy="greedy")

    # --- 4. In và Trực quan hóa kết quả ---
    print("\n--- Kết quả Đánh giá ---")
    print(f"Tổng reward của Random Agent:   {random_reward:.2f} Mbps")
    print(f"Tổng reward của Greedy Agent:   {greedy_reward:.2f} Mbps")
    print(f"Tổng reward của DRL Agent:      {drl_reward:.2f} Mbps")

    # Vẽ đồ thị cột
    strategies = ['Random', 'Greedy (Max-SNR)', 'DRL (PPO)']
    rewards = [random_reward, greedy_reward, drl_reward]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategies, rewards, color=['lightcoral', 'gold', 'lightgreen'])
    plt.ylabel('Total Reward (Cumulative Mbps over Episode)')
    plt.title('Performance Comparison of Different Agents')
    plt.xticks(rotation=7)

    # Thêm giá trị lên trên mỗi cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig("evaluation_comparison.png", dpi=600)
    print("\nĐã lưu đồ thị so sánh vào file: evaluation_comparison.png")

if __name__ == "__main__":
    main()

# evaluate_handover.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from stable_baselines3 import PPO

import config as cfg
# Đảm bảo import đúng file môi trường hỗ trợ đa kịch bản
from satellite_env import SatelliteEnv 

def jains_fairness_index(throughputs):
    """
    Tính chỉ số công bằng Jain's Fairness Index.
    """
    if np.sum(throughputs) == 0:
        return 0.0
    
    sum_of_throughputs = np.sum(throughputs)
    sum_of_squared_throughputs = np.sum(throughputs**2)
    
    # Thêm kiểm tra để tránh chia cho 0 nếu tất cả throughput là 0
    if sum_of_squared_throughputs == 0:
        return 1.0

    fairness_index = (sum_of_throughputs**2) / (len(throughputs) * sum_of_squared_throughputs)
    return fairness_index

def evaluate_agent(env, model=None, strategy="drl"):
    """
    Chạy một episode và trả về các chỉ số hiệu năng (Thông lượng và Công bằng).
    """
    obs, info = env.reset()
    done = False
    user_throughputs = np.zeros(cfg.NUM_USERS)

    while not done:
        # --- Logic chọn action ---
        action = None
        if strategy == "drl":
            action, _ = model.predict(obs, deterministic=True)
        elif strategy == "random":
            action = env.action_space.sample()
        elif strategy == "greedy":
            # Trích xuất SNR từ observation
            snr_db_from_obs = obs[2 : 2 + cfg.NUM_USERS]
            action = np.argmax(snr_db_from_obs)
        else:
            raise ValueError(f"Chiến lược không xác định: {strategy}")

        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Tính throughput thực tế của bước này ---
        snr_db_current = info.get("snr_db")
        if snr_db_current is None:
            # Fallback (không nên xảy ra)
            snr_db_current = obs[2 : 2 + cfg.NUM_USERS]
            
        selected_user_snr_linear = 10**(snr_db_current[action] / 10)
        bandwidth_hz = cfg.TOTAL_BANDWIDTH_MHZ * 1e6
        throughput_this_step = (bandwidth_hz * math.log2(1 + selected_user_snr_linear)) / 1e6 # Mbps
        user_throughputs[action] += throughput_this_step
        
        done = terminated or truncated

    total_throughput = np.sum(user_throughputs)
    fairness = jains_fairness_index(user_throughputs)
    return total_throughput, fairness

def run_evaluation(model_path, output_dir):
    """Hàm chính để chạy và lưu kết quả đánh giá."""
    print(f"--- Bắt đầu Đánh giá cho model: {model_path} ---")
    
    scenario = os.path.basename(output_dir).replace('scenario_', '')
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return
        
    eval_env = SatelliteEnv(scenario=scenario)
    common_seed = 42
    
    results = {}
    strategies = {"DRL (PPO)": model, "Random": None, "Greedy (Max-SNR)": None}
    for name, agent_model in strategies.items():
        print(f"Đánh giá {name}...")
        eval_env.reset(seed=common_seed)
        strategy_type = "drl" if name == "DRL (PPO)" else ("random" if name == "Random" else "greedy")
        throughput, fairness = evaluate_agent(eval_env, agent_model, strategy=strategy_type)
        results[name] = {"Total Throughput (Mbps)": throughput, "Jain's Fairness Index": fairness}
        
    # --- Lưu kết quả ---
    df = pd.DataFrame.from_dict(results, orient='index')
    csv_path = os.path.join(output_dir, "evaluation_data.csv")
    df.to_csv(csv_path)
    print("\n--- Kết quả Đánh giá ---")
    print(df)
    print(f"\nĐã lưu dữ liệu dạng bảng vào: {csv_path}")
    
    fig_path = os.path.join(output_dir, "evaluation_comparison.png")
    df.plot(kind='bar', subplots=True, figsize=(15, 6), layout=(1, 2), legend=False, rot=0)
    plt.suptitle(f'Performance Comparison - Scenario: {scenario.upper()}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_path, dpi=600)
    print(f"Đã lưu đồ thị so sánh vào: {fig_path}")

if __name__ == "__main__":
    scenario_to_eval = "handover"
    result_dir = os.path.join("results", f"scenario_{scenario_to_eval}")
    
    try:
        log_dirs = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
        latest_log_dir = sorted(log_dirs)[-1]
        model_files = [f for f in os.listdir(os.path.join(result_dir, latest_log_dir)) if f.endswith('.zip')]
        latest_model_path = os.path.join(result_dir, latest_log_dir, model_files[0])
        
        run_evaluation(latest_model_path, result_dir)
        
    except (IndexError, FileNotFoundError) as e:
        print(f"Lỗi: Không tìm thấy model để đánh giá trong thư mục '{result_dir}'.")
        print("Hãy chạy train_handover.py trước.")

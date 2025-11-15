# evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
from stable_baselines3 import PPO

import config as cfg
# Sửa đổi để import env có hỗ trợ đa kịch bản
from satellite_env_multiscenario import SatelliteEnv

def jains_fairness_index(throughputs):
    # ... (hàm này giữ nguyên) ...
    if np.sum(throughputs) == 0: return 0.0
    sum_sq = np.sum(throughputs**2)
    if sum_sq == 0: return 1.0
    return (np.sum(throughputs)**2) / (len(throughputs) * sum_sq)

# evaluate.py -> chỉ thay thế hàm evaluate_agent

def evaluate_agent(env, model=None, strategy="drl"):
    """
    Chạy một episode và trả về các chỉ số hiệu năng.
    """
    obs, info = env.reset()
    done = False
    user_throughputs = np.zeros(cfg.NUM_USERS)

    while not done:
        # --- LOGIC CHỌN ACTION ĐÃ SỬA LỖI ---
        action = None
        if strategy == "drl":
            action, _ = model.predict(obs, deterministic=True)
        elif strategy == "random":
            action = env.action_space.sample()
        elif strategy == "greedy":
            # Lấy SNR từ observation `obs` thay vì `info`
            # State của chúng ta có cấu trúc: [sat_x, sat_y, snr_0, ..., snr_N-1, ...]
            # Do đó, SNR bắt đầu từ chỉ số 2 và có N_USERS phần tử.
            snr_db_from_obs = obs[2 : 2 + cfg.NUM_USERS]
            action = np.argmax(snr_db_from_obs)
        else:
            raise ValueError(f"Chiến lược không xác định: {strategy}")
        # -----------------------------------

        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- LOGIC TÍNH THROUGHPUT ĐÃ SỬA LỖI ---
        # `info` bây giờ chứa SNR cập nhật chính xác nhất cho bước vừa rồi
        snr_db_current = info.get("snr_db")
        if snr_db_current is None:
            # Fallback nếu info không có snr_db (không nên xảy ra)
            snr_db_current = obs[2 : 2 + cfg.NUM_USERS]
            
        selected_user_snr_linear = 10**(snr_db_current[action] / 10)
        bandwidth_hz = cfg.TOTAL_BANDWIDTH_MHZ * 1e6
        throughput_this_step = (bandwidth_hz * math.log2(1 + selected_user_snr_linear)) / 1e6 # Mbps
        user_throughputs[action] += throughput_this_step
        # ----------------------------------------
        
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

    # Chạy các chiến lược
    results = {}
    strategies = {"DRL (PPO)": model, "Random": None, "Greedy (Max-SNR)": None}
    for name, agent_model in strategies.items():
        print(f"Đánh giá {name}...")
        eval_env.reset(seed=common_seed)
        strategy_type = "drl" if name == "DRL (PPO)" else ("random" if name == "Random" else "greedy")
        throughput, fairness = evaluate_agent(eval_env, agent_model, strategy=strategy_type)
        results[name] = {"Total Throughput (Mbps)": throughput, "Jain's Fairness Index": fairness}

    # --- Lưu kết quả ---
    # 1. Dạng số (CSV)
    df = pd.DataFrame.from_dict(results, orient='index')
    csv_path = os.path.join(output_dir, "evaluation_data.csv")
    df.to_csv(csv_path)
    print("\n--- Kết quả Đánh giá ---")
    print(df)
    print(f"\nĐã lưu dữ liệu dạng bảng vào: {csv_path}")

    # 2. Dạng đồ thị
    fig_path = os.path.join(output_dir, "evaluation_comparison.png")
    df.plot(kind='bar', subplots=True, figsize=(15, 6), layout=(1, 2), legend=False, rot=0)
    plt.suptitle(f'Performance Comparison - Scenario: {scenario.upper()}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_path, dpi=600)
    print(f"Đã lưu đồ thị so sánh vào: {fig_path}")

if __name__ == "__main__":
    # Lấy đường dẫn model từ lần chạy huấn luyện gần nhất
    # Đây là phần cần tự động hóa hoặc chỉ định thủ công
    # Ví dụ:
    scenario_to_eval = "baseline"
    result_dir = os.path.join("results", f"scenario_{scenario_to_eval}")

    # Tìm file model .zip mới nhất trong thư mục log
    try:
        log_dirs = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
        latest_log_dir = sorted(log_dirs)[-1]
        model_files = [f for f in os.listdir(os.path.join(result_dir, latest_log_dir)) if f.endswith('.zip')]
        latest_model_path = os.path.join(result_dir, latest_log_dir, model_files[0])

        run_evaluation(latest_model_path, result_dir)

    except (IndexError, FileNotFoundError) as e:
        print(f"Lỗi: Không tìm thấy model để đánh giá trong thư mục '{result_dir}'.")
        print("Hãy chạy train.py trước.")

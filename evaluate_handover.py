# evaluate_handover.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from stable_baselines3 import PPO

import config as cfg
from satellite_env import SatelliteEnv 

# --- HÀM jains_fairness_index và evaluate_agent giữ nguyên như phiên bản trước ---
# (Bạn có thể copy-paste chúng vào đây)
def jains_fairness_index(throughputs):
    # ... (code giữ nguyên)
def evaluate_agent(env, model=None, strategy="drl"):
    # ... (code giữ nguyên)
# ---

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

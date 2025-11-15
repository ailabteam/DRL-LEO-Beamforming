# test_simulation.py
import time
import numpy as np

# Import các class và config
import config as cfg
from satellite import Satellite
from user import Users
from channel import calculate_fspl_db  # <-- DÒNG MỚI: Import hàm tính FSPL

def main():
    print("--- Bắt đầu kịch bản mô phỏng TÍCH HỢP ---")

    # 1. Khởi tạo các đối tượng
    leo_satellite = Satellite(cfg.SAT_INITIAL_POS_KM, cfg.SAT_VELOCITY_KM_S)
    ground_users = Users(cfg.NUM_USERS, cfg.USER_DEPLOY_AREA_KM)

    # Lấy vị trí của tất cả người dùng một lần
    user_positions = ground_users.get_positions_km()

    # 2. Chạy vòng lặp mô phỏng
    total_steps = int(cfg.SIM_DURATION_S / cfg.TIME_STEP_S)
    print(f"\nTổng số bước mô phỏng: {total_steps}")

    # Chuẩn bị để lưu kết quả
    results = []

    for step in range(total_steps):
        # Cập nhật vị trí vệ tinh
        leo_satellite.move(cfg.TIME_STEP_S)
        sat_pos = leo_satellite.get_position_km()

        # --- LOGIC TÍNH TOÁN MỚI ---
        # Tính khoảng cách từ vệ tinh đến TẤT CẢ người dùng cùng lúc
        # Sử dụng broadcasting của numpy để tính toán hiệu quả
        distances_to_users = np.linalg.norm(sat_pos - user_positions, axis=1)

        # Tính FSPL đến TẤT CẢ người dùng
        fspl_values_db = calculate_fspl_db(distances_to_users, cfg.CENTER_FREQUENCY_GHZ)
        # ---------------------------

        # Lấy thông tin của người dùng đầu tiên (user 0) để in ra
        dist_to_user_0 = distances_to_users[0]
        fspl_to_user_0 = fspl_values_db[0]

        # Lưu kết quả của bước này
        results.append({
            "step": step,
            "sat_pos_x": sat_pos[0],
            "dist_user0": dist_to_user_0,
            "fspl_user0": fspl_to_user_0
        })

        # In thông tin mỗi 100 bước để theo dõi
        if step % 100 == 0:
            print(f"Step {step:4d}: Sat_x={sat_pos[0]:.2f} km | "
                  f"Dist_U0={dist_to_user_0:.2f} km | FSPL_U0={fspl_to_user_0:.2f} dB")

    print("\n--- Mô phỏng kết thúc ---")
    final_sat_pos = leo_satellite.get_position_km()
    print(f"Vị trí cuối cùng của vệ tinh: [{final_sat_pos[0]:.2f}, {final_sat_pos[1]:.2f}, {final_sat_pos[2]:.2f}] km")

    # --- VẼ ĐỒ THỊ ---
    # Chúng ta sẽ thêm phần vẽ đồ thị ở đây trong bước tiếp theo
    from plotter import plot_simulation_results
    if results:  # Chỉ vẽ nếu list results có dữ liệu
        plot_simulation_results(results)


if __name__ == "__main__":
    main()

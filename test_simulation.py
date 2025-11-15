# test_simulation.py
import time
import numpy as np

# Import các class và config chúng ta vừa tạo
import config as cfg
from satellite import Satellite
from user import Users

def main():
    print("--- Bắt đầu kịch bản mô phỏng đơn giản ---")

    # 1. Khởi tạo các đối tượng từ config
    leo_satellite = Satellite(cfg.SAT_INITIAL_POS_KM, cfg.SAT_VELOCITY_KM_S)
    ground_users = Users(cfg.NUM_USERS, cfg.USER_DEPLOY_AREA_KM)

    # 2. Chạy vòng lặp mô phỏng
    total_steps = int(cfg.SIM_DURATION_S / cfg.TIME_STEP_S)
    print(f"\nTổng số bước mô phỏng: {total_steps}")
    
    for step in range(total_steps):
        # Cập nhật vị trí vệ tinh
        leo_satellite.move(cfg.TIME_STEP_S)
        
        # Lấy vị trí hiện tại
        sat_pos = leo_satellite.get_position_km()
        user_positions = ground_users.get_positions_km()
        
        # Tính khoảng cách từ vệ tinh đến người dùng đầu tiên
        dist_to_user_0 = np.linalg.norm(sat_pos - user_positions[0])
        
        # In thông tin mỗi 100 bước để theo dõi
        if step % 100 == 0:
            print(f"Step {step:4d}: Vị trí Sat [x,y,z] = [{sat_pos[0]:.2f}, {sat_pos[1]:.2f}, {sat_pos[2]:.2f}] km. "
                  f"Khoảng cách đến User 0: {dist_to_user_0:.2f} km")

    print("\n--- Mô phỏng kết thúc ---")
    final_sat_pos = leo_satellite.get_position_km()
    print(f"Vị trí cuối cùng của vệ tinh: [{final_sat_pos[0]:.2f}, {final_sat_pos[1]:.2f}, {final_sat_pos[2]:.2f}] km")

if __name__ == "__main__":
    main()

# check_env.py
from stable_baselines3.common.env_checker import check_env
from satellite_env import SatelliteEnv

# Khởi tạo môi trường
env = SatelliteEnv()

# Chạy công cụ kiểm tra
# Nếu không có lỗi nào được in ra, môi trường của bạn đã hợp lệ!
try:
    check_env(env)
    print("\n[SUCCESS] Môi trường SatelliteEnv đã vượt qua kiểm tra của Stable-Baselines3!")
except Exception as e:
    print(f"\n[ERROR] Có lỗi xảy ra trong quá trình kiểm tra môi trường: {e}")

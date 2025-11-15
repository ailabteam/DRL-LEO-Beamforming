# config.py

# ============================================
# Simulation Parameters
# ============================================
SIM_DURATION_S = 100.0       # (seconds) Tổng thời gian mô phỏng
TIME_STEP_S = 0.1          # (seconds) Mỗi bước mô phỏng dài 100ms
# -> Tổng cộng sẽ có SIM_DURATION_S / TIME_STEP_S = 1000 bước

# ============================================
# Satellite Parameters
# ============================================
SAT_ALTITUDE_KM = 550.0      # (km) Độ cao của vệ tinh LEO
SAT_VELOCITY_KM_S = 7.5      # (km/s) Vận tốc quỹ đạo
# Vị trí ban đầu [x, y, z] (km).
# Vệ tinh ở độ cao z=550km, bắt đầu bay từ x=-200km.
SAT_INITIAL_POS_KM = [-200.0, 0.0, SAT_ALTITUDE_KM]

# ============================================
# User Equipment (Ground Station) Parameters
# ============================================
NUM_USERS = 10
# Người dùng được phân bố ngẫu nhiên trong một khu vực hình vuông.
# Tọa độ (x, y) từ -50km đến 50km. Người dùng ở mặt đất (z=0).
USER_DEPLOY_AREA_KM = {"x_min": -50.0, "x_max": 50.0, "y_min": -50.0, "y_max": 50.0}

# ============================================
# Physical Layer Parameters
# ============================================
# Đây là phần chúng ta sẽ đào sâu sau này. Bây giờ chỉ là các giá trị giữ chỗ.
CENTER_FREQUENCY_GHZ = 140.0
TOTAL_BANDWIDTH_MHZ = 1000.0
TOTAL_POWER_W = 20.0
NUM_BEAMS = 16
BEAM_WIDTH_DEG = 0.5  # Chùm tia rất hẹp, đặc trưng của Sub-THz

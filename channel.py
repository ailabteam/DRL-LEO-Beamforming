# channel.py
import numpy as np
import config as cfg

# Hằng số vật lý
SPEED_OF_LIGHT = 299792458.0  # (m/s)

# Tham số mô hình suy hao khí quyển (đơn giản hóa)
# Đây là giá trị suy hao khi tín hiệu đi thẳng từ trên đỉnh đầu xuống (zenith).
# Giá trị thực tế phụ thuộc nhiều vào tần số, độ ẩm, etc.
# 2.0 dB là một con số hợp lý cho tần số ~140 GHz trong điều kiện trời quang.
ATMOSPHERIC_LOSS_ZENITH_DB = 2.0 

def calculate_elevation_angle(sat_pos_km, user_pos_km):
    """
    Tính góc ngẩng từ người dùng đến vệ tinh.
    
    Args:
        sat_pos_km (np.ndarray): Vị trí vệ tinh [x, y, z].
        user_pos_km (np.ndarray): Vị trí người dùng [x, y, z].
                                 Có thể là một mảng (N_users, 3).
    
    Returns:
        float or np.ndarray: Góc ngẩng tính bằng độ (degrees).
    """
    # Vector từ người dùng đến vệ tinh
    vec_user_to_sat = sat_pos_km - user_pos_km
    
    # Vector pháp tuyến của mặt đất tại vị trí người dùng (chỉ lên trên)
    # Giả định mặt đất phẳng, nên vector này luôn là [0, 0, 1]
    # user_pos_km chỉ có dạng (N,3) nên cần reshape để broadcast
    z_axis = np.array([0, 0, 1])
    
    # Góc ngẩng là góc giữa vector user->sat và mặt phẳng ngang.
    # Nó cũng là 90 độ trừ đi góc giữa vector user->sat và vector thẳng đứng (z_axis).
    # sin(elevation) = (vec_user_to_sat . z_axis) / (||vec_user_to_sat|| * ||z_axis||)
    # Vì ||z_axis|| = 1 và vec_user_to_sat . z_axis = z_component_of_vec
    
    z_component = vec_user_to_sat[..., 2] # Lấy thành phần z
    distance = np.linalg.norm(vec_user_to_sat, axis=-1)
    
    # Tính sin của góc ngẩng, kẹp giá trị trong [-1, 1] để tránh lỗi toán học
    sin_elevation = np.clip(z_component / distance, -1.0, 1.0)
    
    # Chuyển từ radian sang độ
    elevation_rad = np.arcsin(sin_elevation)
    elevation_deg = np.rad2deg(elevation_rad)
    
    return elevation_deg


def calculate_atmospheric_loss_db(elevation_deg):
    """
    Tính suy hao do khí quyển dựa trên góc ngẩng.
    
    Args:
        elevation_deg (float or np.ndarray): Góc ngẩng (độ).
        
    Returns:
        float or np.ndarray: Suy hao khí quyển (dB).
    """
    # Góc ngẩng phải > 0
    elevation_deg = np.maximum(elevation_deg, 1e-6) # Tránh chia cho 0
    sin_elevation = np.sin(np.deg2rad(elevation_deg))
    
    atmospheric_loss = ATMOSPHERIC_LOSS_ZENITH_DB / sin_elevation
    return atmospheric_loss



def calculate_fspl_db(distance_km, frequency_ghz):
    """
    Tính toán Suy hao trong không gian tự do (Free Space Path Loss - FSPL).

    Args:
        distance_km (float or np.ndarray): Khoảng cách từ phát đến thu (km).
        frequency_ghz (float): Tần số sóng mang (GHz).

    Returns:
        float or np.ndarray: Giá trị FSPL tính bằng dB.
    """
    # Chuyển đổi đơn vị về đơn vị chuẩn (SI)
    d_meter = distance_km * 1000.0
    f_hz = frequency_ghz * 1e9

    # Tính toán FSPL theo công thức logarit
    # log10(x*y) = log10(x) + log10(y)
    # np.log10 hoạt động được với cả số vô hướng và mảng numpy
    fspl_db = 20 * np.log10(d_meter) + 20 * np.log10(f_hz) + 20 * np.log10(4 * np.pi / SPEED_OF_LIGHT)

    return fspl_db

# --- Bạn có thể chạy file này trực tiếp để test ---
if __name__ == "__main__":
    print("--- Chạy thử nghiệm cho module Channel ---")

    # Kịch bản 1: Khoảng cách 550 km (ngay trên đầu), tần số 140 GHz
    dist1 = 550.0
    freq1 = cfg.CENTER_FREQUENCY_GHZ
    fspl1 = calculate_fspl_db(dist1, freq1)
    print(f"FSPL ở khoảng cách {dist1} km, tần số {freq1} GHz là: {fspl1:.2f} dB")

    # Kịch bản 2: Khoảng cách 700 km (ở rìa vùng phủ)
    dist2 = 700.0
    freq2 = cfg.CENTER_FREQUENCY_GHZ
    fspl2 = calculate_fspl_db(dist2, freq2)
    print(f"FSPL ở khoảng cách {dist2} km, tần số {freq2} GHz là: {fspl2:.2f} dB")

    # Phân tích: Suy hao phải tăng lên khi khoảng cách tăng
    print(f"Sự khác biệt suy hao: {fspl2 - fspl1:.2f} dB")

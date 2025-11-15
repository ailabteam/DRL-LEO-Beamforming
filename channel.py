# channel.py
import numpy as np
import config as cfg

# Hằng số vật lý
SPEED_OF_LIGHT = 299792458.0  # (m/s)

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

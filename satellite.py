# satellite.py
import numpy as np

class Satellite:
    def __init__(self, initial_pos_km, velocity_km_s):
        """
        Khởi tạo một đối tượng vệ tinh.
        
        Args:
            initial_pos_km (list or np.ndarray): Vị trí ban đầu [x, y, z] theo km.
            velocity_km_s (float): Tốc độ của vệ tinh theo km/s.
        """
        # Chuyển đổi list thành numpy array để dễ tính toán vector
        self.pos_km = np.array(initial_pos_km, dtype=np.float64)
        
        # Giả định đơn giản: vệ tinh bay song song với trục x
        self.velocity_km_s = np.array([velocity_km_s, 0.0, 0.0], dtype=np.float64)
        
        print(f"Vệ tinh được tạo tại vị trí: {self.pos_km} km")

    def move(self, dt_s):
        """
        Cập nhật vị trí của vệ tinh sau một khoảng thời gian dt_s.
        Công thức vật lý cơ bản: quãng đường = vận tốc * thời gian.
        
        Args:
            dt_s (float): Khoảng thời gian di chuyển (tính bằng giây).
        """
        delta_pos = self.velocity_km_s * dt_s
        self.pos_km += delta_pos

    def get_position_km(self):
        """Trả về vị trí hiện tại của vệ tinh."""
        return self.pos_km

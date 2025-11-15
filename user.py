# user.py
import numpy as np

class Users:
    def __init__(self, num_users, deploy_area_km):
        """
        Khởi tạo và quản lý một nhóm người dùng mặt đất.
        
        Args:
            num_users (int): Số lượng người dùng.
            deploy_area_km (dict): Khu vực phân bố người dùng.
        """
        self.num_users = num_users
        self.positions_km = self._deploy(deploy_area_km)
        
        print(f"Đã tạo {self.num_users} người dùng.")
        # print("Vị trí của họ (km):\n", self.positions_km)

    def _deploy(self, area):
        """
        Phân bố ngẫu nhiên người dùng trong một khu vực cho trước.
        """
        x_coords = np.random.uniform(low=area["x_min"], high=area["x_max"], size=self.num_users)
        y_coords = np.random.uniform(low=area["y_min"], high=area["y_max"], size=self.num_users)
        z_coords = np.zeros(self.num_users) # Người dùng ở mặt đất, z=0
        
        # np.vstack và .T dùng để xếp các mảng 1D thành một mảng 2D (N_users, 3)
        return np.vstack((x_coords, y_coords, z_coords)).T

    def get_positions_km(self):
        """Trả về vị trí của tất cả người dùng."""
        return self.positions_km

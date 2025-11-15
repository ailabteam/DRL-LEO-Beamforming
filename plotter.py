# plotter.py
import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_results_v2(results):
    """
    Vẽ đồ thị kết quả mô phỏng (phiên bản 2).
    Hàm này vẽ 3 biểu đồ: Góc ngẩng, Khoảng cách, và Tổng suy hao.
    
    Args:
        results (list of dict): List chứa dữ liệu từ mỗi bước mô phỏng.
    """
    # Chuyển đổi list of dicts thành một dict of arrays
    try:
        results_dict = {key: np.array([res[key] for res in results]) for key in results[0]}
    except IndexError:
        print("Lỗi: List 'results' rỗng, không có gì để vẽ.")
        return

    # Tạo một figure với 3 subplot xếp chồng lên nhau
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # --- Biểu đồ 1: Góc ngẩng theo thời gian ---
    ax1.plot(results_dict['step'], results_dict['elevation_user0'], label='Elevation to User 0', color='green')
    ax1.set_ylabel('Elevation Angle (degrees)')
    ax1.set_title('Satellite Pass Simulation - Enhanced PHY Model')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    # Thêm một đường ngang để chỉ góc ngẩng tối thiểu thường được yêu cầu (ví dụ: 10 độ)
    ax1.axhline(y=10, color='r', linestyle='--', label='Min Elevation (10 deg)')
    ax1.legend() # Gọi lại legend để hiển thị label mới

    # --- Biểu đồ 2: Khoảng cách theo thời gian ---
    ax2.plot(results_dict['step'], results_dict['dist_user0'], label='Distance to User 0', color='royalblue')
    ax2.set_ylabel('Distance (km)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    # --- Biểu đồ 3: Tổng Suy hao theo thời gian ---
    ax3.plot(results_dict['step'], results_dict['total_loss_user0'], label='Total Loss to User 0', color='orangered')
    ax3.set_xlabel('Simulation Step')
    ax3.set_ylabel('Total Path Loss (dB)')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()
    
    plt.tight_layout()
    
    output_filename = 'simulation_results_v2.png'
    plt.savefig(output_filename, dpi=600)
    
    print(f"\nĐã vẽ và lưu đồ thị kết quả vào file: {output_filename}")

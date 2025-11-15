# plotter.py
import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_results(results):
    """
    Vẽ đồ thị kết quả mô phỏng.
    
    Args:
        results (list of dict): List chứa dữ liệu từ mỗi bước mô phỏng.
    """
    # Chuyển đổi list of dicts thành một dict of lists/arrays để dễ dàng truy cập
    # Ví dụ: results_dict['step'] sẽ là một mảng chứa tất cả các bước
    results_dict = {key: np.array([res[key] for res in results]) for key in results[0]}
    
    # Tạo một figure với 2 subplot (biểu đồ con) xếp chồng lên nhau
    # sharex=True có nghĩa là cả hai biểu đồ sẽ dùng chung trục x (thời gian)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- Biểu đồ 1: Khoảng cách theo thời gian ---
    ax1.plot(results_dict['step'], results_dict['dist_user0'], label='Distance to User 0', color='royalblue')
    ax1.set_ylabel('Distance (km)')
    ax1.set_title('Satellite Pass Simulation Results for User 0')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # --- Biểu đồ 2: FSPL theo thời gian ---
    ax2.plot(results_dict['step'], results_dict['fspl_user0'], label='FSPL to User 0', color='orangered')
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Free Space Path Loss (dB)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    # Tinh chỉnh layout để các thành phần không bị đè lên nhau
    plt.tight_layout()
    
    # Lưu figure ra file với chất lượng cao
    # dpi=600 là yêu cầu của bạn để có hình ảnh sắc nét cho paper
    output_filename = 'simulation_results.png'
    plt.savefig(output_filename, dpi=600)
    
    print(f"\nĐã vẽ và lưu đồ thị kết quả vào file: {output_filename}")
    
    # Hiển thị đồ thị (nếu bạn có giao diện đồ họa)
    # plt.show()

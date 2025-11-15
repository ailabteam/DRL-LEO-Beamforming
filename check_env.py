import sys
import torch
import stable_baselines3 as sb3

print(f"Python version: {sys.version}")
print("-" * 30)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version used by PyTorch: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("-" * 30)
print(f"Stable-Baselines3 version: {sb3.__version__}")

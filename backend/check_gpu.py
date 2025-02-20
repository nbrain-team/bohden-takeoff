import torch
import os

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("PyTorch version:", torch.__version__)
else:
    print("CUDA is not available. Please check your CUDA installation and GPU drivers.")
    print("Environment variables:")
    print(f"PATH: {os.environ.get('PATH')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

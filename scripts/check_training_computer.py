import sys
import platform

print("RespirAI training computer check")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print("PyTorch check failed:")
    print(exc)
import torch


def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)} with {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB VRAM")
    else:
        print("No GPU available, using CPU.")


print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
check_gpu()
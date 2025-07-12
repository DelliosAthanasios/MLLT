import torch

def get_device(force_cpu=False, force_cuda=False, force_hip=False, force_mps=False):
    """Return the best available device, or force a specific one if requested."""
    if force_cpu:
        return torch.device('cpu')
    if force_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    if force_hip and hasattr(torch, 'hip') and torch.version.hip is not None:
        return torch.device('hip')
    if force_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    # Auto-detect order: CUDA, HIP, MPS, CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch, 'hip') and torch.version.hip is not None:
        return torch.device('hip')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def move_model_to_device(model, device):
    """Move a model to the specified device."""
    return model.to(device)

def move_tensor_to_device(tensor, device):
    """Move a tensor to the specified device."""
    return tensor.to(device)

def print_gpu_info():
    print("Device availability:")
    print(f"  CUDA (NVIDIA): {'Yes' if torch.cuda.is_available() else 'No'}")
    print(f"  HIP (AMD/ROCm): {'Yes' if hasattr(torch, 'hip') and torch.version.hip is not None else 'No'}")
    print(f"  MPS (Apple): {'Yes' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'No'}")
    print(f"  CPU: Yes")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    if hasattr(torch, 'hip') and torch.version.hip is not None:
        print("HIP (AMD/ROCm) device detected.")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) device detected.")

def get_available_device_types():
    types = ['CPU']
    if torch.cuda.is_available():
        types.append('CUDA (NVIDIA)')
    if hasattr(torch, 'hip') and torch.version.hip is not None:
        types.append('HIP (AMD/ROCm)')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        types.append('MPS (Apple)')
    return types 
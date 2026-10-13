import os
import shutil
import sys
from typing import Any, Dict


def is_kaggle_environment() -> bool:
    """Detects whether the code is executing inside a Kaggle Notebook container."""
    return os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def get_hardware_info() -> Dict[str, Any]:
    """Detects runtime hardware, GPU specifications, VRAM, RAM, and disk space."""
    info: Dict[str, Any] = {
        "is_kaggle": is_kaggle_environment(),
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "bf16_supported": False,
        "ram_gb": None,
        "disk_free_gb": None,
        "recommended_dtype": "float32"
    }

    # System memory
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_gb"] = round(vm.total / (1024 ** 3), 2)
        info["ram_available_gb"] = round(vm.available / (1024 ** 3), 2)
    except ImportError:
        pass

    # Disk space
    try:
        disk_path = "/kaggle/working" if is_kaggle_environment() else "."
        total, used, free = shutil.disk_usage(disk_path)
        info["disk_total_gb"] = round(total / (1024 ** 3), 2)
        info["disk_free_gb"] = round(free / (1024 ** 3), 2)
    except Exception:
        pass

    # PyTorch and GPU info
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                dev_info = {
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / (1024 ** 3), 2),
                    "major": props.major,
                    "minor": props.minor,
                }
                devices.append(dev_info)
            info["devices"] = devices

            # Check BF16 capability (Ampere sm_80+ or torch.cuda.is_bf16_supported())
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                info["bf16_supported"] = True
                info["recommended_dtype"] = "bfloat16"
            else:
                info["bf16_supported"] = False
                info["recommended_dtype"] = "float16"
        else:
            info["recommended_dtype"] = "float32"

    except ImportError:
        info["torch_version"] = "not_installed"

    return info

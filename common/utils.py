import psutil
import GPUtil
import torch
from transformers.utils import logging

# Set logging verbosity
logging.set_verbosity_info()

def get_system_info():
    """Get current system information.
    
    Returns:
        dict: System information including CPU, memory, and GPU stats
    """
    system_info = {
        "cpu": {
            "count": psutil.cpu_count(),
            "physical_count": psutil.cpu_count(logical=False),
            "frequency": {
                "current": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "min": psutil.cpu_freq().min if psutil.cpu_freq() else None,
                "max": psutil.cpu_freq().max if psutil.cpu_freq() else None
            },
            "usage_percent": psutil.cpu_percent(interval=1)
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "used": psutil.virtual_memory().used,
            "percent": psutil.virtual_memory().percent
        },
        "gpu": []
    }
    
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                system_info["gpu"].append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "temperature": gpu.temperature,
                    "load": gpu.load * 100
                })
        except Exception as e:
            logging.warning(f"Failed to get GPU information: {str(e)}")
    
    return system_info 
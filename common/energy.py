import os
import time
import logging
from datetime import datetime
import psutil
import GPUtil
from codecarbon import EmissionsTracker

logger = logging.getLogger(__name__)

class EnergyMonitor:
    """Monitor energy consumption during model training and inference."""
    
    def __init__(self, save_traces=True, trace_interval=1.0):
        """Initialize the energy monitor.
        
        Args:
            save_traces (bool): Whether to save detailed traces
            trace_interval (float): Interval between trace measurements in seconds
        """
        self.save_traces = save_traces
        self.trace_interval = trace_interval
        self.start_time = None
        self.end_time = None
        self.traces = []
        self.tracker = None
        
    def start(self):
        """Start monitoring energy consumption."""
        self.start_time = time.time()
        self.traces = []
        
        # Create energy_logs directory if it doesn't exist
        os.makedirs("energy_logs", exist_ok=True)
        
        # Initialize CodeCarbon tracker
        self.tracker = EmissionsTracker(
            project_name="greenai-pipeline",
            output_dir="energy_logs",
            log_level="error"
        )
        self.tracker.start()
        
        if self.save_traces:
            self._start_tracing()
    
    def stop(self):
        """Stop monitoring and return energy statistics.
        
        Returns:
            dict: Energy consumption statistics
        """
        if self.tracker:
            self.tracker.stop()
        
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Get final system stats
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        gpu_stats = []
        
        if GPUtil.getGPUs():
            for gpu in GPUtil.getGPUs():
                gpu_stats.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature,
                    "load": gpu.load * 100
                })
        
        stats = {
            "duration_seconds": duration,
            "cpu_percent": cpu_percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "gpu_stats": gpu_stats,
            "traces": self.traces if self.save_traces else None
        }
        
        return stats
    
    def _start_tracing(self):
        """Start periodic tracing of system stats."""
        def trace_worker():
            while self.start_time and not self.end_time:
                try:
                    # Get CPU stats
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    
                    # Get GPU stats
                    gpu_stats = []
                    if GPUtil.getGPUs():
                        for gpu in GPUtil.getGPUs():
                            gpu_stats.append({
                                "id": gpu.id,
                                "name": gpu.name,
                                "memory_used": gpu.memoryUsed,
                                "memory_total": gpu.memoryTotal,
                                "temperature": gpu.temperature,
                                "load": gpu.load * 100
                            })
                    
                    # Record trace
                    self.traces.append({
                        "timestamp": datetime.now().isoformat(),
                        "cpu_percent": cpu_percent,
                        "memory_used_gb": memory.used / (1024**3),
                        "memory_total_gb": memory.total / (1024**3),
                        "gpu_stats": gpu_stats
                    })
                    
                    time.sleep(self.trace_interval)
                except Exception as e:
                    logger.error(f"Error in trace worker: {str(e)}")
                    break
        
        import threading
        self.trace_thread = threading.Thread(target=trace_worker, daemon=True)
        self.trace_thread.start()

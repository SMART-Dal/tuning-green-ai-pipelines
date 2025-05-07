import time
from contextlib import contextmanager
import pynvml
from codecarbon import EmissionsTracker
import logging

logger = logging.getLogger(__name__)

class EnergyMonitor:
    def __init__(self):
        """Initialize energy monitoring for both GPU and CPU/system."""
        self.stage_metrics = {
            "data": 0.0,
            "architecture": 0.0,
            "training": 0.0,
            "system": 0.0,
            "inference": 0.0,
            "total": 0.0
        }
        
        # Initialize NVIDIA monitoring if GPU is available
        self.has_gpu = False
        try:
            pynvml.nvmlInit()
            self.has_gpu = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            logger.warning(f"GPU monitoring not available: {e}")
        
        # Initialize CodeCarbon for CPU/system monitoring
        self.carbon_tracker = EmissionsTracker(
            tracking_mode="process",
            log_level="error",
            measure_power_secs=1,
            save_to_file=False
        )

    @contextmanager
    def measure(self, stage_name):
        """Context manager to measure energy consumption for a pipeline stage.
        
        Args:
            stage_name (str): Name of the pipeline stage being measured
        """
        if stage_name not in self.stage_metrics:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        # Start measurements
        start_time = time.time()
        gpu_start_energy = self._get_gpu_energy() if self.has_gpu else 0
        self.carbon_tracker.start()
        
        try:
            yield
        finally:
            # End measurements
            self.carbon_tracker.stop()
            gpu_end_energy = self._get_gpu_energy() if self.has_gpu else 0
            end_time = time.time()
            
            # Calculate energy consumption
            gpu_energy = (gpu_end_energy - gpu_start_energy) / 3600  # Convert to kWh
            cpu_energy = self.carbon_tracker.final_emissions_data.get("energy_consumed", 0) / 3600  # Convert to kWh
            
            # Store total energy for this stage
            self.stage_metrics[stage_name] = gpu_energy + cpu_energy
            self.stage_metrics["total"] += self.stage_metrics[stage_name]
            
            logger.info(f"Stage '{stage_name}' completed in {end_time - start_time:.2f}s")
            logger.info(f"Energy consumption: {self.stage_metrics[stage_name]:.4f} kWh")

    def _get_gpu_energy(self):
        """Get total energy consumption from GPU in joules."""
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)
        except Exception as e:
            logger.warning(f"Failed to get GPU energy: {e}")
            return 0

    def get_metrics(self):
        """Get the current energy metrics for all stages.
        
        Returns:
            dict: Dictionary containing energy consumption per stage in kWh
        """
        return self.stage_metrics.copy()

    def reset(self):
        """Reset all energy metrics to zero."""
        for key in self.stage_metrics:
            self.stage_metrics[key] = 0.0

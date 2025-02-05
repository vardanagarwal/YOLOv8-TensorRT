import torch
from contextlib import contextmanager
from collections import defaultdict
import time

class NVTXProfiler:
    def __init__(self):
        self.timing_stats = defaultdict(list)
        
    @contextmanager
    def range(self, name):
        """Context manager for timing CUDA operations"""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        try:
            yield
        finally:
            end.record()
            # Synchronize and measure time
            end.synchronize()
            elapsed_time = start.elapsed_time(end)
            self.timing_stats[name].append(elapsed_time)
    
    def print_stats(self):
        """Print timing statistics"""
        print("\nNVTX Profiling Results:")
        print("-" * 80)
        print(f"{'Operation Name':<40} {'Mean (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Calls':<8}")
        print("-" * 80)
        
        total_inference_time = 0
        for name, times in sorted(self.timing_stats.items()):
            mean_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            calls = len(times)
            
            print(f"{name:<40} {mean_time:>10.3f} {min_time:>10.3f} {max_time:>10.3f} {calls:>8}")
            
            if name == "Inference":
                total_inference_time = sum(times)
        
        print("-" * 80)
        if total_inference_time > 0:
            print(f"\nTotal Inference Time: {total_inference_time:.2f} ms")
            print(f"Average Inference Time per call: {total_inference_time/len(self.timing_stats['Inference']):.2f} ms")

# Global profiler instance
profiler = NVTXProfiler()
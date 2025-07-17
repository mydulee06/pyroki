import psutil
import jax
import jax.numpy as jnp
import time
import os
from typing import Dict, Any, Optional
from memory_profiler import profile
import gc

def get_system_memory_info() -> Dict[str, float]:
    """Get current system memory usage information."""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent_used': memory.percent
    }

def get_gpu_memory_info() -> Optional[Dict[str, Any]]:
    """Get GPU memory information if available."""
    try:
        # Try to get GPU memory info using JAX
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        
        if gpu_devices:
            # For now, return basic info since detailed GPU memory info
            # requires additional libraries like nvidia-ml-py
            return {
                'gpu_count': len(gpu_devices),
                'gpu_platforms': [d.platform for d in gpu_devices],
                'note': 'Detailed GPU memory info requires nvidia-ml-py'
            }
        else:
            return None
    except Exception as e:
        print(f"Error getting GPU memory info: {e}")
        return None

def print_memory_status(label: str = "Current"):
    """Print current memory status."""
    print(f"\n=== {label} Memory Status ===")
    
    # System memory
    sys_mem = get_system_memory_info()
    print(f"System Memory:")
    print(f"  Total: {sys_mem['total_gb']:.2f} GB")
    print(f"  Used: {sys_mem['used_gb']:.2f} GB ({sys_mem['percent_used']:.1f}%)")
    print(f"  Available: {sys_mem['available_gb']:.2f} GB")
    
    # GPU memory
    gpu_mem = get_gpu_memory_info()
    if gpu_mem:
        print(f"GPU Memory:")
        print(f"  Devices: {gpu_mem['gpu_count']}")
        print(f"  Platforms: {gpu_mem['gpu_platforms']}")
        if 'note' in gpu_mem:
            print(f"  Note: {gpu_mem['note']}")
    else:
        print("GPU Memory: Not available")

def monitor_memory_usage(func):
    """Decorator to monitor memory usage during function execution."""
    def wrapper(*args, **kwargs):
        print_memory_status("Before function execution")
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print_memory_status("After function execution")
        print(f"Function execution time: {end_time - start_time:.2f} seconds")
        
        return result
    return wrapper

@profile
def profile_memory_usage(func, *args, **kwargs):
    """Profile memory usage of a function using memory_profiler."""
    return func(*args, **kwargs)

def clear_jax_cache():
    """Clear JAX compilation cache to free memory."""
    try:
        jax.clear_caches()
        print("JAX cache cleared")
    except Exception as e:
        print(f"Error clearing JAX cache: {e}")

def force_garbage_collection():
    """Force garbage collection to free memory."""
    collected = gc.collect()
    print(f"Garbage collection: {collected} objects collected")

def optimize_memory_settings():
    """Apply memory optimization settings."""
    # Set JAX memory settings
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    # Enable X64 for better precision
    jax.config.update('jax_enable_x64', True)
    
    print("Memory optimization settings applied:")
    print("  - XLA_PYTHON_CLIENT_PREALLOCATE = false")
    print("  - XLA_PYTHON_CLIENT_ALLOCATOR = platform")
    print("  - jax_enable_x64 = True")

def create_memory_report(output_file: str = "memory_report.txt"):
    """Create a detailed memory usage report."""
    with open(output_file, 'w') as f:
        f.write("=== Memory Usage Report ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System memory
        sys_mem = get_system_memory_info()
        f.write("System Memory:\n")
        f.write(f"  Total: {sys_mem['total_gb']:.2f} GB\n")
        f.write(f"  Used: {sys_mem['used_gb']:.2f} GB ({sys_mem['percent_used']:.1f}%)\n")
        f.write(f"  Available: {sys_mem['available_gb']:.2f} GB\n\n")
        
        # GPU memory
        gpu_mem = get_gpu_memory_info()
        if gpu_mem:
            f.write("GPU Memory:\n")
            f.write(f"  Devices: {gpu_mem['gpu_count']}\n")
            f.write(f"  Platforms: {gpu_mem['gpu_platforms']}\n\n")
        
        # JAX device info
        try:
            devices = jax.devices()
            f.write("JAX Devices:\n")
            for i, device in enumerate(devices):
                f.write(f"  Device {i}: {device}\n")
        except Exception as e:
            f.write(f"Error getting JAX devices: {e}\n")
    
    print(f"Memory report saved to: {output_file}")

def check_memory_threshold(threshold_gb: float = 8.0) -> bool:
    """Check if available memory is above threshold."""
    sys_mem = get_system_memory_info()
    available_gb = sys_mem['available_gb']
    
    if available_gb < threshold_gb:
        print(f"⚠️  Warning: Low memory available ({available_gb:.2f} GB < {threshold_gb} GB)")
        return False
    else:
        print(f"✅ Sufficient memory available ({available_gb:.2f} GB >= {threshold_gb} GB)")
        return True

# Example usage functions
def example_memory_intensive_function():
    """Example function that uses a lot of memory."""
    print("Creating large arrays...")
    
    # Create large arrays
    large_array1 = jnp.random.random((1000, 1000))
    large_array2 = jnp.random.random((1000, 1000))
    
    # Perform computation
    result = jnp.dot(large_array1, large_array2)
    
    print("Computation completed")
    return result

if __name__ == "__main__":
    # Example usage
    print("Memory Profiling Utilities")
    print("=" * 30)
    
    # Print initial memory status
    print_memory_status("Initial")
    
    # Apply optimization settings
    optimize_memory_settings()
    
    # Check memory threshold
    check_memory_threshold(4.0)
    
    # Monitor memory usage during function execution
    print("\nMonitoring memory usage during example function...")
    monitored_result = monitor_memory_usage(example_memory_intensive_function)()
    
    # Create memory report
    create_memory_report()
    
    # Clear caches
    clear_jax_cache()
    force_garbage_collection()
    
    print_memory_status("Final") 
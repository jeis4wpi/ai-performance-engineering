#!/usr/bin/env python3

import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))


"""
Comprehensive Profiling and Roofline Analysis Toolkit (Chapter 17)

Implements complete profiling workflows including roofline analysis, kernel profiling,
and performance bottleneck detection as described in Chapter 17.

Key features:
- Automated roofline model generation
- Kernel-level performance analysis
- Memory bandwidth and compute utilization tracking
- Bottleneck identification
- Integration with Nsight tools

Usage:
    from comprehensive_profiling_toolkit import ProfilerToolkit
    
    profiler = ProfilerToolkit()
    with profiler.profile("my_operation"):
        # Code to profile
        result = model(input_data)
    
    profiler.generate_roofline_plot()
    profiler.print_summary()
"""

import torch
import time
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import json
from pathlib import Path
import numpy as np


class BottleneckType(Enum):
    """Types of performance bottlenecks"""
    MEMORY_BOUND = "memory_bound"
    COMPUTE_BOUND = "compute_bound"
    COMMUNICATION_BOUND = "communication_bound"
    CPU_BOUND = "cpu_bound"
    UNKNOWN = "unknown"


@dataclass
class KernelProfile:
    """Profile data for a single kernel"""
    name: str
    duration_ms: float
    gflops: float  # Achieved GFLOPS
    memory_bandwidth_gbps: float  # Achieved memory bandwidth
    arithmetic_intensity: float  # FLOPS per byte
    occupancy_percent: float
    sm_efficiency_percent: float
    memory_efficiency_percent: float
    registers_per_thread: int
    shared_memory_bytes: int
    bottleneck: BottleneckType = BottleneckType.UNKNOWN
    
    @property
    def is_memory_bound(self) -> bool:
        """Check if kernel is memory-bound"""
        return self.memory_efficiency_percent > 70 and self.sm_efficiency_percent < 60
    
    @property
    def is_compute_bound(self) -> bool:
        """Check if kernel is compute-bound"""
        return self.sm_efficiency_percent > 70 and self.memory_efficiency_percent < 60


@dataclass
class RooflinePoint:
    """A point on the roofline plot"""
    name: str
    arithmetic_intensity: float  # FLOPS/byte
    performance_gflops: float  # Achieved GFLOPS
    is_memory_bound: bool
    is_compute_bound: bool


@dataclass
class DeviceSpecs:
    """GPU device specifications for roofline model"""
    peak_flops: float  # Peak FLOPS
    peak_memory_bandwidth: float  # Peak memory bandwidth (GB/s)
    peak_tensor_core_flops: float  # Peak Tensor Core FLOPS
    hbm_bandwidth: float  # HBM bandwidth
    l2_bandwidth: float  # L2 cache bandwidth
    device_name: str
    
    @property
    def ridge_point(self) -> float:
        """Calculate ridge point (FLOPS/byte)"""
        return self.peak_flops / self.peak_memory_bandwidth


class ProfilerToolkit:
    """
    Comprehensive profiling toolkit for GPU kernels and operations.
    
    Implements profiling workflows from Chapter 17 including:
    - Kernel profiling with PyTorch profiler
    - Roofline analysis
    - Bottleneck detection
    - Performance recommendations
    """
    
    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        output_dir: str = "./profiling_results"
    ):
        """
        Initialize profiler toolkit.
        
        Args:
            device: Device to profile
            output_dir: Directory to save profiling results
        """
        self.device = self._resolve_device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.kernel_profiles: List[KernelProfile] = []
        self.roofline_points: List[RooflinePoint] = []
        
        # Get device specifications
        self.device_specs = self._get_device_specs()
        
        print(f"Initialized ProfilerToolkit for {self.device_specs.device_name}")
        print(f"Peak FLOPS: {self.device_specs.peak_flops / 1e12:.2f} TFLOPS")
        print(f"Peak Bandwidth: {self.device_specs.peak_memory_bandwidth:.2f} GB/s")
    
    def _resolve_device(self, requested: torch.device) -> torch.device:
        if requested.type != "cuda":
            return torch.device("cpu")
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available; ProfilerToolkit running on CPU.")
            return torch.device("cpu")
        try:
            torch.cuda.set_device(requested)
            torch.ones(1, device=requested)
            torch.cuda.synchronize()
            return requested
        except Exception as exc:
            print(f"WARNING: Unable to use CUDA device ({exc}); ProfilerToolkit running on CPU.")
            return torch.device("cpu")

    def _get_device_specs(self) -> DeviceSpecs:
        """
        Get device specifications for roofline model.
        
        Returns:
            DeviceSpecs for the current device
        """
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return DeviceSpecs(
                peak_flops=1e12,
                peak_memory_bandwidth=100.0,
                peak_tensor_core_flops=2e12,
                hbm_bandwidth=100.0,
                l2_bandwidth=1000.0,
                device_name="CPU"
            )
        
        try:
            props = torch.cuda.get_device_properties(self.device)
        except Exception:
            return DeviceSpecs(
                peak_flops=1e12,
                peak_memory_bandwidth=100.0,
                peak_tensor_core_flops=2e12,
                hbm_bandwidth=100.0,
                l2_bandwidth=1000.0,
                device_name="CPU"
            )
        device_name = props.name
        
        # Determine specs based on architecture
        if any(token in device_name for token in ("B200", "B300", "Blackwell")):
            # NVIDIA B200 specifications
            peak_flops = 2000e12  # 2000 TFLOPS FP16
            peak_tensor_core_flops = 2000e12
            hbm_bandwidth = 8000.0  # 8 TB/s
            l2_bandwidth = 16000.0  # Estimate
        elif "H100" in device_name or "H200" in device_name:
            # NVIDIA H100/H200 specifications  
            peak_flops = 1000e12  # 1000 TFLOPS FP16
            peak_tensor_core_flops = 1000e12
            hbm_bandwidth = 3350.0  # 3.35 TB/s
            l2_bandwidth = 10000.0
        elif "A100" in device_name:
            # NVIDIA A100 specifications
            peak_flops = 312e12  # 312 TFLOPS FP16
            peak_tensor_core_flops = 312e12
            hbm_bandwidth = 1555.0  # 1.555 TB/s for 40GB
            l2_bandwidth = 7000.0
        else:
            # Generic estimates
            sm_count = props.multi_processor_count
            clock_rate_khz = getattr(props, "clock_rate", None)
            if clock_rate_khz is None:
                clock_rate_khz = getattr(props, "clockRate", None)
            if clock_rate_khz is None:
                clock_rate_khz = getattr(props, "clock_rate_khz", None)
            if clock_rate_khz is None:
                clock_rate_khz = 0.0
            clock_rate_ghz = float(clock_rate_khz) / 1e6
            # Rough estimate: SMs * clock * ops_per_clock
            peak_flops = sm_count * clock_rate_ghz * 128 * 1e9
            peak_tensor_core_flops = peak_flops * 2
            hbm_bandwidth = 900.0  # Generic
            l2_bandwidth = 3000.0
        
        return DeviceSpecs(
            peak_flops=peak_flops,
            peak_memory_bandwidth=hbm_bandwidth,
            peak_tensor_core_flops=peak_tensor_core_flops,
            hbm_bandwidth=hbm_bandwidth,
            l2_bandwidth=l2_bandwidth,
            device_name=device_name
        )
    
    def profile(
        self,
        name: str,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True
    ):
        """
        Context manager for profiling a code block.
        
        Args:
            name: Name for this profiling session
            record_shapes: Record tensor shapes
            profile_memory: Profile memory usage
            with_stack: Record stack traces
            
        Returns:
            torch.profiler.profile context manager
        """
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.output_dir / name)
            )
        )
    
    def analyze_kernel(
        self,
        kernel_name: str,
        flops: float,
        memory_bytes: float,
        duration_ms: float,
        occupancy: float = 0.0,
        sm_efficiency: float = 0.0
    ) -> KernelProfile:
        """
        Analyze a kernel and determine its bottleneck.
        
        Args:
            kernel_name: Name of the kernel
            flops: Number of floating point operations
            memory_bytes: Amount of memory accessed
            duration_ms: Kernel duration in milliseconds
            occupancy: Occupancy percentage
            sm_efficiency: SM efficiency percentage
            
        Returns:
            KernelProfile with analysis
        """
        # Calculate achieved metrics
        duration_s = duration_ms / 1000.0
        achieved_gflops = (flops / duration_s) / 1e9
        achieved_bandwidth_gbps = (memory_bytes / duration_s) / 1e9
        
        # Calculate arithmetic intensity (FLOPS per byte)
        arithmetic_intensity = flops / memory_bytes if memory_bytes > 0 else 0
        
        # Determine memory efficiency
        memory_efficiency = (achieved_bandwidth_gbps / self.device_specs.peak_memory_bandwidth) * 100
        
        # Determine bottleneck
        if memory_efficiency > 70 and sm_efficiency < 60:
            bottleneck = BottleneckType.MEMORY_BOUND
        elif sm_efficiency > 70 and memory_efficiency < 60:
            bottleneck = BottleneckType.COMPUTE_BOUND
        else:
            bottleneck = BottleneckType.UNKNOWN
        
        profile = KernelProfile(
            name=kernel_name,
            duration_ms=duration_ms,
            gflops=achieved_gflops,
            memory_bandwidth_gbps=achieved_bandwidth_gbps,
            arithmetic_intensity=arithmetic_intensity,
            occupancy_percent=occupancy,
            sm_efficiency_percent=sm_efficiency,
            memory_efficiency_percent=memory_efficiency,
            registers_per_thread=0,
            shared_memory_bytes=0,
            bottleneck=bottleneck
        )
        
        self.kernel_profiles.append(profile)
        
        # Add to roofline points
        roofline_point = RooflinePoint(
            name=kernel_name,
            arithmetic_intensity=arithmetic_intensity,
            performance_gflops=achieved_gflops,
            is_memory_bound=profile.is_memory_bound,
            is_compute_bound=profile.is_compute_bound
        )
        self.roofline_points.append(roofline_point)
        
        return profile
    
    def generate_roofline_plot(self, output_file: str = "roofline.png"):
        """
        Generate roofline model plot.
        
        Args:
            output_file: Output filename for the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plot generation")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Generate roofline curves
        intensity_range = np.logspace(-2, 3, 1000)
        
        # Memory bandwidth roofline
        memory_roofline = intensity_range * self.device_specs.peak_memory_bandwidth
        
        # Compute roofline (flat line at peak FLOPS)
        compute_roofline = np.full_like(intensity_range, self.device_specs.peak_flops / 1e9)
        
        # Combined roofline (minimum of memory and compute)
        roofline = np.minimum(memory_roofline, compute_roofline)
        
        # Plot roofline
        ax.loglog(intensity_range, roofline, 'k-', linewidth=2, label='Roofline')
        ax.loglog(intensity_range, memory_roofline, 'k--', alpha=0.5, label='Memory Bound')
        ax.axhline(y=self.device_specs.peak_flops / 1e9, color='k', linestyle='--', 
                   alpha=0.5, label='Compute Bound')
        
        # Plot kernel points
        if self.roofline_points:
            memory_bound = [p for p in self.roofline_points if p.is_memory_bound]
            compute_bound = [p for p in self.roofline_points if p.is_compute_bound]
            unknown = [p for p in self.roofline_points 
                      if not p.is_memory_bound and not p.is_compute_bound]
            
            if memory_bound:
                ax.scatter(
                    [p.arithmetic_intensity for p in memory_bound],
                    [p.performance_gflops for p in memory_bound],
                    c='red', s=100, alpha=0.7, label='Memory Bound Kernels'
                )
            
            if compute_bound:
                ax.scatter(
                    [p.arithmetic_intensity for p in compute_bound],
                    [p.performance_gflops for p in compute_bound],
                    c='blue', s=100, alpha=0.7, label='Compute Bound Kernels'
                )
            
            if unknown:
                ax.scatter(
                    [p.arithmetic_intensity for p in unknown],
                    [p.performance_gflops for p in unknown],
                    c='gray', s=100, alpha=0.7, label='Other'
                )
            
            # Label points
            for p in self.roofline_points:
                ax.annotate(
                    p.name,
                    (p.arithmetic_intensity, p.performance_gflops),
                    fontsize=8,
                    alpha=0.7
                )
        
        # Mark ridge point
        ridge_x = self.device_specs.ridge_point
        ridge_y = self.device_specs.peak_flops / 1e9
        ax.axvline(x=ridge_x, color='green', linestyle=':', alpha=0.5, label='Ridge Point')
        
        ax.set_xlabel('Arithmetic Intensity (FLOPS/Byte)', fontsize=12)
        ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
        ax.set_title(f'Roofline Model - {self.device_specs.device_name}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Roofline plot saved to {output_path}")
        plt.close()
    
    def identify_bottlenecks(self) -> Dict[BottleneckType, List[KernelProfile]]:
        """
        Identify and categorize bottlenecks.
        
        Returns:
            Dictionary mapping bottleneck types to kernel profiles
        """
        bottlenecks: Dict[BottleneckType, List[KernelProfile]] = {
            bt: [] for bt in BottleneckType
        }
        
        for profile in self.kernel_profiles:
            bottlenecks[profile.bottleneck].append(profile)
        
        return bottlenecks
    
    def generate_recommendations(self) -> List[str]:
        """
        Generate optimization recommendations based on profiling data.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        bottlenecks = self.identify_bottlenecks()
        
        # Memory-bound kernels
        if bottlenecks[BottleneckType.MEMORY_BOUND]:
            recommendations.append(
                "MEMORY-BOUND KERNELS DETECTED:\n"
                "  - Consider kernel fusion to reduce memory traffic\n"
                "  - Use shared memory for data reuse\n"
                "  - Increase arithmetic intensity with loop tiling\n"
                "  - Consider using lower precision (FP16/FP8)"
            )
        
        # Compute-bound kernels
        if bottlenecks[BottleneckType.COMPUTE_BOUND]:
            recommendations.append(
                "COMPUTE-BOUND KERNELS DETECTED:\n"
                "  - Good! These kernels are utilizing the GPU well\n"
                "  - Consider Tensor Cores if not already using them\n"
                "  - Ensure sufficient parallelism (occupancy)\n"
                "  - Profile instruction mix for opportunities"
            )
        
        # Low occupancy kernels
        low_occupancy = [p for p in self.kernel_profiles if p.occupancy_percent < 50]
        if low_occupancy:
            recommendations.append(
                f"LOW OCCUPANCY DETECTED ({len(low_occupancy)} kernels):\n"
                "  - Reduce register usage\n"
                "  - Reduce shared memory usage\n"
                "  - Adjust block size\n"
                "  - Use occupancy calculator"
            )
        
        return recommendations
    
    def print_summary(self):
        """Print comprehensive profiling summary."""
        print("\n" + "="*80)
        print("PROFILING SUMMARY")
        print("="*80)
        
        print(f"\nDevice: {self.device_specs.device_name}")
        print(f"Peak Performance: {self.device_specs.peak_flops / 1e12:.2f} TFLOPS")
        print(f"Peak Bandwidth: {self.device_specs.peak_memory_bandwidth:.2f} GB/s")
        print(f"Ridge Point: {self.device_specs.ridge_point:.2f} FLOPS/Byte")
        
        if self.kernel_profiles:
            print(f"\n{len(self.kernel_profiles)} Kernels Profiled:")
            print("-" * 80)
            
            for profile in sorted(self.kernel_profiles, key=lambda p: p.duration_ms, reverse=True)[:10]:
                print(f"\n{profile.name}:")
                print(f"  Duration:             {profile.duration_ms:.3f} ms")
                print(f"  Performance:          {profile.gflops:.2f} GFLOPS "
                      f"({profile.gflops / (self.device_specs.peak_flops/1e9) * 100:.1f}% of peak)")
                print(f"  Memory Bandwidth:     {profile.memory_bandwidth_gbps:.2f} GB/s "
                      f"({profile.memory_efficiency_percent:.1f}% of peak)")
                print(f"  Arithmetic Intensity: {profile.arithmetic_intensity:.2f} FLOPS/Byte")
                print(f"  Bottleneck:           {profile.bottleneck.value.upper()}")
                if profile.occupancy_percent > 0:
                    print(f"  Occupancy:            {profile.occupancy_percent:.1f}%")
        
        # Print bottleneck summary
        print("\n" + "="*80)
        print("BOTTLENECK ANALYSIS")
        print("="*80)
        
        bottlenecks = self.identify_bottlenecks()
        for btype, profiles in bottlenecks.items():
            if profiles:
                total_time = sum(p.duration_ms for p in profiles)
                print(f"\n{btype.value.upper()}: {len(profiles)} kernels, {total_time:.2f} ms total")
        
        # Print recommendations
        print("\n" + "="*80)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*80)
        
        recommendations = self.generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
        
        print("\n" + "="*80 + "\n")


# Example usage
if __name__ == '__main__':
    print("=" * 80)
    print("Comprehensive Profiling Toolkit Demo (Chapter 17)")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("\nWarning: CUDA not available. Demo will use mock data.")
    
    profiler = ProfilerToolkit()
    
    # Example: Profile matrix multiplication
    print("\nProfiling matrix multiplication...")
    
    M, N, K = 4096, 4096, 4096
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    
    # Warmup
    _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Profile
    start = time.time()
    C = torch.matmul(A, B)
    torch.cuda.synchronize()
    duration_ms = (time.time() - start) * 1000
    
    # Analyze
    flops = 2 * M * N * K  # Matrix multiply FLOPs
    memory_bytes = (M * K + K * N + M * N) * 4  # FP32 = 4 bytes
    
    profile = profiler.analyze_kernel(
        kernel_name="matmul_4096x4096x4096",
        flops=flops,
        memory_bytes=memory_bytes,
        duration_ms=duration_ms,
        occupancy=85.0,
        sm_efficiency=75.0
    )
    
    print(f"Completed in {duration_ms:.2f} ms")
    print(f"Achieved: {profile.gflops:.2f} GFLOPS")
    print(f"Bottleneck: {profile.bottleneck.value}")
    
    # Generate roofline plot
    print("\nGenerating roofline plot...")
    profiler.generate_roofline_plot()
    
    # Print comprehensive summary
    profiler.print_summary()
    
    print("=" * 80)
    print("Demo complete! Check ./profiling_results/ for outputs")
    print("=" * 80)

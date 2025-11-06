# Chapter 3: System Tuning

## Overview

Even the best code can underperform on a misconfigured system. This chapter covers system-level tuning for NVIDIA GPU systems, including NUMA binding, CPU affinity, kernel parameters, and containerization best practices. These optimizations ensure your hardware operates at peak efficiency.

## Learning Objectives

After completing this chapter, you can:

- [OK] Configure NUMA (Non-Uniform Memory Access) binding for optimal CPU-GPU affinity
- [OK] Apply system-level tuning for GPU workloads
- [OK] Configure Docker and Kubernetes for GPU-accelerated containers
- [OK] Measure and validate system configuration impact on performance
- [OK] Troubleshoot common system-level bottlenecks

## Prerequisites

**Previous chapters**: 
- [Chapter 1: Performance Basics](.[executable]/[file]) - profiling methodology
- [Chapter 2: NVIDIA GPU Hardware](.[executable]/[file]) - hardware topology understanding

**Required knowledge**: Basic Linux system administration

## Examples

###  NUMA Binding for Multi-GPU

**Purpose**: Demonstrate proper CPU-GPU NUMA affinity binding for optimal performance.

**Problem**: Without NUMA binding, CPU threads may run on cores far from their target GPU, causing:
- Increased PCIe latency for H2D/D2H transfers
- Cross-NUMA memory traffic
- 10-30% performance degradation

**Solution**: Bind processes to CPUs in same NUMA node as target GPU:

```python
import os
import torch

def get_numa_node_for_gpu(gpu_id):
    """Get NUMA node for GPU."""
    # Read from sysfs
    numa_path = f"/sys/class/pci_bus/..[executable]"
    # Return NUMA node ID

def bind_to_numa_node(numa_node):
    """Bind current process to NUMA node."""
    [file]_setaffinity(0, get_cpus_for_numa(numa_node))
```

**How to run**:
```bash
python3 [script] --gpu 0

# Or for distributed training
torchrun [script]
```

**Expected impact**: **5-15% throughput improvement** for data-intensive workloads.

---

###  System-Level Optimizations

**Purpose**: Apply recommended system tuning for GPU workloads.

**What it configures**:

#### CPU Governor
```bash
# Set CPU to performance mode (disable frequency scaling)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```
**Impact**: Eliminates CPU frequency ramping delay (5-10% improvement)

#### Transparent Huge Pages (THP)
```bash
# Enable THP for large memory allocations
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```
**Impact**: Reduces TLB misses for large models (2-5% improvement)

#### PCIe Settings
```bash
# Set PCIe ASPM (Active State Power Management) to performance
setpci -s ${PCI_ADDR} [file]=0x40
```
**Impact**: Eliminates PCIe link state transition delays

#### IRQ Affinity
```bash
# Bind GPU interrupt handling to local NUMA node
echo ${CPU_MASK} > /proc/irq/${GPU_IRQ}/smp_affinity
```
**Impact**: Reduces interrupt handling latency

**How to run**:
```bash
sudo [executable].sh

# Verify settings
[executable].sh --verify
```

**Total expected impact**: **10-20% improvement** for multi-GPU workloads.

---

###  CPU-GPU Specific

**Purpose**: Additional tuning for NVIDIA GPU systems with Grace CPU.

**What it adds**:
- ARM-specific CPU governor settings
- CPU-GPU C2C link optimization
- Coherent memory pool configuration
- ARM PMU (Performance Monitoring Unit) setup

**How to run** (NVIDIA GPU only):
```bash
sudo [executable].sh
```

---

### 4. Container Configuration

**Purpose**: Dockerfile template with GPU optimizations.

**Key configurations**:

```dockerfile
# Use NVIDIA base image with CUDA 13 + PyTorch [file]
FROM [file]/nvidia/pytorch:[file]-py3

# Set allocator + arch flags for NVIDIA GPU (modern compute capability + PTX fallback)
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TORCH_CUDA_ARCH_LIST="[file]+PTX"
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Install NUMA tooling and utilities
RUN apt-get update && apt-get install -y numactl && rm -rf /var/lib/apt/lists/*

# Default entrypoint keeps NUMA interleave as example
ENTRYPOINT ["numactl", "--interleave=all", "python"]
```

**Build and run**:
```bash
docker build -f [file] -t gpu-optimized .
docker run --gpus all --ipc=host --ulimit memlock=-1 gpu-optimized
```

**Required flags**:
- `--gpus all`: GPU access
- `--ipc=host`: Shared memory for multi-process (NCCL)
- `--ulimit memlock=-1`: Unlimited pinned memory

---

### 5. K8s GPU Scheduling

**Purpose**: Kubernetes pod spec with topology-aware GPU scheduling.

**Key features**:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload
spec:
  containers:
  - name: training
    image: gpu-optimized:latest
    resources:
      limits:
        [file]/gpu: 8  # Request 8 GPUs
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: "all"
    volumeMounts:
    - name: shm
      mountPath: /dev/shm  # Shared memory for NCCL
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 64Gi  # Large SHM for multi-GPU
  nodeSelector:
    [file]/[file]: NVIDIA-NVIDIA GPU  # Target NVIDIA GPU nodes
  affinity:
    podAntiAffinity:  # Don't co-locate pods
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values: [training]
        topologyKey: [file]/hostname
```

**Deploy**:
```bash
kubectl apply -f [file]
```

---

###  Topology Discovery

**Purpose**: Display system NUMA topology for diagnostics.

**How to run**:
```bash
[executable].sh
```

**Expected output**:
```
System NUMA Topology:
Node 0: CPUs 0-35, GPUs 0-3
Node 1: CPUs 36-71, GPUs 4-7

GPU-to-NUMA Mapping:
GPU 0 -> NUMA Node 0 (CPUs 0-35)
GPU 1 -> NUMA Node 0 (CPUs 0-35)
...
```

---

## System Tuning Checklist

### Before Training/Inference

Run this checklist for optimal performance:

```bash
# 1. Set CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 2. Enable THP
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# 3. Disable NUMA balancing (can cause issues)
echo 0 | sudo tee /proc/sys/kernel/numa_balancing

# 4. Increase max map count (for large models)
echo 262144 | sudo tee /proc/sys/vm/max_map_count

# 5. Set memlock limits (for pinned memory)
ulimit -l unlimited

# 6. Verify GPU clocks (should be at max)
nvidia-smi -q -d CLOCK

# 7. Verify P2P access enabled
nvidia-smi topo -m
```

### Quick Validation Script

```bash
# Run comprehensive system check
cd ch3
python3 [script] --validate

# Expected output:
# [OK] CPU governor: performance
# [OK] THP enabled
# [OK] NUMA binding correct
# [OK] PCIe Gen5 active
# [OK] GPU clocks at maximum
```

---

## Performance Analysis

### Measuring System Configuration Impact

**Baseline (unconfigured system)**:
```bash
# Run without tuning
python3 .[executable]/[file] --benchmark
# Result: 1250 samples/sec
```

**Optimized (with system tuning)**:
```bash
# Apply system tuning
sudo [executable].sh

# Bind to NUMA
python3 [script] --gpu 0 .[executable]/[file] --benchmark
# Result: 1450 samples/sec
```

**Expected improvement**: **10-20%** (160% → 116% in this example)

### Common Issues and Diagnosis

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Variable performance | CPU frequency scaling | Set governor to `performance` |
| Slow H2D transfers | Wrong NUMA binding | Use `numactl` or [source file] |
| NCCL hangs | Insufficient shared memory | `--ipc=host` in Docker |
| OOM with pinned memory | memlock limit too low | `ulimit -l unlimited` |
| Poor multi-GPU scaling | Cross-NUMA traffic | Verify GPU-NUMA mapping |

---

## How to Run All Examples

```bash
cd ch3

# Install dependencies
pip install -r [file]

# 1. Check current system state
[executable].sh

# 2. Apply system tuning (requires sudo)
sudo [executable].sh

# 3. Verify tuning applied
[executable].sh --verify

# 4. Test NUMA binding
python3 [script] --gpu 0 --validate

# 5. NVIDIA GPU only: Apply Grace-specific tuning
sudo [executable].sh

# 6. Container examples
docker build -f [file] -t gpu-optimized .
docker run --gpus all --ipc=host gpu-optimized python3 -c "import torch; print([file].is_available())"
```

---

## Key Takeaways

1. **System tuning is invisible but impactful**: 10-20% improvements with no code changes!

2. **NUMA binding matters for multi-GPU**: Wrong NUMA placement causes 10-30% slowdown. Always bind processes to local NUMA node.

3. **CPU governor affects GPU workloads**: CPU frequency scaling delays kernel launches and data preparation. Use `performance` mode.

4. **Container configuration is critical**: Missing `--ipc=host` or memlock limits breaks NCCL and pinned memory.

5. **One-time setup, permanent benefit**: Apply these configurations once at system boot (add to startup scripts).

6. **Validate, don't assume**: Use validation scripts to verify tuning is applied and working.

---

## Common Pitfalls

### Pitfall 1: Forgetting to Reapply After Reboot
**Problem**: System tuning resets on reboot.

**Solution**: Add tuning script to systemd or `/etc/[file]`:
```bash
sudo cp [file] /usr/local/bin/
sudo systemctl enable gpu-[file]
```

### Pitfall 2: Wrong NUMA Binding in Multi-Process
**Problem**: All processes bound to same NUMA node → contention.

**Solution**: Bind each process to its GPU's NUMA node:
```bash
# Correct: Each GPU bound to its NUMA node
numactl --cpunodebind=0 python [file] --local_rank=0 &
numactl --cpunodebind=0 python [file] --local_rank=1 &
numactl --cpunodebind=1 python [file] --local_rank=4 &
```

### Pitfall 3: Insufficient Shared Memory in Docker
**Problem**: NCCL multi-GPU communication fails in containers.

**Solution**: Always use `--ipc=host` or mount large `/dev/shm`:
```bash
docker run --ipc=host ...  # Easiest
# OR
docker run --shm-size=64g ...  # Explicit size
```

### Pitfall 4: Not Disabling NUMA Balancing
**Problem**: Kernel automatically migrates pages between NUMA nodes → unpredictable performance.

**Solution**: Disable for GPU workloads:
```bash
echo 0 | sudo tee /proc/sys/kernel/numa_balancing
```

---

## Next Steps

**Continue the journey** → [Chapter 4: Multi-GPU Training](.[executable]/[file])

Learn about:
- NCCL collectives and communication primitives
- Tensor parallelism and pipeline parallelism
- NVSHMEM for fine-grained GPU communication
- Scaling from 1 to 8 GPUs efficiently

**Back to basics?** → [Chapter 1: Performance Basics](.[executable]/[file])

---

## Additional Resources

- **NUMA Documentation**: `man numa` and `man numactl`
- **Docker GPU Support**: [NVIDIA Container Toolkit](https://[file]/NVIDIA/nvidia-container-toolkit)
- **Kubernetes GPU**: [NVIDIA GPU Operator](https://[file].com/datacenter/cloud-native/gpu-operator/)
- **System Tuning Guide**: See `docs/[file]` in repository

---

**Chapter Status**: [OK] Complete


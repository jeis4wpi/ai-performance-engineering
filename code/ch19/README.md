# Chapter 19: FP4/FP6/FP8 Training and Quantization

## Overview

Low-precision training with FP4, FP6, and FP8 enables faster training and inference while reducing memory usage. This chapter covers NVIDIA's hardware-accelerated low-precision formats on NVIDIA GPU NVIDIA GPU/B300 GPUs, quantization techniques, dynamic precision switching, and production deployment patterns.

## Learning Objectives

After completing this chapter, you can:

- [OK] Implement FP4/FP6/FP8 quantization for training and inference
- [OK] Use dynamic precision switching for accuracy/performance tradeoffs
- [OK] Apply FP8 with Transformer Engine for production training
- [OK] Measure and optimize quantization performance
- [OK] Choose appropriate precision for different model components
- [OK] Deploy quantized models with 2-7x speedups

## Prerequisites

**Previous chapters**:
- [Chapter 10: Tensor Cores](.[executable]/[file]) - matrix operations with tensor cores
- [Chapter 16: Inference Optimization](.[executable]/[file]) - FP8 inference basics

**Required**: NVIDIA GPU NVIDIA GPU/B300 GPU (SM [file]) or NVIDIA GPU (SM [file]), PyTorch [file]+, CUDA [file]+

---

## Low-Precision Formats

### Format Comparison

| Precision | Bits | TFLOPS (NVIDIA GPU) | Memory vs FP16 | Best For |
|-----------|------|---------------|----------------|----------|
| **FP4 (E2M1)** | 4 | ~1600 | 75% savings (4x) | Draft models, speculative decoding |
| **FP6 (E3M2)** | 6 | ~1400 | 50% savings ([file]) | Balanced accuracy/compression |
| **FP8 (E4M3)** | 8 | ~450 | 50% savings (2x) | Production training & inference |
| **FP16** | 16 | ~225 | Baseline | Standard precision |
| **FP32** | 32 | ~225 | 2x memory | Legacy/debugging |

### Format Details

#### FP4 (E2M1) - NVFP4
- **Exponent**: 2 bits → Range: ~[[file], [file]]
- **Mantissa**: 1 bit + implicit leading 1
- **Use case**: Maximum compression, ~25% quantization error acceptable
- **NVIDIA GPU feature**: Hardware microscaling support

#### FP6 (E3M2) - NVFP6
- **Exponent**: 3 bits → Range: ~[[file], 60]
- **Mantissa**: 2 bits + implicit leading 1
- **Use case**: Better accuracy than FP4 (~[file]% error), still high compression
- **NVIDIA GPU feature**: Native hardware support

#### FP8 (E4M3FN) - NVFP8
- **Exponent**: 4 bits → Range: ~[2^-9, 448]
- **Mantissa**: 3 bits + implicit leading 1
- **Use case**: Production training/inference with minimal accuracy loss
- **NVIDIA GPU feature**: Full tensor core acceleration

---

## Examples

###  Production FP8 Training

**Purpose**: Full production FP8 training pipeline with scaling management.

**Key features**:
- `FP8ScalingManager` for numerical stability
- Transformer Engine integration
- Automatic loss scaling
- Mixed FP8/FP16 training

**How to run**:
```bash
# Basic training
python [file] --epochs 10

# With profiling
nsys profile [script]

# Validate against FP16
python [file] --epochs 10 --validate-fp16
```

**Expected results** (8x NVIDIA GPU):
```
FP16 training: 50ms/iteration, 2048 MB memory
FP8 training:  28ms/iteration, 1024 MB memory
Speedup: [file], Memory: 50% savings [OK]
Accuracy: <[file]% loss vs FP16
```

**Code structure**:
```python
from [file].amp import autocast
import [file] as nn

class FP8ScalingManager:
    """Manages scaling factors for FP8 training"""
    def __init__(self, init_scale=2**8):
        [file] = init_scale
        [file]_interval = 2000
    
    def update(self, grads_finite):
        if grads_finite:
            [file] *= [file]  # Grow slowly
        else:
            [file] *= [file]   # Shrink quickly on overflow

# Training loop
scaler = FP8ScalingManager()
for batch in dataloader:
    with autocast(dtype=[file]_e4m3fn):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    [file]()
    [file](check_finite([file]()))
    [file]()
```

**Performance on NVIDIA GPU**:
- **Training**: [file]-[file] faster than FP16
- **Memory**: 50% reduction
- **Accuracy**: <[file]% validation loss vs FP16
- **Scaling**: Linear to 8 GPUs

---

###  FP4 for Maximum Compression

**Purpose**: Implement FP4 (E2M1) for draft models and speculative decoding.

**Classes provided**:
- `FP4Tensor`: FP4 tensor with automatic dequantization
- `FP4Linear`: FP4 linear layer with optimized matmul
- `FP4MLP`: Complete FP4 multi-layer perceptron

**How to run**:
```bash
python [file]

# With benchmarking
python [file] --benchmark

# Profile
nsys profile [script]
```

**Expected results** (NVIDIA GPU):
```
FP32: [file], [file] TFLOPS
FP16: [file], [file] TFLOPS
FP8:  [file], [file] TFLOPS ([file] vs FP32)
FP4:  ~[file], ~54 TFLOPS (estimate, [file] vs FP32)
```

**Use cases**:
- [OK] Draft models for speculative decoding (7x more throughput)
- [OK] Multi-model serving (4x more models per GPU)
- [OK] Edge deployment (75% memory savings)
- ERROR: High-accuracy production (too much quantization error)

**Quantization error**:
- Typical: 20-30% L2 error on weights
- Acceptable for draft models where speed > accuracy
- Not suitable for final generation

---

###  FP6 Balanced Approach

**Purpose**: FP6 (E3M2) for better accuracy than FP4 with still-high compression.

**How to run**:
```bash
python [file] --benchmark
```

**Expected results**:
```
Memory savings: 50% ([file] compression)
TFLOPS: ~1400 on NVIDIA GPU
Quantization error: ~[file]% (vs ~25% for FP4)
Speedup: ~6x vs FP32
```

**Use cases**:
- [OK] Models where FP4 is too lossy but FP8 insufficient compression
- [OK] Intermediate draft/production scenarios
- [OK] Balance between accuracy and memory

---

###  FP8 with [file]

**Purpose**: Combine FP8 with `[file]` for maximum performance.

**How to run**:
```bash
python [file]

# With profiling
ncu --set full -o fp8_compiled_ncu --launch-skip 5 --launch-count 10 python [file]
```

**Expected speedup**:
```
FP16 (eager):     [file]
FP16 (compiled):  [file] ([file])
FP8 (eager):      [file] ([file])
FP8 (compiled):   [file] ([file]) [OK]
```

**Key optimizations**:
- Kernel fusion from `[file]`
- FP8 tensor core utilization
- Reduced memory bandwidth (half the data)

**Code**:
```python
@[file](mode='max-autotune')
def fp8_matmul(A, B):
    A_fp8 = [file]([file]_e4m3fn)
    B_fp8 = [file]([file]_e4m3fn)
    return [file](A_fp8, B_fp8).to([file])
```

---

###  Adaptive Precision

**Purpose**: Dynamically switch precision based on workload and accuracy requirements.

**Strategies**:
1. **Confidence-based**: High confidence → FP4, uncertain → FP16
2. **Layer-based**: Attention in FP8, FFN in FP4
3. **Token-based**: Important tokens in FP8, filler in FP4
4. **Memory-pressure**: FP8 normally, FP4 under memory pressure

**How to run**:
```bash
python [file] --strategy confidence

# Profile switching overhead
nsys profile [script]
```

**Example strategy**:
```python
class AdaptivePrecisionModel([file]):
    def forward(self, x, confidence=None):
        if confidence is not None and confidence > [file]:
            # High confidence: use FP4 for speed
            with autocast(dtype=[file]_e4m3fn):
                return [file]_fp4(x)
        else:
            # Low confidence: use FP8/FP16 for accuracy
            with autocast(dtype=[file]_e4m3fn):
                return [file]_fp8(x)
```

**Performance**:
- FP4 path: 7x faster, 5-10% accuracy loss
- FP8 path: 2x faster, <[file]% accuracy loss
- Adaptive: 4x average speedup, 2% accuracy loss

---

###  Per-Token Precision

**Purpose**: Switch precision per token based on importance.

**Strategy**:
```python
def forward_with_token_precision(self, input_ids, token_importance):
    # Important tokens (attention, special tokens): FP16/FP8
    important_mask = token_importance > threshold
    
    # Filler tokens (padding, common words): FP4
    filler_mask = ~important_mask
    
    # Process with different precisions
    important_output = [file]_fp8(input_ids[important_mask])
    filler_output = [file]_fp4(input_ids[filler_mask])
    
    return merge_outputs(important_output, filler_output)
```

**Use case**: Long context inference where most tokens don't need high precision.

**Expected**: 3-5x speedup for long sequences (>8K tokens)

---

###  Comprehensive Validation

**Purpose**: Automated validation framework with profiling and reporting.

**How to run**:
```bash
# Run all validations with profiling
python [file] --profile-all

# Single example
python [file] --example fp8_matmul --profile

# Generate report
python [file] --generate-report
```

**Output**:
```
[executable]/
├── [file]    # Comprehensive report
├── [file]              # Raw metrics
└── profiler_output/
    ├── [file]      # PyTorch profiler traces
    ├── [file]
    └── [file]
```

**Features**:
- [OK] NVTX markers for nsys integration
- [OK] Memory tracking (allocated, reserved, peak)
- [OK] TFLOPS calculation
- [OK] Speedup analysis
- [OK] Automated report generation

---

## Performance Analysis

### Expected Performance on NVIDIA GPU NVIDIA GPU

#### Transformer Layer (d=4096, ff=16384, seq=2048, batch=64)

| Precision | Time (ms) | Memory (MB) | Tokens/sec | Quality |
|-----------|-----------|-------------|------------|---------|
| FP32      | ~100      | 4096        | [file]       | Baseline |
| FP16      | ~50       | 2048        | [file]       | Baseline |
| **FP8**   | **~28**   | **1024**    | **[file]**   | <[file]% loss |
| **FP6**   | **~32**   | **1024**    | **[file]**   | ~1% loss |
| **FP4**   | **~15**   | **512**     | **[file]**   | 5-10% loss |

#### GEMM Performance (M=N=K=4096)

| Precision | Time (ms) | TFLOPS | Memory (MB) | Speedup |
|-----------|-----------|--------|-------------|---------|
| FP32      | ~[file]      | 225    | 512         | [file]    |
| FP16      | ~[file]      | 450    | 256         | [file]    |
| **FP8**   | **~[file]**  | **450**| **128**     | **[file]** |
| **FP4**   | **~[file]**  | **1600**| **64**     | **[file]** |

### Measured Results (NVIDIA GPU, SM [file])

From [source file]:
```
[OK] FP8 validation complete!
   Expected: 450 TFLOPS on NVIDIA GPU (vs 225 TFLOPS FP16)
   Actual FP32: [file] TFLOPS
   Actual FP16: [file] TFLOPS
   Actual FP8:  [file] TFLOPS

Speedup Analysis:
  FP8 vs FP32: [file] faster
  FP8 throughput gain: [file]
```

*Note: NVIDIA GPU (SM [file]) has different absolute performance than NVIDIA GPU (SM [file]) but demonstrates FP8 speedup ratios*

---

## Use Cases by Precision

### FP4 (NVFP4)
**Best for**:
- [OK] Draft models for speculative decoding (7x speedup)
- [OK] Cost-optimized large-scale inference
- [OK] Edge deployment (75% memory savings)
- [OK] Multi-model serving (4x more models per GPU)

**Avoid for**:
- ERROR: High-accuracy production models
- ERROR: Training (too low precision for gradients)

### FP6 (NVFP6)
**Best for**:
- [OK] Balance between FP4 compression and FP8 accuracy
- [OK] Models where FP4 too lossy but FP8 insufficient compression
- [OK] Intermediate draft/production scenarios

### FP8 (NVFP8)
**Best for**:
- [OK] Production LLM training ([file]-[file] speedup)
- [OK] Production inference with minimal accuracy loss
- [OK] Memory-constrained training (50% savings)
- [OK] High-throughput serving
- [OK] KV cache quantization

---

## How to Run All Examples

```bash
cd ch19

# Install dependencies
pip install torch>=[file].0 numpy

# 1. Validate all precisions
python [file] --profile-all

# 2. FP4 examples
python [file] --benchmark

# 3. FP6 examples
python [file] --benchmark

# 4. FP8 training
python [file] --epochs 10

# 5. FP8 with compile
python [file]

# 6. Dynamic precision
python [file] --strategy confidence

# 7. Token-level precision
python [file]

# Profile with nsys
nsys profile -o fp8_training --trace=cuda,nvtx,osrt,cudnn,cublas \
    python [file]

# Profile with ncu
ncu --set full -o fp8_ncu --launch-skip 5 --launch-count 10 \
    python [file]
```

---

## Key Takeaways

1. **FP8 is production-ready**: 2x speedup with <[file]% accuracy loss on NVIDIA GPU.

2. **Memory savings enable larger models**: 50% reduction allows 2x batch size or longer sequences.

3. **FP4 for draft models**: 7x speedup for speculative decoding where accuracy is secondary.

4. **Dynamic precision switching**: Adaptive strategies can achieve 4x average speedup.

5. **Tensor cores required**: These speedups only apply on NVIDIA GPU/Hopper GPUs with FP8 tensor cores.

6. **Scaling management critical**: FP8 training requires careful loss scaling to avoid under/overflow.

7. **Profile to validate**: Always measure actual speedup on your hardware and workload.

---

## Common Pitfalls

### Pitfall 1: Not Using Scaling with FP8
**Problem**: FP8 overflow/underflow without proper scaling → NaN losses.

**Solution**: Use `FP8ScalingManager` or Transformer Engine:
```python
scaler = FP8ScalingManager()
[file]()
[file](check_finite([file]()))
```

### Pitfall 2: Quantizing Everything to FP4
**Problem**: 25% quantization error on all layers → Poor quality.

**Solution**: Use FP4 only where acceptable (draft models, less critical layers).

### Pitfall 3: Not Checking Hardware Support
**Problem**: Running FP8 on GPU without tensor core support → Slow emulation.

**Check**:
```python
if [file].get_device_capability() < (9, 0):  # Hopper/NVIDIA GPU
    print("FP8 tensor cores not available!")
```

### Pitfall 4: Ignoring Memory Layout
**Problem**: Quantized tensors with poor memory layout → No speedup.

**Solution**: Use contiguous tensors and align to 128-byte boundaries:
```python
tensor = [file]()
```

### Pitfall 5: Over-Quantizing KV Cache
**Problem**: Quantizing KV cache to FP4 → Attention accuracy degraded.

**Solution**: Use FP8 for KV cache (2x savings, <1% accuracy loss):
```python
k_cache = [file]([file]_e4m3fn)
v_cache = [file]([file]_e4m3fn)
```

---

## Profiling and Debugging

### PyTorch Profiler
```bash
python [file] --example fp8_matmul --profile
```

View Chrome trace: `chrome://tracing` → Load `profiler_output/[file]`

### NVIDIA Nsight Systems
```bash
nsys profile -o fp8_profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    python [file]

# View
nsys profile [script]-rep
```

**Look for**:
- Tensor core utilization (should be >80%)
- Memory bandwidth (should be <50% for compute-bound)
- Launch overhead (should be minimal with FP8 batching)

### NVIDIA Nsight Compute
```bash
ncu --set full -o fp8_ncu \
    --target-processes all \
    --launch-skip 5 --launch-count 10 \
    python [file]

# Roofline analysis
ncu --set roofline -o fp8_roofline python [file]

# View
ncu-ui [file]-rep
```

---

## Next Steps

**Final chapter** → [Chapter 20: Putting It All Together](.[executable]/[file])

Learn about:
- End-to-end optimization workflows
- Real-world case studies combining FP8 + other techniques
- Production deployment patterns

**Related advanced topic** → [Chapter 18: Advanced Attention](.[executable]/[file])
- FlashAttention with FP8
- MLA (Multi-head Latent Attention) for reduced KV cache

---

## Additional Resources

- **NVIDIA Transformer Engine**: [GitHub](https://[file]/NVIDIA/TransformerEngine)
- **NVIDIA GPU Architecture**: [NVIDIA Whitepaper](https://[file].com/en-us/data-center/technologies/blackwell-architecture/)
- **FP8 Training Guide**: [NVIDIA Developer Blog](https://[file].com/blog/fp8-training)
- **Quantization Survey**: [arXiv:[file]](https://[file]/abs/[file])

### Additional Insights

**Weight-only quantization** (GPTQ, AWQ):
- Activation quantization (SmoothQuant)
- KV cache compression with FP4/FP8
- FP4 dynamic range and scaling
- Memory savings: FP16 weights + FP8 activations → 50%, INT4 weights + FP4 activations → 20%

**Dynamic precision strategies**:
- Dynamic precision strategy based on confidence
- KV cache: FP8 normally, FP4 under memory pressure
- Compute-limited: FP8 achieves 2× speedup
- Memory-bound: [file]× achievable with FP8

---

**Chapter Status**: [OK] Complete

---

## Reference Materials

**For batched GEMM techniques** (cuBLAS batched operations, grouped GEMM for MoE), see `[file]` in this directory. While not directly related to FP8 training, batched operations can be combined with FP8 for maximum performance in multi-head attention and MoE layers.


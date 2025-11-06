# AI Systems Performance Engineering: Code

Production playbook for standing up, validating, and tuning PyTorch LLM workloads on 8x NVIDIA B200 systems.

---

## Overview
**Target hardware:** 

- NVIDIA Blackwell (B200/B300, sm100)
- Grace Blackwell (GB200/GB300, sm103)
- DGX Spark (GB10, sm121)

**Reference stack:** CUDA 13+, PyTorch 2.9+, Triton 3.5+, and Python 3.10+

The repository packages everything needed to:
- Provision a reproducible software stack (`setup.sh`) for new lab machines.
- Exercise and benchmark the platform end-to-end before deploying workloads.

## Quick Start

### Prerequisites
- Root access to the host (the setup script installs NVIDIA driver 580+, CUDA 13.0, and dependencies)
- Python 3.10+ on the path (the setup script installs required packages in-place)
- Network access to fetch Python wheels and Nsight tooling

### Setup
1. Clone and enter the repository:
   ```bash
   git clone <repo-url> && cd ai-performance-engineering/code
   ```
2. Run the automated bootstrap:
   ```bash
   sudo ./setup.sh
   ```
3. If the script upgrades the driver, reboot and rerun `sudo ./setup.sh` to finish verification.


## Verification & Testing

### Quick Verification
Run the quick smoke tests after installation:
1. Confirm the hardware and driver:
   ```bash
   nvidia-smi
   ```
   Expect at least 1 Blackwell GPU and driver 580+.
2. Verify benchmarks can load (syntax + import check):
   ```bash
   python3 tools/verification/verify_all_benchmarks.py
   ```

### Running All Benchmarks
**Use `benchmark.py` - it's the unified entry point for running benchmarks:**

```bash
# Run ALL benchmarks (discover + run + summarize)
python benchmark.py

# Run single chapter (accepts number or ch prefix)
python benchmark.py --chapter 12
python benchmark.py --chapter ch12
```

**What it does:**
- Discovers all `baseline_*.py` / `optimized_*.py` pairs across all chapters
- Runs actual benchmarks using BenchmarkHarness
- Measures performance (baseline vs optimized) and calculates speedups
- **Automatically detects and skips hardware/software limitations** with clear notifications
- Resets CUDA state between benchmarks to prevent cascading failures
- Generates summary reports:
  - `benchmark_test_results.json` - Machine-readable detailed results
  - `benchmark_test_results.md` - Human-readable markdown summary (includes skipped benchmarks)

**Exit codes:** `0` = all passed, `1` = some failed (perfect for CI/CD)

**Hardware Limitations:** Benchmarks that cannot run due to hardware/software incompatibilities are automatically skipped with clear notifications.

### Peak Performance Validation
During `setup.sh`, the system automatically runs `benchmark_peak.py` to capture actual peak hardware performance metrics:
- HBM memory bandwidth
- FP4 compute TFLOPS (if available)
- FP6 compute TFLOPS (if available)
- FP8 compute TFLOPS (if available)
- FP16 compute TFLOPS
- L2 cache bandwidth
- Shared memory (L1-equivalent) characteristics
- GPU hardware information (SMs, cache sizes, registers, etc.)
- NVLink bandwidth (if multi-GPU available)
- torch.compile speedup

These measured values are saved to `benchmark_peak_results_*.json` and used as dynamic performance targets instead of hardcoded values. The `performance_targets.py` system automatically loads these measured values and uses them for validation.

**Automatic execution**: If `benchmark_peak_results_*.json` files don't exist, `benchmark.py` will automatically run peak detection (~30-60 seconds) before running benchmarks. The system gracefully continues even if peak detection fails.

To manually re-run peak benchmarks:
```bash
python tools/benchmarking/benchmark_peak.py
```

**Note**: Use `python benchmark.py` for all benchmark testing and comparison.

## Repository Layout
```text
code/
├── setup.sh                # End-to-end system bootstrap
├── ch1...ch20/             # Chapter walkthroughs with focused READMEs
├── scripts/                # Capture and profiling helpers
├── tools/                  # Verification utilities
└── tests/                  # PyTorch regression tests (`pytest -v tests/`)
```

## Cleanup Generated Artifacts
Remove generated artifacts, caches, and binaries:
```bash
python cleanup.py
```

## Next Steps
- Record measured metrics or new findings for future reference
- For questions or new issues, escalate via the team's issue tracker

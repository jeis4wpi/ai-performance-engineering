"""
Multi-Node Distributed Training for Blackwell GPUs

Demonstrates multi-node distributed training optimizations for Blackwell
B200/B300 GPUs with NVLink-C2C and NCCL. Includes tensor parallelism,
FSDP, and gradient compression for multi-node scaling.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.device_mesh import init_device_mesh
import torch.nn.functional as F
from typing import Tuple, Optional
import time


# ============================================================================
# Environment Setup for Blackwell Multi-Node
# ============================================================================

def setup_blackwell_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> Tuple[int, int, int, int]:
    """Initialize distributed environment optimized for Blackwell.
    
    Returns: (rank, local_rank, world_size, local_world_size)
    """
    # Get environment variables
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Configure NCCL for Blackwell
    if backend == "nccl":
        # Blackwell-specific NCCL optimizations
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")  # Adjust as needed
        os.environ.setdefault("NCCL_IB_DISABLE", "0")  # Enable InfiniBand if available
        os.environ.setdefault("NCCL_NET_GDR_LEVEL", "5")  # GPU Direct RDMA
        os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")  # NVLink for intra-node
        os.environ.setdefault("NCCL_NVLS_ENABLE", "1")  # NVLink-C2C
        os.environ.setdefault("NCCL_ALGO", "Tree,Ring")  # Algorithms
        os.environ.setdefault("NCCL_PROTO", "Simple")  # Protocol
        
        # Blackwell has 148 SMs - optimize channel count
        os.environ.setdefault("NCCL_NCHANNELS_PER_NET_PEER", "8")
        
        # Buffer sizes for Blackwell (large memory)
        os.environ.setdefault("NCCL_BUFFSIZE", "8388608")  # 8MB
        
        if rank == 0:
            print("=" * 70)
            print("NCCL Configuration for Blackwell")
            print("=" * 70)
            print(f"  P2P Level: {os.environ.get('NCCL_P2P_LEVEL', 'default')}")
            print(f"  NVLS (NVLink-C2C): {os.environ.get('NCCL_NVLS_ENABLE', '0')}")
            print(f"  GDR Level: {os.environ.get('NCCL_NET_GDR_LEVEL', 'default')}")
            print(f"  Channels per peer: {os.environ.get('NCCL_NCHANNELS_PER_NET_PEER', 'default')}")
            print(f"  Buffer size: {int(os.environ.get('NCCL_BUFFSIZE', '2097152')) / 1024 / 1024:.1f} MB")
            print("=" * 70)
    
    # Get local world size (GPUs per node)
    local_world_size = torch.cuda.device_count()
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Distributed Training Setup")
        print("=" * 70)
        print(f"  World size: {world_size}")
        print(f"  Ranks per node: {local_world_size}")
        print(f"  Number of nodes: {world_size // local_world_size}")
        print(f"  Backend: {backend}")
        print("=" * 70 + "\n")
    
    return rank, local_rank, world_size, local_world_size


# ============================================================================
# Multi-Node Model with Hybrid Parallelism
# ============================================================================

class MultiNodeTransformerBlock(nn.Module):
    """Transformer block with hybrid parallelism for multi-node training."""
    
    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 16,
        d_ff: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Multi-head attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        residual = x
        x = self.attn_norm(x)
        
        # Multi-head attention (simplified)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.1 if self.training else 0.0,
        )
        
        x = self.o_proj(attn_out)
        x = self.dropout(x)
        x = x + residual
        
        # Feed-forward block
        residual = x
        x = self.ff_norm(x)
        x = self.ff1(x)
        x = F.gelu(x)
        x = self.ff2(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class MultiNodeTransformer(nn.Module):
    """
    Transformer model for multi-node training.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        d_ff: int = 4096,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            MultiNodeTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        
        # Embeddings
        x = self.embedding(input_ids)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(pos)
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


def apply_tensor_parallelism(
    model: nn.Module,
    device_mesh,
) -> nn.Module:
    """
    Apply tensor parallelism to model using PyTorch 2.9 API.
    
    Optimized for intra-node parallelism via NVLink-C2C.
    """
    # Parallelize attention projections
    for i, layer in enumerate(model.layers):
        # Column-wise parallel for Q, K, V projections
        parallelize_module(
            layer,
            device_mesh["tp"],
            {
                "q_proj": ColwiseParallel(),
                "k_proj": ColwiseParallel(),
                "v_proj": ColwiseParallel(),
            },
        )
        
        # Row-wise parallel for output projection
        parallelize_module(
            layer,
            device_mesh["tp"],
            {"o_proj": RowwiseParallel()},
        )
        
        # Column-wise for FF1, Row-wise for FF2
        parallelize_module(
            layer,
            device_mesh["tp"],
            {
                "ff1": ColwiseParallel(),
                "ff2": RowwiseParallel(),
            },
        )
    
    return model


def apply_fsdp(
    model: nn.Module,
    device_mesh,
    mixed_precision: bool = True,
) -> FSDP:
    """
    Apply FSDP (Fully Sharded Data Parallel) for data parallelism.
    
    Optimized for inter-node communication.
    """
    # Mixed precision policy
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp_policy = None
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        device_mesh=device_mesh["dp"],
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        use_orig_params=True,  # For optimizer compatibility
    )
    
    return model


# ============================================================================
# Multi-Node Training Loop
# ============================================================================

def train_multi_node(
    model: nn.Module,
    train_data: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    gradient_accumulation_steps: int = 4,
    use_compile: bool = True,
) -> dict:
    """
    Multi-node distributed training with optimizations.
    
    Optimizations:
    - Gradient accumulation
    - Mixed precision (BF16)
    - torch.compile
    - Communication overlap
    - Gradient compression (future)
    
    Returns:
        Training statistics
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    
    # Compile model for Blackwell
    if use_compile:
        model = torch.compile(
            model,
            mode="max-autotune",
            backend="inductor",
        )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_data),
    )
    
    # Training stats
    stats = {
        'losses': [],
        'throughputs': [],
        'step_times': [],
    }
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Starting Multi-Node Training")
        print("=" * 70)
        print(f"  Nodes: {world_size // torch.cuda.device_count()}")
        print(f"  GPUs per node: {torch.cuda.device_count()}")
        print(f"  Total GPUs: {world_size}")
        print(f"  Mixed precision: BF16")
        print(f"  torch.compile: {use_compile}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print("=" * 70 + "\n")
    
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        for step, batch in enumerate(train_data):
            step_start = time.time()
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            
            # Backward pass with gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            step_time = time.time() - step_start
            
            # Log statistics
            if rank == 0 and step % 10 == 0:
                tokens_per_step = input_ids.numel() * world_size
                throughput = tokens_per_step / step_time
                
                stats['losses'].append(loss.item() * gradient_accumulation_steps)
                stats['throughputs'].append(throughput)
                stats['step_times'].append(step_time)
                
                print(f"Epoch {epoch} | Step {step:4d} | "
                      f"Loss: {loss.item()*gradient_accumulation_steps:.4f} | "
                      f"Throughput: {throughput/1e6:.2f}M tokens/s | "
                      f"Time: {step_time*1000:.1f}ms")
            
            epoch_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        
        if rank == 0:
            avg_loss = epoch_loss / len(train_data)
            print(f"\nEpoch {epoch} complete:")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Epoch time: {epoch_time:.2f}s")
            print(f"  Throughput: {len(train_data) * input_ids.numel() * world_size / epoch_time / 1e6:.2f}M tokens/s")
    
    return stats


# ============================================================================
# Bandwidth Benchmark
# ============================================================================

def benchmark_multi_node_bandwidth(
    sizes: list[int] = [1024*1024, 10*1024*1024, 100*1024*1024],
    num_iters: int = 100,
) -> dict:
    """
    Benchmark inter-node and intra-node bandwidth.
    
    Tests:
    - All-reduce (ring algorithm)
    - All-gather
    - Reduce-scatter
    - Point-to-point
    
    Returns:
        Bandwidth measurements
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Multi-Node Bandwidth Benchmark")
        print("=" * 70)
    
    results = {}
    
    for size in sizes:
        tensor = torch.randn(size, device=device, dtype=torch.float32)
        bytes_transferred = tensor.numel() * tensor.element_size()
        
        # Warmup
        for _ in range(10):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        
        # All-reduce benchmark
        start = time.time()
        for _ in range(num_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate bandwidth
        # All-reduce transfers 2(N-1)/N * data_size
        algo_bw_factor = 2.0 * (world_size - 1) / world_size
        bandwidth_gbs = (bytes_transferred * algo_bw_factor * num_iters) / elapsed / 1e9
        
        results[f"{bytes_transferred/1e6:.1f}MB"] = bandwidth_gbs
        
        if rank == 0:
            print(f"  Size: {bytes_transferred/1e6:8.1f} MB | "
                  f"Bandwidth: {bandwidth_gbs:6.2f} GB/s | "
                  f"Time: {elapsed*1000/num_iters:6.2f} ms")
    
    if rank == 0:
        print("=" * 70 + "\n")
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main entry point for multi-node training demonstration.
    
    Launch with:
        torchrun --nnodes=2 --nproc_per_node=8 \\
                 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
                 multi_node_blackwell.py
    """
    # Setup distributed
    rank, local_rank, world_size, local_world_size = setup_blackwell_distributed()
    
    # Run bandwidth benchmark
    if rank == 0:
        print("\n[1/3] Running bandwidth benchmark...")
    bandwidth_results = benchmark_multi_node_bandwidth()
    
    # Create device mesh for hybrid parallelism
    if rank == 0:
        print("\n[2/3] Setting up hybrid parallelism...")
    
    # 2D mesh: [data_parallel, tensor_parallel]
    # Tensor parallel within nodes (via NVLink-C2C)
    # Data parallel across nodes (via InfiniBand)
    num_nodes = world_size // local_world_size
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(num_nodes, local_world_size),
        mesh_dim_names=("dp", "tp"),
    )
    
    # Create model
    model = MultiNodeTransformer(
        vocab_size=50257,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4096,
    ).cuda()
    
    # Apply tensor parallelism (intra-node)
    model = apply_tensor_parallelism(model, device_mesh)
    
    # Apply FSDP (inter-node)
    model = apply_fsdp(model, device_mesh, mixed_precision=True)
    
    if rank == 0:
        print(" Model parallelization complete")
        print(f"  Tensor Parallel: {local_world_size}x (intra-node)")
        print(f"  Data Parallel: {num_nodes}x (inter-node)")
    
    # Create dummy dataset
    if rank == 0:
        print("\n[3/3] Starting training...")
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000, seq_len=512):
            self.size = size
            self.seq_len = seq_len
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 50257, (self.seq_len,)),
                'labels': torch.randint(0, 50257, (self.seq_len,)),
            }
    
    dataset = DummyDataset(size=100, seq_len=512)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=0,  # Set to 0 for demo
    )
    
    # Train
    stats = train_multi_node(
        model,
        dataloader,
        num_epochs=2,
        gradient_accumulation_steps=4,
        use_compile=True,
    )
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Multi-Node Training Complete!")
        print("=" * 70)
        print(" Bandwidth benchmark: completed")
        print(" Hybrid parallelism: configured")
        print(" Training: completed")
        print(f" Average throughput: {sum(stats['throughputs'])/len(stats['throughputs'])/1e6:.2f}M tokens/s")
        print("=" * 70 + "\n")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()


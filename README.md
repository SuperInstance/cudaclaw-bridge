# cudaclaw-bridge

**Rust bridge between the Flux→PTX oxide compilation pipeline and cudaclaw's GPU execution runtime.**

Deploy compiled PTX modules as persistent CUDA kernels, hotswap them live without dropping the device context, and track every byte of VRAM and every microsecond of latency while cudaclaw's warp-level consensus keeps your kernels coherent across the SIMT array.

---

## Table of Contents

- [What This Is](#what-this-is)
- [PTX Module Deployment](#ptx-module-deployment)
- [Hotswap: Live Kernel Replacement](#hotswap-live-kernel-replacement)
- [VRAM Tracking & Worker Management](#vram-tracking--worker-management)
- [Kernel Statistics](#kernel-statistics)
- [Code Examples](#code-examples)
- [Warp-Level Consensus](#warp-level-consensus)
- [DeployStatus State Machine](#deploystatus-state-machine)

---

## What This Is

The oxide compiler turns your Flux kernels into PTX. **cudaclaw-bridge** is the load-bearing structure that gets that PTX onto the GPU and keeps it there.

It is not a driver. It is not a runtime. It is the **contract layer** between the compiler's output and cudaclaw's execution engine. The bridge owns:

- **Worker allocation** — mapping kernels to persistent GPU worker slots.
- **VRAM accounting** — every module is sized before upload; over-subscription is rejected at the gate.
- **Live hotswap** — replace a kernel's PTX module in-place without stopping the worker or freeing device memory.
- **Telemetry** — invocation counts, cumulative latency, error rates, and throughput estimates.

If you are running cudaclaw, you are running this bridge.

---

## PTX Module Deployment

A `PtxModule` is the compiled artifact the bridge consumes:

```rust
use cudaclaw_bridge::PtxModule;

let module = PtxModule {
    ptx: ptx_bytes,                       // Compiled PTX as raw bytes
    kernel_name: "flash_attention".into(),
    grid_dim: (64, 1, 1),                // Grid shape
    block_dim: (256, 1, 1),              // Block shape
    shared_mem_bytes: 49152,             // Dynamic shared memory per block
    min_compute_capability: 80,          // sm_80 minimum
};
```

Deployment is **persistent**. Once a module passes VRAM validation and claims a worker, it stays resident until you explicitly stop it or hotswap it. The bridge does not treat kernels as fire-and-forter jobs; it treats them as **long-running services** on the device.

```rust
use cudaclaw_bridge::CudaclawBridge;

let mut bridge = CudaclawBridge::new(
    max_workers: 16,      // GPU worker slots
    total_vram_mb: 24576, // 24 GB budget
);

let kernel_id = bridge.deploy(module)?;
// kernel_id == "kernel-0"
```

---

## Hotswap: Live Kernel Replacement

Stopping a kernel to reload PTX costs context switches, pipeline flushes, and precious milliseconds. The bridge supports **in-place hotswap**: the worker keeps its slot, the old module metadata is overwritten, and the new PTX is live on the next launch.

```rust
// v1 is running on worker 3
bridge.hotswap(&kernel_id, v2_module)?;
// v2 is now running on worker 3 — no teardown, no re-allocation
```

Hotswap preserves:
- Worker ID and slot reservation
- Kernel statistics (cumulative counters continue)
- VRAM footprint (re-validated; fails fast if the new module exceeds budget)

Hotswap is only valid on kernels in `Running` status. Attempting to hotswap a `Draining` or `Stopped` kernel returns `BridgeError::InvalidStatus`.

---

## VRAM Tracking & Worker Management

The bridge maintains two hard limits:

| Resource | Enforcement |
|----------|-------------|
| **Workers** | Fixed pool (`max_workers`). `deploy()` returns `NoAvailableWorkers` when exhausted. |
| **VRAM**    | Fixed ceiling (`total_vram_mb`). `deploy()` returns `InsufficientVram` when the module would over-subscribe. |

VRAM is estimated heuristically from PTX size and block dimensions:

```
estimate_vram = ceil(ptx_bytes / 1024) + (block_size * 4 / 1024)
```

This is a **conservative lower bound**, not a full CUDA driver allocation trace. It is fast and safe for admission control. If you need byte-exact accounting, instrument at the driver layer and feed `update_stats`.

Stopping a kernel returns its worker to the pool and its VRAM to the budget:

```rust
bridge.stop(&kernel_id)?;   // worker freed, VRAM reclaimed
```

---

## Kernel Statistics

Every deployed kernel carries a `KernelStats` struct that accumulates runtime telemetry:

```rust
pub struct KernelStats {
    pub invocations: u64,        // Total kernel launches
    pub total_time_us: u64,      // Cumulative GPU time
    pub errors: u64,             // Failed launches / exceptions
    pub avg_time_us: f64,        // Derived: total / invocations
    pub throughput_ops_s: f64,   // Ops/sec (filled by caller/driver)
    pub gpu_utilization_pct: u8, // 0-100 (filled by caller/driver)
}
```

The bridge computes `avg_time_us` automatically. `throughput_ops_s` and `gpu_utilization_pct` are **write-through fields**: your telemetry agent updates them alongside invocation data.

```rust
bridge.update_stats(&kernel_id, invocations: 1000, time_us: 52000, errors: 0);
```

---

## Code Examples

### Deploy a kernel

```rust
use cudaclaw_bridge::{CudaclawBridge, PtxModule};

let mut bridge = CudaclawBridge::new(8, 8192);

let module = PtxModule {
    ptx: include_bytes!("attention.ptx").to_vec(),
    kernel_name: "warp_attn".into(),
    grid_dim: (32, 1, 1),
    block_dim: (128, 1, 1),
    shared_mem_bytes: 32768,
    min_compute_capability: 80,
};

let id = bridge.deploy(module).expect("deploy failed");
println!("Kernel {} running. VRAM: {:.1}%", id, bridge.vram_utilization());
```

### Hotswap without stopping

```rust
let v2 = PtxModule {
    ptx: include_bytes!("attention_v2.ptx").to_vec(),
    kernel_name: "warp_attn_v2".into(),
    grid_dim: (32, 1, 1),
    block_dim: (128, 1, 1),
    shared_mem_bytes: 32768,
    min_compute_capability: 80,
};

bridge.hotswap(&id, v2).expect("hotswap failed");
println!("Live-replaced {} with v2", id);
```

### Stop and reclaim

```rust
let final_status = bridge.stop(&id).expect("stop failed");
assert_eq!(final_status, DeployStatus::Stopped);
println!("Workers available: {}", bridge.available_workers());
```

### Monitor

```rust
if let Some(k) = bridge.get_kernel(&id) {
    println!("{}: {} invocations, {:.1} us avg, {} errors",
        k.id, k.stats.invocations, k.stats.avg_time_us, k.stats.errors);
}

for k in bridge.running_kernels() {
    println!("Active: {} on worker {:?}", k.id, k.worker_id);
}
```

---

## Warp-Level Consensus

This bridge does not implement consensus. **cudaclaw does.**

The bridge's job is to get PTX onto the device and keep the metadata straight. Once launched, the kernel executes inside cudaclaw's warp-level consensus fabric: intra-warp ballot voting, cooperative groups for grid-wide barriers, and divergence-aware scheduling.

The bridge stays out of the SIMT. It handles the **host-side orchestration** so cudaclaw can focus on the **device-side execution**. If your kernel hangs, the bridge's stats will show zero progress; the fix is in cudaclaw, not here. If your kernel fails to upload, the fix is in the bridge.

Know your layer.

---

## DeployStatus State Machine

```
                    ┌───────────┐
         deploy()   │  Compiled │  (initial compiler output)
                    └─────┬─────┘
                          │ upload
                          ▼
                    ┌───────────┐
                    │  Uploaded │  (PTX resident in driver)
                    └─────┬─────┘
                          │ assign worker
                          ▼
               ┌────────────────────┐
         ┌────►│ Running { worker } │◄────────────────┐
         │     └─────────┬──────────┘                 │
         │               │                            │
    stop()          hotswap()                    hotswap()
         │               │                            │
         │     ┌─────────▼──────────┐                 │
         │     │ Running { worker } │  (new module)   │
         │     │   (new PTX)        │                 │
         │     └────────────────────┘                 │
         │               │                            │
         │               │ drain                      │
         │               ▼                            │
         │     ┌─────────────┐                        │
         └─────┤   Draining  │────────────────────────┘
               └──────┬──────┘   (finish in-flight, reclaim)
                      │
                 stop()
                      ▼
               ┌─────────────┐
               │   Stopped   │  (worker & VRAM freed)
               └─────────────┘
                      ▲
                      │
              any stage failure
                      │
               ┌──────┴──────┐
               │Failed(String)│
               └─────────────┘
```

### State Definitions

| State | Meaning |
|-------|---------|
| `Compiled` | PTX has been emitted by the oxide pipeline but not yet handed to the driver. |
| `Uploaded` | PTX bytes are loaded into the CUDA driver context; no worker assigned yet. |
| `Running { worker_id }` | Kernel is bound to a worker slot and eligible for launch. |
| `Draining` | Worker is finishing in-flight grid launches; no new work accepted. |
| `Stopped` | Worker and VRAM fully reclaimed. Terminal state. |
| `Failed(String)` | Terminal error (empty PTX, OOM, invalid status transition, etc.). |

---

## Error Handling

All fallible operations return `BridgeError`:

| Variant | Trigger |
|---------|---------|
| `EmptyPtx` | `deploy()` called with zero-length PTX. |
| `NoAvailableWorkers` | Worker pool exhausted. |
| `InsufficientVram { required, available }` | Module would exceed VRAM budget. |
| `KernelNotFound(id)` | Operation references a non-existent kernel ID. |
| `InvalidStatus { expected, actual }` | State transition violation (e.g., hotswapping a `Stopped` kernel). |
| `HotswapFailed(reason)` | Reserved for driver-level hotswap failures. |

---

## License

Apache-2.0

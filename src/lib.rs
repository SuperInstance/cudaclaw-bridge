//! # cudaclaw-bridge
//!
//! Bridge between the Flux→PTX compilation pipeline and cudaclaw's GPU execution runtime.

use std::collections::HashMap;

/// A compiled PTX module ready for deployment.
#[derive(Debug, Clone)]
pub struct PtxModule {
    pub ptx: Vec<u8>,
    pub kernel_name: String,
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
    pub min_compute_capability: u32,
}

/// Kernel deployment status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeployStatus {
    Compiled,
    Uploaded,
    Running { worker_id: u32 },
    Draining,
    Stopped,
    Failed(String),
}

/// A deployed kernel with runtime metadata.
#[derive(Debug, Clone)]
pub struct DeployedKernel {
    pub id: String,
    pub module: PtxModule,
    pub status: DeployStatus,
    pub worker_id: Option<u32>,
    pub stats: KernelStats,
}

/// Runtime statistics for a deployed kernel.
#[derive(Debug, Clone, Default)]
pub struct KernelStats {
    pub invocations: u64,
    pub total_time_us: u64,
    pub errors: u64,
    pub avg_time_us: f64,
    pub throughput_ops_s: f64,
    pub gpu_utilization_pct: u8,
}

fn estimate_vram(module: &PtxModule) -> u32 {
    let base = module.ptx.len() as u32 / 1024 + 1;
    let blocks = module.block_dim.0 * module.block_dim.1 * module.block_dim.2;
    base + (blocks * 4 / 1024)
}

/// The bridge — connects the oxide pipeline to cudaclaw.
pub struct CudaclawBridge {
    deployed: HashMap<String, DeployedKernel>,
    max_workers: u32,
    next_kernel_id: u32,
    available_workers: Vec<u32>,
    total_vram_mb: u32,
    used_vram_mb: u32,
}

impl CudaclawBridge {
    pub fn new(max_workers: u32, total_vram_mb: u32) -> Self {
        Self {
            deployed: HashMap::new(),
            max_workers,
            next_kernel_id: 0,
            available_workers: (0..max_workers).collect(),
            total_vram_mb,
            used_vram_mb: 0,
        }
    }

    /// Deploy a PTX module as a persistent kernel.
    pub fn deploy(&mut self, module: PtxModule) -> Result<String, BridgeError> {
        if module.ptx.is_empty() {
            return Err(BridgeError::EmptyPtx);
        }

        let vram_needed = estimate_vram(&module);
        if self.used_vram_mb + vram_needed > self.total_vram_mb {
            return Err(BridgeError::InsufficientVram {
                required: vram_needed,
                available: self.total_vram_mb - self.used_vram_mb,
            });
        }

        let worker_id = self.available_workers.pop()
            .ok_or(BridgeError::NoAvailableWorkers)?;

        let id = format!("kernel-{}", self.next_kernel_id);
        self.next_kernel_id += 1;
        self.used_vram_mb += vram_needed;

        self.deployed.insert(id.clone(), DeployedKernel {
            id: id.clone(),
            module,
            status: DeployStatus::Running { worker_id },
            worker_id: Some(worker_id),
            stats: KernelStats::default(),
        });

        Ok(id)
    }

    /// Hotswap: replace a running kernel with new PTX.
    pub fn hotswap(&mut self, kernel_id: &str, new_module: PtxModule) -> Result<(), BridgeError> {
        let kernel = self.deployed.get(kernel_id)
            .ok_or_else(|| BridgeError::KernelNotFound(kernel_id.to_string()))?;

        let worker = match kernel.status {
            DeployStatus::Running { worker_id } => worker_id,
            _ => return Err(BridgeError::InvalidStatus {
                expected: "Running",
                actual: format!("{:?}", kernel.status),
            }),
        };

        let kernel = self.deployed.get_mut(kernel_id).unwrap();
        kernel.module = new_module;
        kernel.status = DeployStatus::Running { worker_id: worker };
        Ok(())
    }

    /// Stop a deployed kernel and free resources.
    pub fn stop(&mut self, kernel_id: &str) -> Result<DeployStatus, BridgeError> {
        // First gather info we need
        let (worker_id, vram) = {
            let kernel = self.deployed.get(kernel_id)
                .ok_or_else(|| BridgeError::KernelNotFound(kernel_id.to_string()))?;
            (kernel.worker_id, estimate_vram(&kernel.module))
        };

        // Now mutate
        if let Some(wid) = worker_id {
            self.available_workers.push(wid);
        }
        self.used_vram_mb = self.used_vram_mb.saturating_sub(vram);

        let kernel = self.deployed.get_mut(kernel_id).unwrap();
        kernel.status = DeployStatus::Stopped;
        kernel.worker_id = None;
        Ok(DeployStatus::Stopped)
    }

    /// Update kernel statistics.
    pub fn update_stats(&mut self, kernel_id: &str, invocations: u64, time_us: u64, errors: u64) {
        if let Some(kernel) = self.deployed.get_mut(kernel_id) {
            kernel.stats.invocations += invocations;
            kernel.stats.total_time_us += time_us;
            kernel.stats.errors += errors;
            kernel.stats.avg_time_us = if kernel.stats.invocations > 0 {
                kernel.stats.total_time_us as f64 / kernel.stats.invocations as f64
            } else { 0.0 };
        }
    }

    pub fn get_kernel(&self, kernel_id: &str) -> Option<&DeployedKernel> {
        self.deployed.get(kernel_id)
    }

    pub fn running_kernels(&self) -> Vec<&DeployedKernel> {
        self.deployed.values()
            .filter(|k| matches!(k.status, DeployStatus::Running { .. }))
            .collect()
    }

    pub fn available_workers(&self) -> usize {
        self.available_workers.len()
    }

    pub fn vram_utilization(&self) -> f64 {
        if self.total_vram_mb == 0 { return 0.0; }
        self.used_vram_mb as f64 / self.total_vram_mb as f64 * 100.0
    }
}

/// Bridge errors.
#[derive(Debug, Clone)]
pub enum BridgeError {
    EmptyPtx,
    NoAvailableWorkers,
    InsufficientVram { required: u32, available: u32 },
    KernelNotFound(String),
    InvalidStatus { expected: &'static str, actual: String },
    HotswapFailed(String),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPtx => write!(f, "PTX module is empty"),
            Self::NoAvailableWorkers => write!(f, "no available GPU workers"),
            Self::InsufficientVram { required, available } => {
                write!(f, "insufficient VRAM: need {}MB, have {}MB", required, available)
            }
            Self::KernelNotFound(id) => write!(f, "kernel not found: {}", id),
            Self::InvalidStatus { expected, actual } => {
                write!(f, "invalid status: expected {}, got {}", expected, actual)
            }
            Self::HotswapFailed(reason) => write!(f, "hotswap failed: {}", reason),
        }
    }
}

impl std::error::Error for BridgeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ptx(name: &str) -> PtxModule {
        PtxModule {
            ptx: vec![0x00; 1024],
            kernel_name: name.to_string(),
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
            min_compute_capability: 80,
        }
    }

    #[test]
    fn test_deploy() {
        let mut bridge = CudaclawBridge::new(4, 8192);
        let id = bridge.deploy(make_ptx("attention")).unwrap();
        assert!(matches!(bridge.get_kernel(&id).unwrap().status, DeployStatus::Running { .. }));
        assert_eq!(bridge.available_workers(), 3);
    }

    #[test]
    fn test_deploy_exhausts_workers() {
        let mut bridge = CudaclawBridge::new(2, 8192);
        bridge.deploy(make_ptx("k1")).unwrap();
        bridge.deploy(make_ptx("k2")).unwrap();
        assert!(matches!(bridge.deploy(make_ptx("k3")).unwrap_err(), BridgeError::NoAvailableWorkers));
    }

    #[test]
    fn test_stop() {
        let mut bridge = CudaclawBridge::new(4, 8192);
        let id = bridge.deploy(make_ptx("k1")).unwrap();
        assert_eq!(bridge.available_workers(), 3);
        bridge.stop(&id).unwrap();
        assert_eq!(bridge.available_workers(), 4);
    }

    #[test]
    fn test_hotswap() {
        let mut bridge = CudaclawBridge::new(4, 8192);
        let id = bridge.deploy(make_ptx("v1")).unwrap();
        bridge.hotswap(&id, make_ptx("v2")).unwrap();
        assert_eq!(bridge.get_kernel(&id).unwrap().module.kernel_name, "v2");
    }

    #[test]
    fn test_stats() {
        let mut bridge = CudaclawBridge::new(4, 8192);
        let id = bridge.deploy(make_ptx("k1")).unwrap();
        bridge.update_stats(&id, 100, 5000, 2);
        bridge.update_stats(&id, 50, 2500, 0);
        let kernel = bridge.get_kernel(&id).unwrap();
        assert_eq!(kernel.stats.invocations, 150);
        assert!((kernel.stats.avg_time_us - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_running_kernels() {
        let mut bridge = CudaclawBridge::new(4, 8192);
        bridge.deploy(make_ptx("k1")).unwrap();
        bridge.deploy(make_ptx("k2")).unwrap();
        assert_eq!(bridge.running_kernels().len(), 2);
    }

    #[test]
    fn test_empty_ptx() {
        let mut bridge = CudaclawBridge::new(4, 8192);
        let mut m = make_ptx("empty");
        m.ptx.clear();
        assert!(matches!(bridge.deploy(m).unwrap_err(), BridgeError::EmptyPtx));
    }

    #[test]
    fn test_vram_tracking() {
        let mut bridge = CudaclawBridge::new(4, 100);
        let id = bridge.deploy(make_ptx("k1")).unwrap();
        assert!(bridge.vram_utilization() > 0.0);
        bridge.stop(&id).unwrap();
        assert_eq!(bridge.vram_utilization(), 0.0);
    }
}

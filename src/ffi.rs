//! # OctaSoma Python FFI
//!
//! Exposes [`FractalMemory3D`] as the native Python class `OctaSomaCore` via
//! PyO3.  Internally uses a **Read-Copy-Update (RCU)** pattern backed by
//! [`arc_swap::ArcSwap`], allowing any number of Python threads to issue
//! lock-free `.query()` calls while a single background tokio task
//! serialises writes, clones the tree, applies batched insertions, and
//! atomically swaps the root pointer.

use std::sync::Arc;

use arc_swap::ArcSwap;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tokio::sync::mpsc;

use crate::{FractalMemory3D, NodeId};

// ---------------------------------------------------------------------------
// Write command sent from Python → background worker
// ---------------------------------------------------------------------------

struct WriteOp {
    embedding: Vec<f32>,
    payload: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Background writer task
// ---------------------------------------------------------------------------

/// Infinite async loop that drains the MPSC channel, clones the current
/// tree, applies all pending insertions, and atomically swaps the new tree.
async fn writer_loop(
    mut rx: mpsc::UnboundedReceiver<WriteOp>,
    tree_ref: Arc<ArcSwap<FractalMemory3D>>,
    min_half_size: f32,
) {
    loop {
        let op = match rx.recv().await {
            Some(op) => op,
            None => return, // channel closed — shut down
        };

        // Load current snapshot.
        let current = tree_ref.load_full();
        let mut shadow = (*current).clone();

        // Apply the first operation.
        shadow.insert(&op.embedding, Some(&op.payload), min_half_size);

        // Drain any additional queued operations to amortise clone cost.
        while let Ok(op2) = rx.try_recv() {
            shadow.insert(&op2.embedding, Some(&op2.payload), min_half_size);
        }

        // Atomically publish the new tree.
        tree_ref.store(Arc::new(shadow));
    }
}

// ---------------------------------------------------------------------------
// Python class: OctaSomaCore
// ---------------------------------------------------------------------------

/// Python-facing 3-D fractal memory engine.
///
/// Internally uses RCU: **reads** (`.query()`) are lock-free and never block;
/// **writes** (`.insert()`) are dispatched to a background async worker that
/// batch-processes and atomically swaps the tree.
#[pyclass(name = "OctaSomaCore")]
pub struct OctaSomaCore {
    tree: Arc<ArcSwap<FractalMemory3D>>,
    tx: mpsc::UnboundedSender<WriteOp>,
    /// Held alive to keep the tokio runtime running for the writer task.
    _rt: tokio::runtime::Runtime,
}

impl Drop for OctaSomaCore {
    fn drop(&mut self) {
        // Dropping the sender closes the channel, which causes the writer
        // task to exit gracefully.
    }
}

// Private helpers (not exposed to Python).
impl OctaSomaCore {
    /// Shared RCU engine bootstrap: wraps `mem` in an `ArcSwap`, spawns the
    /// tokio writer task, and returns the initialised `OctaSomaCore`.
    fn spawn_rcu_engine(mem: FractalMemory3D, min_half_size: f32) -> PyResult<Self> {
        let tree = Arc::new(ArcSwap::from(Arc::new(mem)));

        let (tx, rx) = mpsc::unbounded_channel();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;

        let writer_tree = tree.clone();
        rt.spawn(async move {
            writer_loop(rx, writer_tree, min_half_size).await;
        });

        Ok(Self {
            tree,
            tx,
            _rt: rt,
        })
    }
}

#[pymethods]
impl OctaSomaCore {
    /// Constructs a new engine with a deterministically seeded (Johnson–
    /// Lindenstrauss) projection matrix.
    ///
    /// Parameters
    /// ----------
    /// high_dim : int
    ///     Dimensionality of input embeddings.
    /// seed : int
    ///     u64 seed for the Xorshift64 RNG (0 is promoted to a non‑zero constant).
    /// relaxation_factor : float, default 1.05
    ///     Loose-octree relaxation factor.  1.0 = strict octree.
    /// min_half_size : float, default 1e-12
    ///     Stop subdividing when a node's half-size reaches this value.
    #[new]
    #[pyo3(signature = (high_dim, seed, relaxation_factor=1.05, min_half_size=1e-12))]
    fn new(
        high_dim: usize,
        seed: u64,
        relaxation_factor: f32,
        min_half_size: f32,
    ) -> PyResult<Self> {
        let mut mem = FractalMemory3D::new(high_dim, seed);
        mem.relaxation_factor = relaxation_factor;

        Self::spawn_rcu_engine(mem, min_half_size)
    }

    // ------------------------------------------------------------------
    // PCA calibration constructor (classmethod)
    // ------------------------------------------------------------------

    /// Creates an engine from a calibration dataset by running power-iteration
    /// PCA on the Rust side to learn the top three principal components.
    ///
    /// Parameters
    /// ----------
    /// calibration_data : List[List[float]]
    ///     A list of *N* embeddings, each of length *D*.  All rows must have
    ///     the same length.
    /// relaxation_factor : float, default 1.05
    /// min_half_size : float, default 1e-12
    #[staticmethod]
    #[pyo3(signature = (calibration_data, relaxation_factor=1.05, min_half_size=1e-12))]
    fn new_with_pca(
        calibration_data: Vec<Vec<f32>>,
        relaxation_factor: f32,
        min_half_size: f32,
    ) -> PyResult<Self> {
        if calibration_data.is_empty() {
            return Err(PyValueError::new_err("calibration_data must not be empty"));
        }

        let high_dim = calibration_data[0].len();
        for (i, row) in calibration_data.iter().enumerate() {
            if row.len() != high_dim {
                return Err(PyValueError::new_err(format!(
                    "row 0 has length {high_dim}, row {i} has length {}",
                    row.len()
                )));
            }
        }

        let num_samples = calibration_data.len();
        let flat: Vec<f32> = calibration_data.into_iter().flatten().collect();

        let mut mem = FractalMemory3D::new_with_pca(high_dim, &flat, num_samples);
        mem.relaxation_factor = relaxation_factor;

        Self::spawn_rcu_engine(mem, min_half_size)
    }

    // ------------------------------------------------------------------
    // Insert (async write via RCU)
    // ------------------------------------------------------------------

    /// Queues an embedding + payload for insertion via the background writer.
    ///
    /// This method returns immediately — the actual octree update happens
    /// asynchronously.  A small delay may exist before the new memory becomes
    /// visible to `.query()`.
    ///
    /// Parameters
    /// ----------
    /// embedding : List[float]
    ///     High-dimensional embedding vector.
    /// payload : bytes
    ///     Raw byte payload to associate with this memory.
    fn insert(&self, embedding: Vec<f32>, payload: Vec<u8>) -> PyResult<()> {
        self.tx
            .send(WriteOp { embedding, payload })
            .map_err(|e| PyRuntimeError::new_err(format!("writer channel closed: {e}")))?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Query (lock-free read)
    // ------------------------------------------------------------------

    /// Lock-free query: projects the embedding to 3-D, traverses the loose
    /// octree, and returns the payload bytes stored at the deepest matching
    /// node (or `None` if the tree is empty or no payload exists).
    ///
    /// Parameters
    /// ----------
    /// embedding : List[float]
    ///     High-dimensional query vector.
    ///
    /// Returns
    /// -------
    /// Optional[bytes]
    fn query<'py>(&self, py: Python<'py>, embedding: Vec<f32>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let snapshot = self.tree.load();
        let node_id: NodeId = match snapshot.locate(&embedding) {
            Some(id) => id,
            None => return Ok(None),
        };

        match snapshot.get_payload(node_id) {
            Some(data) => Ok(Some(PyBytes::new(py, data))),
            None => Ok(None),
        }
    }

    /// Returns `(node_count, arena_bytes)` for monitoring.
    fn stats(&self) -> (usize, usize) {
        let snapshot = self.tree.load();
        (snapshot.node_count(), snapshot.payload_arena.len())
    }

    /// Returns the number of nodes currently in the tree.
    #[getter]
    fn node_count(&self) -> usize {
        let snapshot = self.tree.load();
        snapshot.node_count()
    }

    /// Returns the total size of the payload arena in bytes.
    #[getter]
    fn arena_size(&self) -> usize {
        let snapshot = self.tree.load();
        snapshot.payload_arena.len()
    }

    // ------------------------------------------------------------------
    // Persistence
    // ------------------------------------------------------------------

    /// Saves the current tree snapshot to disk as a `.frac` file.
    ///
    /// The payload arena is LZ4-compressed on write.
    fn save(&self, path: &str) -> PyResult<()> {
        let snapshot = self.tree.load();
        snapshot
            .save_to_disk(path)
            .map_err(|e| PyIOError::new_err(format!("save failed: {e}")))
    }

    /// Loads an engine from a `.frac` file, replacing the current tree.
    ///
    /// The `high_dim` must match the value stored in the file header.
    #[staticmethod]
    #[pyo3(signature = (path, high_dim, relaxation_factor=1.05, min_half_size=1e-12))]
    fn load(
        path: &str,
        high_dim: usize,
        relaxation_factor: f32,
        min_half_size: f32,
    ) -> PyResult<Self> {
        let mut mem = FractalMemory3D::load_from_disk(path, high_dim)
            .map_err(|e| PyIOError::new_err(format!("load failed: {e}")))?;
        mem.relaxation_factor = relaxation_factor;

        Self::spawn_rcu_engine(mem, min_half_size)
    }
}

// ---------------------------------------------------------------------------
// Top-level Python module entry-point
// ---------------------------------------------------------------------------

/// The `octasoma` Python module.
///
/// Usage: `from octasoma import OctaSomaCore`
#[pymodule]
fn octasoma(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OctaSomaCore>()?;
    Ok(())
}

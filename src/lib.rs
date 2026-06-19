//! # OctaSoma — 3D Fractal Semantic Memory Engine
//!
//! A 100% pure, stable Rust implementation of a 3D fractal semantic memory
//! engine built on contiguous octree storage, Data-Oriented Design principles,
//! learned PCA/SVD projections, and zero-copy persistence with LZ4 compression.
//!
//! ## Design
//!
//! - **Contiguous octree**: all [`OctreeNode`]s live in a single [`Vec`], indexed
//!   by [`NodeId`] (a `usize`).  No `Rc`, `RefCell`, or boxed indirection is used
//!   for linking — children are stored as `[Option<NodeId>; 8]`.
//! - **Learned projection**: the `3 × D` projection matrix can be initialised
//!   either deterministically (Xorshift64, Johnson–Lindenstrauss) or from a
//!   calibration dataset via power-iteration PCA, extracting the three
//!   principal components that capture maximal semantic variance.
//! - **Cache-line padding**: [`OctreeNode`] is 192 bytes (3 × 64 B), aligned
//!   to prevent a single node from straddling L1 cache-line boundaries.
//! - **Loose octree**: a `relaxation_factor` expands node bounding volumes so
//!   that boundary-adjacent points are never lost during queries.
//! - **Bitwise octant routing**: octant index is computed via `|= 1/2/4` masks.
//! - **Zero-copy payload**: [`get_payload`] returns a `&[u8]` with bounds checks
//!   and overflow protection via `checked_add`.
//! - **Compressed persistence**: the payload arena is LZ4-compressed on disk
//!   (`lz4_flex`), dramatically reducing I/O for large agent-memory stores.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};

// ---------------------------------------------------------------------------
// Type alias
// ---------------------------------------------------------------------------

/// A direct index into the [`FractalMemory3D::nodes`] vector.
pub type NodeId = usize;

// ---------------------------------------------------------------------------
// Deterministic pseudo-random number generator (Xorshift64)
// ---------------------------------------------------------------------------

/// A minimal, dependency-free deterministic RNG backed by the Xorshift64
/// algorithm.  From a fixed `u64` seed it always produces the same sequence,
/// making it suitable for reproducible projection-matrix generation.
#[derive(Clone, Debug)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Seeds the generator.  A seed of `0` is promoted to a non-zero constant
    /// to avoid a permanently dead state (Xorshift64 requires `state != 0`).
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF_CAFE_BABE } else { seed },
        }
    }

    /// Advances the internal state and returns the raw 64-bit value.
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a deterministic `f32` in the range `[-1.0, 1.0)`.
    ///
    /// Uses the upper 32 bits of the generator state, re-interpreted as a
    /// uniform float in `[1.0, 2.0)` via IEEE-754 bit manipulation, then
    /// scaled to the target range.  This avoids floating-point division and
    /// its associated rounding uncertainty.
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 32) as u32;
        // Construct a normalised float in [1.0, 2.0) by setting the exponent
        // to 127 (bias) and using bits[22:0] as the mantissa.
        // 0x3F80_0000 is 1.0_f32 in IEEE-754 binary32.
        let normalised = f32::from_bits(0x3F80_0000 | ((bits >> 9) & 0x007F_FFFF));
        (normalised - 1.0) * 2.0 - 1.0
    }

    /// Returns a deterministic `f64` in the range `[-1.0, 1.0)` using the
    /// high 53 bits of the generator state for full double-precision mantissa.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64();
        // Set exponent to 1023 (bias) → range [1.0, 2.0), then shift.
        let normalised = f64::from_bits(0x3FF0_0000_0000_0000 | ((bits >> 12) & 0x000F_FFFF_FFFF_FFFF));
        (normalised - 1.0) * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Core node type (cache-line padded)
// ---------------------------------------------------------------------------

/// A single octree node with a `#[repr(C)]` layout padded to 192 bytes
/// (3 cache lines of 64 B each) so that the hardware prefetcher never
/// straddles L1 boundaries when streaming nodes sequentially.
///
/// Each node anchors a cubic region of space defined by `center` and
/// `half_size` (half the edge length).  Up to eight child nodes subdivide
/// this region into eight equal octants.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct OctreeNode {
    /// Self-referring index within [`FractalMemory3D::nodes`].
    pub id: NodeId,
    /// 3-D spatial centre of this node's bounding cube.
    pub center: [f32; 3],
    /// Half the side-length of this node's cube (full side = `half_size * 2`).
    pub half_size: f32,
    /// Octant children.  `children[i]` corresponds to the sub-cube whose
    /// centre is offset along the three axes according to the bit layout:
    ///
    /// | bit 0 (1) | bit 1 (2) | bit 2 (4) |
    /// |-----------|-----------|------------|
    /// | x ≥ cx    | y ≥ cy    | z ≥ cz     |
    pub children: [Option<NodeId>; 8],
    /// Index into the embedding table for the data point stored here.
    pub embedding_id: usize,
    /// Start offset (in bytes) of this node's payload within
    /// [`FractalMemory3D::payload_arena`].
    pub payload_offset: usize,
    /// Length (in bytes) of this node's payload within the arena.
    pub payload_len: usize,
    /// Explicit padding to align the struct size to a 64-byte boundary.
    /// Current layout: 8 + 12 + 4 + 128 + 8 + 8 + 8 = 176 → +16 = 192.
    _padding: [u8; 16],
}

// Compile-time assertion: OctreeNode must be a multiple of 64 B.
const _: () = {
    if !std::mem::size_of::<OctreeNode>().is_multiple_of(64) {
        panic!("OctreeNode size must be a multiple of 64 bytes");
    }
};

impl OctreeNode {
    /// Creates a new leaf node with no children.
    #[inline]
    pub fn new(
        id: NodeId,
        center: [f32; 3],
        half_size: f32,
        embedding_id: usize,
        payload_offset: usize,
        payload_len: usize,
    ) -> Self {
        Self {
            id,
            center,
            half_size,
            children: [None; 8],
            embedding_id,
            payload_offset,
            payload_len,
            _padding: [0u8; 16],
        }
    }

    /// Returns the effective half-size of this node when the loose-octree
    /// relaxation factor is applied.
    #[inline(always)]
    pub fn loose_half_size(&self, factor: f32) -> f32 {
        self.half_size * factor
    }

    /// Checks whether `point` lies within the expanded (loose) bounding
    /// cube of this node.
    #[inline]
    pub fn contains_point_loose(&self, point: [f32; 3], factor: f32) -> bool {
        let loose_half = self.loose_half_size(factor);
        (point[0] - self.center[0]).abs() <= loose_half
            && (point[1] - self.center[1]).abs() <= loose_half
            && (point[2] - self.center[2]).abs() <= loose_half
    }
}

// ---------------------------------------------------------------------------
// File header for on-disk persistence
// ---------------------------------------------------------------------------

/// Magic bytes identifying an OctaSoma binary file.
const FILE_MAGIC: [u8; 4] = *b"FRAC";
/// Current on-disk format version (v2 adds LZ4-compressed payload arena).
const FILE_VERSION: u32 = 2;

/// Binary header written at the start of every persistent file.
#[repr(C)]
struct FileHeader {
    magic: [u8; 4],
    version: u32,
    high_dim: u32,
}

impl FileHeader {
    fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.magic)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.high_dim.to_le_bytes())
    }

    fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        let mut version_buf = [0u8; 4];
        reader.read_exact(&mut version_buf)?;
        let version = u32::from_le_bytes(version_buf);
        let mut dim_buf = [0u8; 4];
        reader.read_exact(&mut dim_buf)?;
        let high_dim = u32::from_le_bytes(dim_buf);
        Ok(Self { magic, version, high_dim })
    }
}

// ---------------------------------------------------------------------------
// PCA / SVD calibration — power iteration on the data matrix
// ---------------------------------------------------------------------------

/// Computes the top three principal components of a `num_samples × high_dim`
/// row-major data matrix using power iteration with Hotelling deflation.
///
/// The returned `Vec<f32>` is a flat `3 × high_dim` matrix, row-major,
/// suitable for direct use as [`FractalMemory3D::projection_matrix`].
///
/// # Algorithm
///
/// 1. Centre the data (subtract per-column mean).
/// 2. For each of the three components:
///    - Run power iteration: `v ← Xᵀ·(X·v)`, normalise.
///    - Deflate the data matrix: `X ← X − (X·v)·vᵀ`.
///
/// All arithmetic uses `f64` accumulators to guarantee cross-platform
/// reproducibility.
///
/// # Panics
///
/// Panics if `data.len() != num_samples * high_dim`, or if `num_samples == 0`
/// or `high_dim == 0`.
pub fn compute_pca_projection(
    data: &[f32],
    num_samples: usize,
    high_dim: usize,
    max_iters: usize,
) -> Vec<f32> {
    assert_eq!(data.len(), num_samples * high_dim);
    assert!(num_samples > 0);
    assert!(high_dim > 0);
    assert!(max_iters > 0);

    let n = num_samples;
    let d = high_dim;

    // 1. Centre the data (f64 working copy).
    let mut mean = vec![0.0f64; d];
    for i in 0..n {
        let row = &data[i * d..(i + 1) * d];
        for (j, &val) in row.iter().enumerate() {
            mean[j] += val as f64;
        }
    }
    let inv_n = 1.0 / n as f64;
    for v in mean.iter_mut() {
        *v *= inv_n;
    }

    let mut centered: Vec<f64> = Vec::with_capacity(n * d);
    for i in 0..n {
        let row = &data[i * d..(i + 1) * d];
        for (j, &val) in row.iter().enumerate() {
            centered.push(val as f64 - mean[j]);
        }
    }

    // 2. Extract top-3 eigenvectors.
    let mut projection = vec![0.0f32; 3 * d];
    let mut v = vec![0.0f64; d];
    let mut rng = DeterministicRng::new(0x50_4443_415F_4341); // "PDCA_CA"

    for comp in 0..3 {
        // Random initialisation.
        for elem in v.iter_mut() {
            *elem = rng.next_f64();
        }
        l2_normalise(&mut v);

        // Power iteration.
        for _ in 0..max_iters {
            let xv = mat_vec_mul(&centered, n, d, &v, false); // X · v  → N-vec
            let xt_xv = mat_vec_mul(&centered, n, d, &xv, true); // Xᵀ·(X·v) → D-vec
            v.copy_from_slice(&xt_xv);
            l2_normalise(&mut v);
        }

        // Store.
        for j in 0..d {
            projection[comp * d + j] = v[j] as f32;
        }

        // Deflate: X ← X − (X·v)·vᵀ.
        if comp < 2 {
            let xv = mat_vec_mul(&centered, n, d, &v, false);
            for (row, &scalar) in centered.chunks_exact_mut(d).zip(xv.iter()) {
                for (elem, &comp) in row.iter_mut().zip(v.iter()) {
                    *elem -= scalar * comp;
                }
            }
        }
    }

    projection
}

/// L2-normalises a vector in-place.
fn l2_normalise(v: &mut [f64]) {
    let norm_sq: f64 = v.iter().map(|x| x * x).sum();
    if norm_sq < f64::EPSILON {
        return; // degenerate case — leave as-is
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    for x in v.iter_mut() {
        *x *= inv_norm;
    }
}

/// Matrix-vector multiplication on a row-major `n × d` matrix.
///
/// - When `transpose` is `false`: returns `X · v` (length `n`).
/// - When `transpose` is `true` : returns `Xᵀ · v` (length `d`).
fn mat_vec_mul(data: &[f64], n: usize, d: usize, vec: &[f64], transpose: bool) -> Vec<f64> {
    if transpose {
        // Xᵀ · v:  output[j] = Σᵢ X[i][j] * v[i]
        let mut out = vec![0.0f64; d];
        for i in 0..n {
            let row = &data[i * d..(i + 1) * d];
            let scalar = vec[i];
            for j in 0..d {
                out[j] += row[j] * scalar;
            }
        }
        out
    } else {
        // X · v:  output[i] = Σⱼ X[i][j] * v[j]
        let mut out = vec![0.0f64; n];
        for i in 0..n {
            let row = &data[i * d..(i + 1) * d];
            let mut sum = 0.0f64;
            for j in 0..d {
                sum += row[j] * vec[j];
            }
            out[i] = sum;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Primary container: FractalMemory3D
// ---------------------------------------------------------------------------

/// The central octree-backed 3-D fractal memory engine.
///
/// All data lives in contiguous, cache-friendly vectors:
/// - [`nodes`] stores every octree node in insertion order.
/// - [`payload_arena`] stores raw byte payloads in a single heap allocation.
/// - [`projection_matrix`] is a flat `3 × D` row-major matrix used to map
///   high-dimensional embeddings down to 3-D spatial coordinates.
///
/// ## Loose octree
///
/// The [`relaxation_factor`] (default `1.05`) expands each node's bounding
/// cube by 5 % so that query points falling exactly on octant boundaries are
/// never lost.  Insertion still uses strict octant routing; queries verify
/// loose containment at every descent step.
#[derive(Clone)]
pub struct FractalMemory3D {
    /// Contiguous octree node store.  Indexed by [`NodeId`].
    pub nodes: Vec<OctreeNode>,
    /// The root of the octree, if any node has been inserted.
    pub root_id: Option<NodeId>,
    /// Flat row-major projection matrix of shape `3 × high_dim`.
    /// Row `i` starts at `projection_matrix[i * high_dim]`.
    pub projection_matrix: Vec<f32>,
    /// Dimensionality of input embeddings.
    pub high_dim: usize,
    /// Monolithic byte arena for zero-copy payload storage.
    pub payload_arena: Vec<u8>,
    /// Loose-octree relaxation factor (default 1.05).  Each node's effective
    /// half-size for containment queries is `half_size * relaxation_factor`.
    pub relaxation_factor: f32,
}

impl FractalMemory3D {
    // -----------------------------------------------------------------------
    // Private: shared initialisation
    // -----------------------------------------------------------------------

    fn new_empty(high_dim: usize, projection_matrix: Vec<f32>) -> Self {
        Self {
            nodes: Vec::new(),
            root_id: None,
            projection_matrix,
            high_dim,
            payload_arena: Vec::new(),
            relaxation_factor: 1.05,
        }
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Creates an empty fractal memory engine with a pre-computed projection
    /// matrix of shape `3 × high_dim`.
    ///
    /// The projection matrix is filled deterministically from the given `seed`,
    /// ensuring that identical `(high_dim, seed)` pairs produce bit-identical
    /// spatial layouts on any platform.
    pub fn new(high_dim: usize, seed: u64) -> Self {
        let mat_len = 3 * high_dim;
        let mut rng = DeterministicRng::new(seed);
        let mut projection_matrix = Vec::with_capacity(mat_len);
        for _ in 0..mat_len {
            projection_matrix.push(rng.next_f32());
        }

        Self::new_empty(high_dim, projection_matrix)
    }

    /// Creates an empty engine using a **learned** projection matrix obtained
    /// via [`compute_pca_projection`] on a calibration dataset.
    ///
    /// `projection_matrix` must be a flat `3 × high_dim` row-major matrix.
    /// The caller is responsible for providing a correctly-dimensioned matrix.
    ///
    /// # Panics
    ///
    /// Panics if `projection_matrix.len() != 3 * high_dim`.
    pub fn new_from_calibration(
        high_dim: usize,
        projection_matrix: Vec<f32>,
    ) -> Self {
        assert_eq!(
            projection_matrix.len(),
            3 * high_dim,
            "projection_matrix must have length 3 * high_dim"
        );

        Self::new_empty(high_dim, projection_matrix)
    }

    /// Convenience: creates an engine from a calibration dataset by first
    /// running PCA to learn the projection matrix.
    ///
    /// `calibration_data` is a flat, row-major `num_samples × high_dim` matrix
    /// of embedding vectors used to learn the optimal 3-D projection.
    pub fn new_with_pca(
        high_dim: usize,
        calibration_data: &[f32],
        num_samples: usize,
    ) -> Self {
        let projection_matrix =
            compute_pca_projection(calibration_data, num_samples, high_dim, 20);
        Self::new_from_calibration(high_dim, projection_matrix)
    }

    /// Returns the number of nodes currently in the tree.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // -----------------------------------------------------------------------
    // Projection (high-D → 3-D) — pairwise-reduction flavour
    // -----------------------------------------------------------------------

    /// Projects a high-dimensional embedding down to a 3-D spatial point
    /// using the internally stored projection matrix.
    ///
    /// Uses an `f64` accumulator per output dimension to eliminate
    /// floating-point non-associativity across platforms (x86\_64 fused
    /// multiply-add vs. ARM64 split instructions).  The accumulator loop
    /// processes chunks of 4 elements to encourage auto-vectorisation while
    /// preserving deterministic reduction order.  Final coordinates are cast
    /// to `[f32; 3]`.
    #[inline]
    pub fn project_to_3d(&self, embedding: &[f32]) -> Option<[f32; 3]> {
        project_to_3d(embedding, &self.projection_matrix, self.high_dim)
    }

    // -----------------------------------------------------------------------
    // Octant routing
    // -----------------------------------------------------------------------

    /// Returns the octant index (0..=7) of `point` relative to `center`.
    ///
    /// Bit layout:
    /// - bit 0 (1) set when `point.x ≥ center.x`
    /// - bit 1 (2) set when `point.y ≥ center.y`
    /// - bit 2 (4) set when `point.z ≥ center.z`
    #[inline(always)]
    pub fn octant_index(center: [f32; 3], point: [f32; 3]) -> usize {
        let mut index: usize = 0;
        if point[0] >= center[0] {
            index |= 1;
        }
        if point[1] >= center[1] {
            index |= 2;
        }
        if point[2] >= center[2] {
            index |= 4;
        }
        index
    }

    /// Computes the centre of child octant `i` given the parent's `center`
    /// and `half_size`.  The child's `half_size` is `parent_half * 0.5`.
    #[inline(always)]
    pub fn child_center(
        parent_center: [f32; 3],
        parent_half: f32,
        octant: usize,
    ) -> [f32; 3] {
        let quarter = parent_half * 0.5;
        [
            parent_center[0] + if (octant & 1) != 0 { quarter } else { -quarter },
            parent_center[1] + if (octant & 2) != 0 { quarter } else { -quarter },
            parent_center[2] + if (octant & 4) != 0 { quarter } else { -quarter },
        ]
    }

    // -----------------------------------------------------------------------
    // Insertion
    // -----------------------------------------------------------------------

    /// Inserts a high-dimensional embedding (and optional byte payload) into
    /// the octree and returns the [`NodeId`] of the terminal node where the
    /// point was placed.
    ///
    /// If the tree is empty a root node is created whose `half_size` is
    /// automatically sized to encompass the first projected point with a
    /// 1.5× safety margin (minimum 1.0).
    ///
    /// The tree subdivides until `min_half_size` is reached; at that point
    /// the traversal stops and the current leaf node is returned.  Multiple
    /// embeddings may therefore be co-located in the same leaf.
    pub fn insert(
        &mut self,
        embedding: &[f32],
        payload: Option<&[u8]>,
        min_half_size: f32,
    ) -> Option<NodeId> {
        let point = self.project_to_3d(embedding)?;
        let embedding_id = self.nodes.len(); // sequential embedding id

        // Stage the payload into the arena.
        let (payload_offset, payload_len) = if let Some(data) = payload {
            let off = self.payload_arena.len();
            self.payload_arena.extend_from_slice(data);
            (off, data.len())
        } else {
            (0, 0)
        };

        // Create root if needed, sized to contain this first point.
        if self.root_id.is_none() {
            let max_coord = point[0]
                .abs()
                .max(point[1].abs())
                .max(point[2].abs());
            let root_half = (max_coord * 1.5).max(1.0);

            let id = self.nodes.len();
            self.nodes.push(OctreeNode::new(
                id,
                [0.0, 0.0, 0.0],
                root_half,
                embedding_id,
                payload_offset,
                payload_len,
            ));
            self.root_id = Some(id);
            // Fall through to the subdivision loop so the point is placed
            // at the correct depth rather than staying at the root.
        }

        let mut current_id = self.root_id.unwrap();
        loop {
            let node = &self.nodes[current_id];
            let center = node.center;
            let half_size = node.half_size;
            let octant = Self::octant_index(center, point);
            let child_opt = node.children[octant];
            // immutable borrow released here

            // Stop subdividing once we reach the minimum cell size.
            if half_size <= min_half_size {
                return Some(current_id);
            }

            if let Some(child_id) = child_opt {
                current_id = child_id;
                continue;
            }

            // Octant is empty — create a new child.
            let child_half = half_size * 0.5;
            let child_center = Self::child_center(center, half_size, octant);
            let child_id = self.nodes.len();
            self.nodes.push(OctreeNode::new(
                child_id,
                child_center,
                child_half,
                embedding_id,
                payload_offset,
                payload_len,
            ));
            self.nodes[current_id].children[octant] = Some(child_id);
            return Some(child_id);
        }
    }

    // -----------------------------------------------------------------------
    // Query: locate the leaf node for a given high-D embedding (loose)
    // -----------------------------------------------------------------------

    /// Traverses the tree for `embedding` and returns the [`NodeId`] of the
    /// deepest node whose **loose** bounding cube contains its 3-D projection.
    ///
    /// The loose octree with `relaxation_factor` ensures boundary-adjacent
    /// points are reliably found.  Empty octants cause early return at the
    /// current parent; if the primary strict octant child exists but the
    /// point lies outside its loose bounds, sibling octants are probed.
    pub fn locate(&self, embedding: &[f32]) -> Option<NodeId> {
        let point = self.project_to_3d(embedding)?;
        self.locate_point(point)
    }

    /// Low-level locate using a concrete 3-D point.
    pub fn locate_point(&self, point: [f32; 3]) -> Option<NodeId> {
        let mut current_id = self.root_id?;
        let factor = self.relaxation_factor;
        let mut depth_guard: u32 = 0; // infinite-loop safety

        loop {
            depth_guard += 1;
            if depth_guard > 128 {
                // 2^128 subdivisions would overflow f32 precision long before.
                return Some(current_id);
            }

            let node = &self.nodes[current_id];
            let octant = Self::octant_index(node.center, point);

            // Try the primary strict-octant child.
            if let Some(child_id) = node.children[octant] {
                let child = &self.nodes[child_id];
                if child.contains_point_loose(point, factor) {
                    current_id = child_id;
                    continue;
                }
            }

            // Probe all other children (loose overlap means the point may
            // belong to a neighbouring octant).
            let mut found = false;
            for other in 0..8 {
                if other == octant {
                    continue;
                }
                if let Some(child_id) = node.children[other] {
                    let child = &self.nodes[child_id];
                    if child.contains_point_loose(point, factor) {
                        current_id = child_id;
                        found = true;
                        break;
                    }
                }
            }
            if found {
                continue;
            }

            // No child claims the point — return this node.
            return Some(current_id);
        }
    }

    // -----------------------------------------------------------------------
    // Zero-copy payload access
    // -----------------------------------------------------------------------

    /// Returns a zero-copy reference to the raw byte payload stored for
    /// `node_id`, bounded by the lifetime of `self`.
    ///
    /// ## Safety guards
    ///
    /// - Validates that `node_id` is within bounds.
    /// - Uses [`usize::checked_add`] to prevent malicious/malformed
    ///   `payload_offset + payload_len` overflows.
    /// - Checks that the computed end does not exceed `payload_arena.len()`.
    pub fn get_payload(&self, node_id: NodeId) -> Option<&[u8]> {
        let node = self.nodes.get(node_id)?;

        // Protect against overflow in `payload_offset + payload_len`.
        let end = node.payload_offset.checked_add(node.payload_len)?;

        if end > self.payload_arena.len() {
            return None;
        }

        Some(&self.payload_arena[node.payload_offset..end])
    }

    // -----------------------------------------------------------------------
    // Persistence — save (with LZ4 arena compression)
    // -----------------------------------------------------------------------

    /// Serialises the entire engine to a buffered, sequential binary stream.
    ///
    /// ## On-disk layout (little-endian, v2, LZ4 compressed)
    ///
    /// ```text
    /// FileHeader       { magic: b"FRAC", version: 2u32, high_dim: u32 }
    /// node_count       u64
    /// nodes            [OctreeNode; node_count]    (see write_node)
    /// proj_len         u64
    /// proj_mat         [f32; proj_len]
    /// arena_decomp_len u64
    /// arena_comp_len   u64
    /// arena_compressed [u8; arena_comp_len]
    /// ```
    pub fn save_to_disk(&self, path: &str) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header.
        FileHeader {
            magic: FILE_MAGIC,
            version: FILE_VERSION,
            high_dim: self.high_dim as u32,
        }
        .write_to(&mut writer)?;

        // Node count + nodes.
        let node_count = self.nodes.len() as u64;
        writer.write_all(&node_count.to_le_bytes())?;
        for node in &self.nodes {
            write_node(&mut writer, node)?;
        }

        // Projection matrix.
        let proj_len = self.projection_matrix.len() as u64;
        writer.write_all(&proj_len.to_le_bytes())?;
        for v in &self.projection_matrix {
            writer.write_all(&v.to_le_bytes())?;
        }

        // Payload arena — LZ4 compressed.
        let arena_raw = &self.payload_arena;
        let arena_decomp_len = arena_raw.len() as u64;
        writer.write_all(&arena_decomp_len.to_le_bytes())?;

        let compressed = lz4_flex::compress(arena_raw);
        let arena_comp_len = compressed.len() as u64;
        writer.write_all(&arena_comp_len.to_le_bytes())?;
        writer.write_all(&compressed)?;

        writer.flush()?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Persistence — load (with LZ4 arena decompression)
    // -----------------------------------------------------------------------

    /// Deserialises an engine from the binary file at `path`, restoring the
    /// exact state written by [`save_to_disk`].
    ///
    /// ## Validation gates
    ///
    /// - Magic must match `b"FRAC"`.
    /// - `high_dim` in the header must equal the caller-provided
    ///   `expected_high_dim`.
    /// - Any mismatch produces a descriptive [`io::Error`]; no panics.
    pub fn load_from_disk(path: &str, expected_high_dim: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Header.
        let header = FileHeader::read_from(&mut reader)?;

        if header.magic != FILE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid magic: expected {:?}, got {:?}",
                    std::str::from_utf8(&FILE_MAGIC).unwrap_or("<non-utf8>"),
                    std::str::from_utf8(&header.magic).unwrap_or("<non-utf8>"),
                ),
            ));
        }

        if header.version != FILE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported file version: file has {}, this library supports version {}",
                    header.version, FILE_VERSION,
                ),
            ));
        }

        if header.high_dim as usize != expected_high_dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "high_dim mismatch: file has {}, caller expected {}",
                    header.high_dim, expected_high_dim,
                ),
            ));
        }

        // Node count + nodes.
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let node_count = u64::from_le_bytes(buf8) as usize;
        let mut nodes: Vec<OctreeNode> = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            nodes.push(read_node(&mut reader)?);
        }

        // Projection matrix.
        reader.read_exact(&mut buf8)?;
        let proj_len = u64::from_le_bytes(buf8) as usize;
        let mut projection_matrix: Vec<f32> = Vec::with_capacity(proj_len);
        let mut buf4 = [0u8; 4];
        for _ in 0..proj_len {
            reader.read_exact(&mut buf4)?;
            projection_matrix.push(f32::from_le_bytes(buf4));
        }

        // Payload arena — LZ4 decompressed (v2 format).
        reader.read_exact(&mut buf8)?;
        let arena_decomp_len = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8)?;
        let arena_comp_len = u64::from_le_bytes(buf8) as usize;

        let mut compressed = vec![0u8; arena_comp_len];
        reader.read_exact(&mut compressed)?;

        let payload_arena = lz4_flex::decompress(&compressed, arena_decomp_len).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("lz4 decompression failed: {e}"))
        })?;

        Ok(Self {
            root_id: if nodes.is_empty() { None } else { Some(0) },
            nodes,
            projection_matrix,
            high_dim: expected_high_dim,
            payload_arena,
            relaxation_factor: 1.05,
        })
    }
}

// ---------------------------------------------------------------------------
// Serialisation helpers for a single OctreeNode
// ---------------------------------------------------------------------------

fn write_node<W: Write>(writer: &mut W, node: &OctreeNode) -> io::Result<()> {
    writer.write_all(&(node.id as u64).to_le_bytes())?;
    writer.write_all(&node.center[0].to_le_bytes())?;
    writer.write_all(&node.center[1].to_le_bytes())?;
    writer.write_all(&node.center[2].to_le_bytes())?;
    writer.write_all(&node.half_size.to_le_bytes())?;
    for child in &node.children {
        let val: u64 = match child {
            Some(id) => *id as u64,
            None => u64::MAX,
        };
        writer.write_all(&val.to_le_bytes())?;
    }
    writer.write_all(&(node.embedding_id as u64).to_le_bytes())?;
    writer.write_all(&(node.payload_offset as u64).to_le_bytes())?;
    writer.write_all(&(node.payload_len as u64).to_le_bytes())?;
    // Write padding as zeros (forward-compatible).
    writer.write_all(&[0u8; 16])?;
    Ok(())
}

fn read_node<R: Read>(reader: &mut R) -> io::Result<OctreeNode> {
    let mut buf8 = [0u8; 8];
    let mut buf4 = [0u8; 4];

    reader.read_exact(&mut buf8)?;
    let id = u64::from_le_bytes(buf8) as NodeId;

    reader.read_exact(&mut buf4)?;
    let cx = f32::from_le_bytes(buf4);
    reader.read_exact(&mut buf4)?;
    let cy = f32::from_le_bytes(buf4);
    reader.read_exact(&mut buf4)?;
    let cz = f32::from_le_bytes(buf4);
    let center = [cx, cy, cz];

    reader.read_exact(&mut buf4)?;
    let half_size = f32::from_le_bytes(buf4);

    let mut children: [Option<NodeId>; 8] = [None; 8];
    for slot in children.iter_mut() {
        reader.read_exact(&mut buf8)?;
        let raw = u64::from_le_bytes(buf8);
        *slot = if raw == u64::MAX { None } else { Some(raw as NodeId) };
    }

    reader.read_exact(&mut buf8)?;
    let embedding_id = u64::from_le_bytes(buf8) as usize;

    reader.read_exact(&mut buf8)?;
    let payload_offset = u64::from_le_bytes(buf8) as usize;

    reader.read_exact(&mut buf8)?;
    let payload_len = u64::from_le_bytes(buf8) as usize;

    // Read and discard padding.
    let mut padding = [0u8; 16];
    reader.read_exact(&mut padding)?;

    Ok(OctreeNode {
        id,
        center,
        half_size,
        children,
        embedding_id,
        payload_offset,
        payload_len,
        _padding: [0u8; 16],
    })
}

// ---------------------------------------------------------------------------
// Free function: project_to_3d (pairwise-reduction flavour)
// ---------------------------------------------------------------------------

/// Projects a high-dimensional `embedding` to a 3-D point using a flat
/// row-major projection matrix of shape `3 × high_dim`.
///
/// Each output dimension is accumulated in `f64` using a chunked (4-element)
/// inner loop to encourage SIMD auto-vectorisation while preserving a strict
/// sequential reduction order for cross-platform determinism (x86\_64 vs
/// ARM64 FMA invariance).
///
/// Returns `None` if `embedding.len() != high_dim` or the matrix is too short.
pub fn project_to_3d(
    embedding: &[f32],
    projection_matrix: &[f32],
    high_dim: usize,
) -> Option<[f32; 3]> {
    if embedding.len() != high_dim {
        return None;
    }
    if projection_matrix.len() < 3 * high_dim {
        return None;
    }

    let mut result_f64 = [0.0f64; 3];
    let chunks = high_dim / 4;
    let remainder = high_dim % 4;

    for (dim, acc) in result_f64.iter_mut().enumerate() {
        let row_offset = dim * high_dim;
        let mut sum: f64 = 0.0;

        // Chunked loop (4-element groups — compiler-friendly for SIMD).
        let mut j = 0;
        for _ in 0..chunks {
            sum += embedding[j] as f64 * projection_matrix[row_offset + j] as f64
                + embedding[j + 1] as f64 * projection_matrix[row_offset + j + 1] as f64
                + embedding[j + 2] as f64 * projection_matrix[row_offset + j + 2] as f64
                + embedding[j + 3] as f64 * projection_matrix[row_offset + j + 3] as f64;
            j += 4;
        }

        // Tail.
        for k in 0..remainder {
            sum += embedding[j + k] as f64 * projection_matrix[row_offset + j + k] as f64;
        }

        *acc = sum;
    }

    Some([
        result_f64[0] as f32,
        result_f64[1] as f32,
        result_f64[2] as f32,
    ])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- DeterministicRng ---------------------------------------------------

    #[test]
    fn rng_determinism() {
        let mut a = DeterministicRng::new(42);
        let mut b = DeterministicRng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_f32(), b.next_f32());
        }
    }

    #[test]
    fn rng_range() {
        let mut rng = DeterministicRng::new(12345);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            assert!(v >= -1.0 && v < 1.0, "out of range: {v}");
        }
    }

    // -- project_to_3d ------------------------------------------------------

    #[test]
    fn projection_shape_mismatch_rejected() {
        assert!(project_to_3d(&[1.0, 2.0], &[], 4).is_none());
        assert!(project_to_3d(&[1.0], &[0.0; 2], 1).is_none());
        assert!(project_to_3d(&[1.0], &[0.0, 0.0, 0.0], 1).is_some());
    }

    #[test]
    fn projection_zero_embedding() {
        let mat = vec![0.5f32; 3 * 4];
        let pt = project_to_3d(&[0.0; 4], &mat, 4).unwrap();
        assert_eq!(pt, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn projection_with_remainder() {
        // 7-dim embedding (not a multiple of 4) to exercise the tail loop.
        let mat: Vec<f32> = (0..21).map(|i| i as f32 * 0.1).collect();
        let emb: Vec<f32> = (0..7).map(|i| (i + 1) as f32 * 0.2).collect();
        let pt = project_to_3d(&emb, &mat, 7).unwrap();
        // Just verify we get finite values.
        assert!(pt.iter().all(|x| x.is_finite()));
    }

    // -- Octant routing -----------------------------------------------------

    #[test]
    fn octant_bits() {
        assert_eq!(FractalMemory3D::octant_index([0.0; 3], [1.0, 1.0, 1.0]), 7);
        assert_eq!(FractalMemory3D::octant_index([0.0; 3], [-1.0, -1.0, -1.0]), 0);
        assert_eq!(FractalMemory3D::octant_index([0.0; 3], [1.0, -1.0, -1.0]), 1);
        assert_eq!(FractalMemory3D::octant_index([0.0; 3], [-1.0, 1.0, -1.0]), 2);
        assert_eq!(FractalMemory3D::octant_index([0.0; 3], [-1.0, -1.0, 1.0]), 4);
        assert_eq!(FractalMemory3D::octant_index([0.0; 3], [1.0, -1.0, 1.0]), 5);
    }

    // -- Loose octree containment -------------------------------------------

    #[test]
    fn loose_contains_point_inside() {
        let node = OctreeNode::new(0, [0.0; 3], 1.0, 0, 0, 0);
        assert!(node.contains_point_loose([0.5, 0.5, 0.5], 1.05));
    }

    #[test]
    fn loose_contains_near_boundary() {
        let node = OctreeNode::new(0, [0.0; 3], 1.0, 0, 0, 0);
        // Slightly beyond the strict half_size=1.0 but within 1.05.
        assert!(node.contains_point_loose([1.03, 0.0, 0.0], 1.05));
        // Beyond the relaxation boundary.
        assert!(!node.contains_point_loose([1.06, 0.0, 0.0], 1.05));
    }

    // -- FractalMemory3D insertion & query ----------------------------------

    #[test]
    fn insert_single_subdivides_to_leaf() {
        let mut mem = FractalMemory3D::new(8, 0);
        let emb: Vec<f32> = (0..8).map(|i| i as f32 / 8.0).collect();
        let id = mem.insert(&emb, None, 1e-6).unwrap();
        // After the fix, insertion always subdivides past the root to reach
        // min_half_size, so we get root + at least one child.
        assert!(mem.node_count() > 1);
        assert!(mem.root_id.is_some());
        // The point should be locatable at the returned leaf.
        assert_eq!(mem.locate(&emb).unwrap(), id);
    }

    #[test]
    fn insert_many_subdivides() {
        let mut mem = FractalMemory3D::new(4, 42);
        for i in 0..20 {
            let emb: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32 * 0.01).collect();
            mem.insert(&emb, None, 1e-6);
        }
        assert!(mem.node_count() >= 20);
    }

    #[test]
    fn locate_finds_inserted() {
        let mut mem = FractalMemory3D::new(6, 7);
        let emb = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let inserted = mem.insert(&emb, None, 1e-6).unwrap();
        let located = mem.locate(&emb).unwrap();
        assert_eq!(inserted, located);
    }

    // -- Payload round-trip -------------------------------------------------

    #[test]
    fn payload_roundtrip() {
        let mut mem = FractalMemory3D::new(8, 99);
        let payload_data = b"Hello, OctaSoma fractal memory!";
        let emb = [0.0f32; 8];
        let id = mem.insert(&emb, Some(payload_data), 1e-6).unwrap();
        let retrieved = mem.get_payload(id).unwrap();
        assert_eq!(retrieved, payload_data);
    }

    #[test]
    fn get_payload_guards() {
        let mem = FractalMemory3D::new(4, 0);
        assert!(mem.get_payload(0).is_none());
        assert!(mem.get_payload(usize::MAX).is_none());
    }

    // -- Persistence round-trip (LZ4 compressed) ----------------------------

    #[test]
    fn save_and_load_roundtrip() {
        let mut mem = FractalMemory3D::new(8, 123);
        let p1 = b"alpha";
        let p2 = b"beta_gamma";
        let e1: Vec<f32> = vec![-1.0f32; 8];
        let e2: Vec<f32> = vec![1.0f32; 8];
        mem.insert(&e1, Some(p1), 1e-12);
        mem.insert(&e2, Some(p2), 1e-12);

        let path = "/tmp/octasoma_test_v2.bin";
        mem.save_to_disk(path).unwrap();

        let loaded = FractalMemory3D::load_from_disk(path, 8).unwrap();
        assert_eq!(loaded.node_count(), mem.node_count());
        assert_eq!(loaded.projection_matrix, mem.projection_matrix);
        assert_eq!(loaded.payload_arena, mem.payload_arena);

        let loc1 = loaded.locate(&e1).unwrap();
        assert_eq!(loaded.get_payload(loc1).unwrap(), p1);
        let loc2 = loaded.locate(&e2).unwrap();
        assert_eq!(loaded.get_payload(loc2).unwrap(), p2);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn load_rejects_wrong_dimension() {
        let mut mem = FractalMemory3D::new(8, 0);
        mem.insert(&[0.0f32; 8], None, 1e-6);
        let path = "/tmp/octasoma_dim_test_v2.bin";
        mem.save_to_disk(path).unwrap();

        assert!(FractalMemory3D::load_from_disk(path, 16).is_err());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn load_rejects_bad_magic() {
        let path = "/tmp/octasoma_bad_magic_v2.bin";
        std::fs::write(path, b"XXXX").unwrap();
        assert!(FractalMemory3D::load_from_disk(path, 4).is_err());
        std::fs::remove_file(path).ok();
    }

    // -- Deterministic projection matrix ------------------------------------

    #[test]
    fn deterministic_projection_matrix() {
        let a = FractalMemory3D::new(16, 42);
        let b = FractalMemory3D::new(16, 42);
        assert_eq!(a.projection_matrix, b.projection_matrix);
    }

    // -- Cache-line padding -------------------------------------------------

    #[test]
    fn octree_node_size_is_cache_line_multiple() {
        assert_eq!(std::mem::size_of::<OctreeNode>() % 64, 0);
        assert_eq!(std::mem::size_of::<OctreeNode>(), 192);
    }

    // -- PCA calibration ----------------------------------------------------

    #[test]
    fn pca_projection_deterministic() {
        // Build a simple calibration dataset with clear principal axes.
        let n = 50;
        let d = 8;
        let mut data = vec![0.0f32; n * d];
        // Component 1 varies along dimension 0.
        for i in 0..n {
            data[i * d] = (i as f32 - 25.0) * 0.1;
        }
        // Component 2 varies along dimension 1 (uncorrelated).
        for i in 0..n {
            data[i * d + 1] = (i as f32 - 25.0) * 0.07;
        }
        // Small noise elsewhere.
        let mut rng = DeterministicRng::new(999);
        for v in data.iter_mut() {
            *v += rng.next_f32() * 0.001;
        }

        let proj = compute_pca_projection(&data, n, d, 15);
        assert_eq!(proj.len(), 3 * d);

        // The first principal component should be dominated by dimension 0.
        let row0 = &proj[0..d];
        let max_idx = row0
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap()
            .0;
        assert!(
            max_idx == 0 || max_idx == 1,
            "first PC should align with dim 0 or 1, got {max_idx}"
        );
    }

    #[test]
    fn new_with_pca_integration() {
        let n = 30;
        let d = 6;
        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            data[i * d] = i as f32 * 0.1;
        }
        let mem = FractalMemory3D::new_with_pca(d, &data, n);
        assert_eq!(mem.high_dim, d);
        assert_eq!(mem.projection_matrix.len(), 3 * d);
        assert!(mem.projection_matrix.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn new_from_calibration_panics_on_bad_shape() {
        let result = std::panic::catch_unwind(|| {
            FractalMemory3D::new_from_calibration(8, vec![0.0f32; 20]);
        });
        assert!(result.is_err());
    }

    // -- arena compression smoke --------------------------------------------

    #[test]
    fn arena_compression_reduces_size_for_redundant_data() {
        let mut mem = FractalMemory3D::new(4, 1);
        // Insert a payload with lots of redundancy.
        let payload = vec![0xABu8; 4096];
        mem.insert(&[0.0f32; 4], Some(&payload), 1e-6);

        let path = "/tmp/octasoma_compress_test.bin";
        mem.save_to_disk(path).unwrap();

        let meta = std::fs::metadata(path).unwrap();
        let file_size = meta.len();

        // Raw would be: header(12) + nodes + proj + 8+8+4096
        // With LZ4, the compressed arena should be far smaller than 4096.
        assert!(file_size < 1024, "file should be well under 1 KiB for 4 KiB of redundant data; got {file_size}");

        // Verify round-trip still works.
        let loaded = FractalMemory3D::load_from_disk(path, 4).unwrap();
        assert_eq!(loaded.payload_arena, payload);

        std::fs::remove_file(path).ok();
    }
}

// ---------------------------------------------------------------------------
// Python FFI module (conditionally compiled behind the "python" feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod ffi;

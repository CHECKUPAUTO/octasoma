//! Pareto tuning of the SimHash precision tier — proposal C3 of
//! `docs/scirust-improvements.md`.
//!
//! The recall/cost trade-off of the precision tier is set by two constants the
//! docs hand-pick (`bits`, `shortlist`). This example turns them into a
//! **reproducible, seeded Pareto front** with SciRust's NSGA-II (a
//! dev-dependency — the library build keeps its single dependency): minimise
//! `(recall loss@10, scan cost)` over a deterministic clustered corpus, and
//! print the non-dominated configurations, directly answering "how many bits
//! and how large a shortlist does my data need?".
//!
//! Run with: `cargo run --release --example pareto_tuning`

use octasoma::{SketchIndex, metrics};
use scirust_evo::Nsga2;
use std::collections::HashSet;

const DIM: usize = 64;
const N: usize = 2000;
const CLUSTERS: usize = 40;
const K: usize = 10;
const QUERIES: usize = 32;
const GENERATIONS: usize = 25;

/// The discrete grid behind the continuous genome: bits ∈ {64..1024} (steps of
/// 64), shortlist ∈ {16..1024} (log-scaled).
fn decode(genome: &[f64]) -> (usize, usize) {
    let g0 = genome[0].clamp(0.0, 1.0);
    let g1 = genome[1].clamp(0.0, 1.0);
    let bits = 64 * (1 + (g0 * 15.0).round() as usize); // 64..=1024
    let shortlist = (16.0 * 64f64.powf(g1)).round() as usize; // 16..=1024, log scale
    (bits, shortlist.clamp(16, 1024))
}

fn main() {
    // Deterministic clustered corpus + held-out queries (LCG noise — see the
    // int8 test notes for why trig noise would lie about the margins).
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut noise = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let mut items = Vec::with_capacity(N);
    for c in 0..CLUSTERS {
        let base: Vec<f32> = (0..DIM)
            .map(|d| ((c * DIM + d) as f32 * 0.61).sin())
            .collect();
        for _ in 0..N / CLUSTERS {
            items.push(base.iter().map(|x| x + 0.3 * noise()).collect::<Vec<f32>>());
        }
    }
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|_| (0..DIM).map(|_| noise()).collect())
        .collect();

    // One index per distinct `bits` value (16 of them), built once — evaluation
    // of a genome is then just a query sweep at its shortlist.
    println!("[i] building 16 indexes over {N} items…");
    let indexes: Vec<SketchIndex> = (1..=16)
        .map(|w| {
            let mut idx = SketchIndex::new(DIM, 64 * w, 42);
            for (i, item) in items.iter().enumerate() {
                idx.insert(item, &(i as u64).to_le_bytes());
            }
            idx
        })
        .collect();
    let oracle: Vec<HashSet<u64>> = queries
        .iter()
        .map(|q| {
            indexes[0]
                .nearest(q, K, N)
                .into_iter()
                .map(|(p, _)| u64::from_le_bytes(p.try_into().unwrap()))
                .collect()
        })
        .collect();

    let evaluate = |bits: usize, shortlist: usize| -> (f64, f64) {
        let idx = &indexes[bits / 64 - 1];
        let mut loss = 0.0;
        for (q, oracle_ids) in queries.iter().zip(&oracle) {
            let got: Vec<u64> = idx
                .nearest(q, K, shortlist)
                .into_iter()
                .map(|(p, _)| u64::from_le_bytes(p.try_into().unwrap()))
                .collect();
            loss += 1.0 - metrics::recall_at_k(&got, oracle_ids, K);
        }
        // Cost model: the Hamming scan (N popcount words) + the exact rerank
        // (shortlist × dim MACs), normalised to the most expensive corner.
        let cost = (N * (bits / 64)) as f64 + (shortlist * DIM) as f64;
        let max_cost = (N * 16) as f64 + (1024 * DIM) as f64;
        (loss / QUERIES as f64, cost / max_cost)
    };

    let mut nsga = Nsga2::seeded(42);
    nsga.bounds = (0.0, 1.0);
    let mut population = nsga.init_pop(2);
    for generation in 0..GENERATIONS {
        nsga.evolve(&mut population, |pop| {
            pop.iter()
                .map(|ind| {
                    let (bits, shortlist) = decode(&ind.genome);
                    let (loss, cost) = evaluate(bits, shortlist);
                    vec![loss, cost]
                })
                .collect()
        });
        if generation % 5 == 4 {
            println!("[i] generation {}/{GENERATIONS}", generation + 1);
        }
    }

    // The Pareto front (rank 1 in this NSGA-II), deduplicated on the decoded grid.
    let mut front: Vec<(usize, usize, f64, f64)> = population
        .iter()
        .filter(|ind| ind.rank == 1)
        .map(|ind| {
            let (bits, shortlist) = decode(&ind.genome);
            let (loss, cost) = evaluate(bits, shortlist);
            (bits, shortlist, loss, cost)
        })
        .collect();
    front.sort_by(|a, b| a.3.total_cmp(&b.3).then(a.0.cmp(&b.0)));
    front.dedup_by_key(|e| (e.0, e.1));

    println!("\nPareto front (recall loss@{K} vs normalised scan cost):");
    println!(
        "{:>6} {:>9} {:>12} {:>10}",
        "bits", "shortlist", "recall_loss", "cost"
    );
    for (bits, shortlist, loss, cost) in &front {
        println!("{bits:>6} {shortlist:>9} {loss:>12.4} {cost:>10.4}");
    }
    println!(
        "\n[i] pick the cheapest row whose recall loss clears your target — or feed\n\
         [i] these candidates to SketchIndex::certify_shortlist for a guarantee."
    );
}

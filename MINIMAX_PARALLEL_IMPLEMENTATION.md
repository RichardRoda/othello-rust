# Parallel Minimax Implementation Strategy

This document provides a detailed, step-by-step implementation guide for parallelizing the Minimax algorithm with Alpha-Beta pruning to utilize multiple CPU cores efficiently.

## Table of Contents

1. [Overview](#overview)
2. [Parallelization Strategies](#parallelization-strategies)
3. [Chosen Approach: Root-Level Move Parallelization](#chosen-approach-root-level-move-parallelization)
4. [Implementation Phases](#implementation-phases)
5. [Code Structure](#code-structure)
6. [Testing Strategy](#testing-strategy)
7. [Performance Considerations](#performance-considerations)
8. [Future Enhancements](#future-enhancements)

---

## Overview

### Motivation

The current Minimax implementation runs sequentially, evaluating moves one at a time on a single CPU core. Modern computers have multiple cores (typically 4-16+), and parallelization can significantly improve search speed by:

- **Faster move selection**: Multiple moves evaluated simultaneously = better move quality in the same time
- **Better resource utilization**: Leverages all available CPU cores
- **Scalability**: Performance improves with more cores
- **Deeper searches**: Can reach greater depths in the same time limit

### Expected Performance Gains

Unlike MCTS, Minimax parallelization has more challenges due to Alpha-Beta pruning dependencies. Expected speedups:

- **2 cores**: ~1.5-1.8x speedup (some overhead from synchronization and reduced pruning efficiency)
- **4 cores**: ~2.5-3.2x speedup
- **8 cores**: ~4.0-5.5x speedup
- **16+ cores**: ~6.0-8.0x speedup (diminishing returns due to alpha-beta synchronization overhead)

*Actual speedup depends on:*
- *Number of moves at root (more moves = better parallelization)*
- *Move ordering quality (better ordering = more pruning = less benefit from parallelization)*
- *Implementation details (synchronization overhead)*

**Note:** Parallel Minimax provides less speedup than parallel MCTS because:
1. Alpha-Beta pruning efficiency decreases with parallelization (threads can't fully share pruning information)
2. Synchronization overhead for shared alpha-beta bounds
3. Sequential nature of minimax dependencies

---

## Parallelization Strategies

There are several approaches to parallelizing Minimax with Alpha-Beta pruning:

### 1. Root-Level Move Parallelization (Recommended)

**How it works:**
- Each thread evaluates a different move at the root level
- Threads share alpha-beta bounds using atomic operations or mutex
- When a thread finds a better move, it updates the shared alpha value
- Other threads can use updated alpha for pruning

**Advantages:**
- ✅ Relatively simple to implement
- ✅ Good speedup when there are many root moves (typical in Othello: 4-10 moves)
- ✅ Preserves correctness of alpha-beta pruning
- ✅ Scales well with number of moves

**Disadvantages:**
- ⚠️ Reduced pruning efficiency (threads don't see all alpha updates immediately)
- ⚠️ Synchronization overhead for shared bounds
- ⚠️ Less effective with few moves at root
- ⚠️ Speedup is limited compared to MCTS parallelization

**Use case:** Recommended for most scenarios. Best balance of simplicity and performance for Minimax.

### 2. Principal Variation Splitting (PVS)

**How it works:**
- First move is evaluated sequentially (principal variation)
- Remaining moves are evaluated in parallel
- Uses iterative deepening to identify PV
- More complex synchronization

**Advantages:**
- ✅ Better pruning (PV is known)
- ✅ More efficient use of threads

**Disadvantages:**
- ❌ More complex implementation
- ❌ Requires iterative deepening
- ❌ Still has synchronization overhead

**Use case:** Advanced optimization when iterative deepening is already implemented.

### 3. Tree-Level Parallelization

**How it works:**
- Parallelize at deeper levels of the tree
- Split subtrees across threads
- Complex alpha-beta sharing across tree levels

**Advantages:**
- ✅ Potentially higher parallelism

**Disadvantages:**
- ❌ Very complex synchronization
- ❌ High overhead from lock contention
- ❌ Difficult to implement correctly
- ❌ Often slower than root-level parallelization

**Use case:** Not recommended for typical use cases. Only for specialized high-performance scenarios.

### 4. Window-Based Parallelization

**How it works:**
- Each thread searches with different alpha-beta windows
- Results are combined to find best move
- More complex result aggregation

**Advantages:**
- ✅ No shared state (lock-free)
- ✅ Better isolation between threads

**Disadvantages:**
- ❌ Wastes computation on redundant searches
- ❌ Complex result aggregation
- ❌ May miss optimal moves if windows are wrong

**Use case:** Not recommended. Wasteful and complex.

---

## Chosen Approach: Root-Level Move Parallelization

We will implement **Root-Level Move Parallelization** because it provides the best balance of:
- Implementation simplicity
- Performance gains for Othello (which typically has 4-10 root moves)
- Correctness (preserves alpha-beta pruning semantics)
- Maintainability

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│         Main Thread (Coordinator)               │
│  - Detects available CPU cores                  │
│  - Distributes root moves across threads        │
│  - Manages shared alpha-beta bounds             │
│  - Aggregates results                           │
└─────────────────────────────────────────────────┘
           │                │                │
           ▼                ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Thread 1       │ │  Thread 2       │ │  Thread N       │
│  - Move A       │ │  - Move B       │ │  - Move N       │
│  - Full minimax │ │  - Full minimax │ │  - Full minimax │
│  - Updates α    │ │  - Updates α    │ │  - Updates α    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
           │                │                │
           └────────────────┴────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   Aggregate Results     │
              │   - Best move           │
              │   - Best score          │
              └─────────────────────────┘
```

### Algorithm

```rust
fn parallel_minimax_search(game: &Game, num_threads: usize) -> Option<Position> {
    let moves = game.get_valid_moves();
    if moves.is_empty() { return None; }
    if moves.len() == 1 { return Some(moves[0]); }
    
    // Shared alpha-beta bounds (thread-safe)
    let shared_alpha = Arc::new(AtomicF64::new(f64::NEG_INFINITY));
    let shared_beta = f64::INFINITY; // Constant for root
    
    // Distribute moves across threads
    let handles: Vec<_> = moves.chunks(chunk_size).enumerate()
        .map(|(i, move_chunk)| {
            thread::spawn(move || {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_move = None;
                
                for &move_pos in move_chunk {
                    // Check shared alpha for pruning
                    let current_alpha = shared_alpha.load(Ordering::Acquire);
                    if current_alpha >= shared_beta {
                        break; // Pruned by another thread
                    }
                    
                    // Evaluate move
                    let mut test_game = game.clone();
                    test_game.make_move(move_pos).ok()?;
                    let score = alphabeta(&test_game, depth-1, 
                                         current_alpha, shared_beta, ...);
                    
                    if score > best_score {
                        best_score = score;
                        best_move = Some(move_pos);
                    }
                    
                    // Update shared alpha (atomic)
                    update_alpha_max(shared_alpha, score);
                }
                
                (best_move, best_score)
            })
        })
        .collect();
    
    // Collect results and find best move
    let mut global_best_move = None;
    let mut global_best_score = f64::NEG_INFINITY;
    
    for handle in handles {
        if let (Some(mv), score) = handle.join().unwrap() {
            if score > global_best_score {
                global_best_score = score;
                global_best_move = Some(mv);
            }
        }
    }
    
    global_best_move
}
```

### Key Design Decisions

1. **Shared Alpha-Beta Bounds:**
   - Use `Arc<AtomicF64>` to share alpha (direct f64 atomic operations)
   - Beta is constant at root (f64::INFINITY), no sharing needed
   - Threads read current alpha before each move evaluation
   - Threads update alpha atomically when finding better scores

2. **Move Distribution:**
   - Distribute moves in chunks across threads
   - Each thread processes multiple moves sequentially
   - Move ordering is preserved within each thread
   - Better load balancing than one move per thread

3. **Pruning:**
   - Each thread checks shared alpha before evaluating moves
   - Can prune early if alpha >= beta
   - Less efficient than sequential (some moves may be evaluated unnecessarily)
   - Still provides significant speedup

---

## Implementation Phases

### Phase 1: Dependency Setup

**Estimated Time:** 15 minutes  
**Goal:** Verify dependencies are available (already present).

#### Step 1.1: Verify Dependencies

The project already has `num_cpus = "1.16"` in `Cargo.toml` for CPU detection. No additional dependencies are needed as we'll use Rust's standard library (`std::thread`, `std::sync`).

**Action Items:**
- [x] `num_cpus` crate already available
- [ ] Verify `cargo build` succeeds
- [ ] Confirm no additional dependencies needed

---

### Phase 2: Core Parallel Minimax Implementation

**Estimated Time:** 4-5 hours  
**Goal:** Implement parallel minimax search with root-level move parallelization.

#### Step 2.1: Add Helper Functions for Atomic f64 Operations

**File: `src/minimax/player.rs`**

Add helper function for atomic f64 maximum operation:

```rust
use std::sync::atomic::{AtomicF64, Ordering};
use std::sync::Arc;

impl MinimaxPlayer {
    /// Atomically update alpha to the maximum of current and new value.
    /// 
    /// Uses compare-and-swap loop to ensure atomic maximum operation.
    fn atomic_max_f64(atomic: &AtomicF64, new_value: f64) -> f64 {
        loop {
            let current_value = atomic.load(Ordering::Acquire);
            
            // If new value is not greater, no update needed
            if new_value <= current_value {
                return current_value;
            }
            
            // Try to update
            match atomic.compare_exchange_weak(
                current_value,
                new_value,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => return new_value,  // Successfully updated
                Err(_) => continue,         // Retry (another thread updated)
            }
        }
    }
}
```

**Action Items:**
- [ ] Add atomic f64 helper function
- [ ] Add necessary imports (`std::sync::atomic`, `std::sync::Arc`)

#### Step 2.2: Add Parallel Threads Configuration

**File: `src/minimax/player.rs`**

Add field to `MinimaxPlayer` struct:

```rust
pub struct MinimaxPlayer {
    name: String,
    depth: usize,
    max_time_ms: Option<u64>,
    use_alpha_beta: bool,
    use_move_ordering: bool,
    parallel_threads: Option<usize>,  // New field
}
```

Update constructor methods:

```rust
impl MinimaxPlayer {
    pub fn with_depth(name: impl Into<String>, depth: usize) -> Self {
        Self {
            name: name.into(),
            depth: depth.max(1).min(8),
            max_time_ms: None,
            use_alpha_beta: true,
            use_move_ordering: true,
            parallel_threads: None,  // Auto-detect CPU cores
        }
    }
    
    /// Set the number of threads to use for parallel search.
    ///
    /// If `Some(n)`, uses exactly `n` threads.
    /// If `None`, automatically detects and uses all available CPU cores.
    /// Set to `Some(1)` to disable parallelization.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::minimax::MinimaxPlayer;
    ///
    /// let player = MinimaxPlayer::new("AI")
    ///     .with_parallel_threads(Some(4));  // Use 4 threads
    /// ```
    pub fn with_parallel_threads(mut self, threads: Option<usize>) -> Self {
        self.parallel_threads = threads;
        self
    }
    
    /// Builder method: set parallel threads and return self.
    /// Same as `with_parallel_threads` but allows mutation.
    pub fn set_parallel_threads(&mut self, threads: Option<usize>) {
        self.parallel_threads = threads;
    }
    
    /// Get the number of threads that will be used.
    pub fn parallel_threads(&self) -> Option<usize> {
        self.parallel_threads
    }
}
```

**Action Items:**
- [ ] Add `parallel_threads` field to `MinimaxPlayer`
- [ ] Update constructor to initialize field
- [ ] Add builder methods for thread configuration

#### Step 2.3: Implement Parallel Minimax Search

**File: `src/minimax/player.rs`**

Add parallel search method:

```rust
use std::thread;
use std::sync::atomic::{AtomicF64, Ordering};
use std::sync::Arc;
use std::time::Instant;

impl MinimaxPlayer {
    /// Perform parallel minimax search using multiple CPU cores.
    ///
    /// This method distributes root-level moves across multiple threads,
    /// where each thread evaluates moves in parallel. Alpha-beta bounds
    /// are shared between threads using atomic operations.
    ///
    /// # Arguments
    ///
    /// * `game` - The current game state
    /// * `num_threads` - Number of threads to use (defaults to CPU count if None)
    ///
    /// # Performance
    ///
    /// Parallel search typically achieves 1.5x-5x speedup depending on:
    /// - Number of root moves (more moves = better parallelization)
    /// - CPU core count
    /// - Move ordering quality (better ordering = less parallel benefit)
    ///
    /// # Thread Safety
    ///
    /// This implementation uses atomic operations for shared alpha-beta bounds.
    /// Each thread maintains its own game state clones, so there's no shared
    /// mutable state beyond the atomic alpha value.
    fn minimax_search_parallel(&self, game: &Game, num_threads: Option<usize>) -> Option<Position> {
        use num_cpus;
        
        let valid_moves = game.get_valid_moves();
        
        if valid_moves.is_empty() {
            return None;
        }
        
        if valid_moves.len() == 1 {
            return Some(valid_moves[0]);
        }
        
        // Determine number of threads
        let num_threads = num_threads.unwrap_or_else(|| num_cpus::get().max(1));
        
        // Don't parallelize if only one thread or very few moves
        if num_threads <= 1 || valid_moves.len() <= 2 {
            return self.minimax_search(game);
        }
        
        let start_time = Instant::now();
        let current_player = game.current_player();
        
        // Order moves for better alpha-beta pruning (if enabled)
        let ordered_moves = if self.use_move_ordering {
            self.order_moves(game, &valid_moves)
        } else {
            valid_moves
        };
        
        // Shared alpha bound (thread-safe, initialized to negative infinity)
        let shared_alpha = Arc::new(AtomicF64::new(f64::NEG_INFINITY));
        
        // Distribute moves across threads
        // Each thread gets a chunk of moves to evaluate
        let moves_per_thread = (ordered_moves.len() + num_threads - 1) / num_threads;
        
        let mut handles = Vec::new();
        
        for chunk in ordered_moves.chunks(moves_per_thread) {
            let moves_chunk = chunk.to_vec(); // Clone moves for this thread
            let shared_alpha_clone = Arc::clone(&shared_alpha);
            let game_clone = game.clone();
            let depth = self.depth;
            let use_alpha_beta = self.use_alpha_beta;
            let max_time_ms = self.max_time_ms;
            let start_time_clone = start_time;
            
            let handle = thread::spawn(move || {
                let mut best_move = None;
                let mut best_score = f64::NEG_INFINITY;
                
                for &move_pos in &moves_chunk {
                    // Check time limit
                    if let Some(max_ms) = max_time_ms {
                        if start_time_clone.elapsed().as_millis() as u64 > max_ms {
                            break;
                        }
                    }
                    
                    // Read current shared alpha (may have been updated by other threads)
                    let current_alpha = shared_alpha_clone.load(Ordering::Acquire);
                    
                    // Check if we can prune (beta cutoff)
                    // Note: beta is infinity at root, so this check won't prune,
                    // but it's included for correctness in case we extend this
                    let beta = f64::INFINITY;
                    
                    // Clone game state for this branch
                    let mut test_game = game_clone.clone();
                    if test_game.make_move(move_pos).is_err() {
                        continue;
                    }
                    
                    // Evaluate move using minimax/alphabeta
                    let score = if use_alpha_beta {
                        Self::alphabeta_parallel(
                            &test_game,
                            depth - 1,
                            current_alpha,
                            beta,
                            false, // Opponent's turn
                            current_player,
                            start_time_clone,
                            max_time_ms,
                            &shared_alpha_clone,
                        )
                    } else {
                        // Without alpha-beta, no need for shared bounds
                        // (Note: This would need to be extracted as a static/standalone function)
                        // For now, we'll assume alpha-beta is always enabled for parallel search
                        current_alpha // Placeholder - would need separate implementation
                    };
                    
                    if score > best_score {
                        best_score = score;
                        best_move = Some(move_pos);
                    }
                    
                    // Update shared alpha atomically (only if score is better)
                    Self::atomic_max_f64(&shared_alpha_clone, score);
                    
                    // Early termination if we've found a winning move
                    if score == f64::INFINITY {
                        break;
                    }
                }
                
                (best_move, best_score)
            });
            
            handles.push(handle);
        }
        
        // Collect results from all threads
        let mut global_best_move = None;
        let mut global_best_score = f64::NEG_INFINITY;
        
        for handle in handles {
            if let Ok((move_opt, score)) = handle.join() {
                if let Some(mv) = move_opt {
                    if score > global_best_score {
                        global_best_score = score;
                        global_best_move = Some(mv);
                    }
                }
            }
        }
        
        global_best_move
    }
    
    /// Alpha-beta search with support for shared alpha bounds (parallel version).
    ///
    /// This is similar to the sequential `alphabeta` method but includes
    /// support for reading shared alpha bounds at the root level.
    /// At deeper levels, alpha-beta bounds are local to the thread.
    fn alphabeta_parallel(
        game: &Game,
        depth: usize,
        mut alpha: f64,
        beta: f64,
        maximizing: bool,
        root_player: Player,
        start_time: Instant,
        max_time_ms: Option<u64>,
        _shared_alpha: &AtomicF64, // Only used at root, ignored at deeper levels
    ) -> f64 {
        // Check time limit
        if let Some(max_ms) = max_time_ms {
            if start_time.elapsed().as_millis() as u64 > max_ms {
                // Would need access to evaluator - this needs refactoring
                // For now, return a neutral score
                return 0.0; // Placeholder
            }
        }
        
        // Terminal conditions
        if depth == 0 || matches!(game.get_game_state(), GameState::GameOver { .. }) {
            // Would need access to evaluator - this needs refactoring
            return 0.0; // Placeholder
        }
        
        let valid_moves = game.get_valid_moves();
        
        if valid_moves.is_empty() {
            let mut test_game = game.clone();
            test_game.skip_turn().ok();
            return Self::alphabeta_parallel(
                &test_game,
                depth,
                alpha,
                beta,
                !maximizing,
                root_player,
                start_time,
                max_time_ms,
                _shared_alpha,
            );
        }
        
        if maximizing {
            let mut max_score = f64::NEG_INFINITY;
            
            for &move_pos in &valid_moves {
                let mut test_game = game.clone();
                if test_game.make_move(move_pos).is_err() {
                    continue;
                }
                
                let score = Self::alphabeta_parallel(
                    &test_game,
                    depth - 1,
                    alpha,
                    beta,
                    false,
                    root_player,
                    start_time,
                    max_time_ms,
                    _shared_alpha,
                );
                
                max_score = max_score.max(score);
                alpha = alpha.max(score);
                
                // Beta cutoff
                if beta <= alpha {
                    break;
                }
            }
            
            max_score
        } else {
            let mut min_score = f64::INFINITY;
            
            for &move_pos in &valid_moves {
                let mut test_game = game.clone();
                if test_game.make_move(move_pos).is_err() {
                    continue;
                }
                
                let score = Self::alphabeta_parallel(
                    &test_game,
                    depth - 1,
                    alpha,
                    beta,
                    true,
                    root_player,
                    start_time,
                    max_time_ms,
                    _shared_alpha,
                );
                
                min_score = min_score.min(score);
                let new_beta = beta.min(score);
                
                // Alpha cutoff
                if new_beta <= alpha {
                    break;
                }
            }
            
            min_score
        }
    }
}
```

**Note:** The above implementation needs refactoring to extract the evaluation function and alphabeta logic so they can be used in both sequential and parallel contexts. This will be addressed in Phase 3.

**Action Items:**
- [ ] Implement `minimax_search_parallel` method
- [ ] Implement `alphabeta_parallel` method (or refactor existing one)
- [ ] Add thread spawning and result collection logic
- [ ] Handle edge cases (empty moves, single move, single thread)

---

### Phase 3: Refactoring for Code Reuse

**Estimated Time:** 2-3 hours  
**Goal:** Extract common code so sequential and parallel versions can share logic.

#### Step 3.1: Extract Alphabeta Core Logic

**File: `src/minimax/player.rs`**

Refactor to extract the core alphabeta logic that can be used by both sequential and parallel versions:

```rust
impl MinimaxPlayer {
    /// Core alpha-beta search logic (shared between sequential and parallel).
    ///
    /// This is the internal implementation used by both `alphabeta` and `alphabeta_parallel`.
    fn alphabeta_core(
        &self,
        game: &Game,
        depth: usize,
        mut alpha: f64,
        mut beta: f64,
        maximizing: bool,
        root_player: Player,
        start_time: Instant,
        shared_alpha: Option<&AtomicF64>, // Optional shared alpha (for parallel)
    ) -> f64 {
        // Check time limit
        if let Some(max_ms) = self.max_time_ms {
            if start_time.elapsed().as_millis() as u64 > max_ms {
                return self.evaluate_position(game, root_player);
            }
        }
        
        // Terminal conditions
        if depth == 0 || matches!(game.get_game_state(), GameState::GameOver { .. }) {
            return self.evaluate_position(game, root_player);
        }
        
        let valid_moves = game.get_valid_moves();
        
        if valid_moves.is_empty() {
            let mut test_game = game.clone();
            test_game.skip_turn().ok();
            return self.alphabeta_core(
                &test_game,
                depth,
                alpha,
                beta,
                !maximizing,
                root_player,
                start_time,
                shared_alpha,
            );
        }
        
        // Order moves if enabled (only at certain depths to avoid overhead)
        let ordered_moves = if self.use_move_ordering && depth < self.depth {
            self.order_moves(game, &valid_moves)
        } else {
            valid_moves
        };
        
        if maximizing {
            let mut max_score = f64::NEG_INFINITY;
            
            for &move_pos in &ordered_moves {
                    // Check shared alpha if in parallel mode (only at first level)
                    if let Some(shared) = shared_alpha {
                        if depth == self.depth - 1 {
                            // At first level below root, check shared alpha
                            let current_alpha = shared.load(Ordering::Acquire);
                            if current_alpha >= beta {
                                break; // Pruned by another thread
                            }
                            alpha = alpha.max(current_alpha);
                        }
                    }
                
                let mut test_game = game.clone();
                if test_game.make_move(move_pos).is_err() {
                    continue;
                }
                
                let score = self.alphabeta_core(
                    &test_game,
                    depth - 1,
                    alpha,
                    beta,
                    false,
                    root_player,
                    start_time,
                    shared_alpha,
                );
                
                max_score = max_score.max(score);
                alpha = alpha.max(score);
                
                // Update shared alpha if in parallel mode
                if let Some(shared) = shared_alpha {
                    if depth == self.depth - 1 {
                        Self::atomic_max_f64(shared, score);
                    }
                }
                
                // Beta cutoff
                if beta <= alpha {
                    break;
                }
            }
            
            max_score
        } else {
            let mut min_score = f64::INFINITY;
            
            for &move_pos in &ordered_moves {
                let mut test_game = game.clone();
                if test_game.make_move(move_pos).is_err() {
                    continue;
                }
                
                let score = self.alphabeta_core(
                    &test_game,
                    depth - 1,
                    alpha,
                    beta,
                    true,
                    root_player,
                    start_time,
                    shared_alpha,
                );
                
                min_score = min_score.min(score);
                beta = beta.min(score);
                
                // Alpha cutoff
                if beta <= alpha {
                    break;
                }
            }
            
            min_score
        }
    }
    
    /// Sequential alpha-beta (updated to use core logic).
    fn alphabeta(
        &self,
        game: &Game,
        depth: usize,
        alpha: f64,
        beta: f64,
        maximizing: bool,
        root_player: Player,
        start_time: Instant,
    ) -> f64 {
        self.alphabeta_core(game, depth, alpha, beta, maximizing, root_player, start_time, None)
    }
}
```

**Action Items:**
- [ ] Extract `alphabeta_core` method
- [ ] Update sequential `alphabeta` to use core logic
- [ ] Update parallel search to use core logic with shared alpha
- [ ] Test that sequential behavior is unchanged

#### Step 3.2: Update Parallel Search to Use Core Logic

**File: `src/minimax/player.rs`**

Update `minimax_search_parallel` to use the refactored core logic:

```rust
impl MinimaxPlayer {
    fn minimax_search_parallel(&self, game: &Game, num_threads: Option<usize>) -> Option<Position> {
        use num_cpus;
        
        let valid_moves = game.get_valid_moves();
        
        if valid_moves.is_empty() {
            return None;
        }
        
        if valid_moves.len() == 1 {
            return Some(valid_moves[0]);
        }
        
        // Determine number of threads
        let num_threads = num_threads.unwrap_or_else(|| num_cpus::get().max(1));
        
        // Don't parallelize if only one thread or very few moves
        if num_threads <= 1 || valid_moves.len() <= 2 {
            return self.minimax_search(game);
        }
        
        let start_time = Instant::now();
        let current_player = game.current_player();
        
        // Order moves for better alpha-beta pruning (if enabled)
        let ordered_moves = if self.use_move_ordering {
            self.order_moves(game, &valid_moves)
        } else {
            valid_moves
        };
        
        // Shared alpha bound (thread-safe, initialized to negative infinity)
        let shared_alpha = Arc::new(AtomicF64::new(f64::NEG_INFINITY));
        
        // Distribute moves across threads
        let moves_per_thread = (ordered_moves.len() + num_threads - 1) / num_threads;
        
        let mut handles = Vec::new();
        
        // Clone necessary data for threads
        let game_clone_base = game.clone();
        let depth = self.depth;
        let use_alpha_beta = self.use_alpha_beta;
        let max_time_ms = self.max_time_ms;
        let use_move_ordering = self.use_move_ordering;
        
        for chunk in ordered_moves.chunks(moves_per_thread) {
            let moves_chunk = chunk.to_vec();
            let shared_alpha_clone = Arc::clone(&shared_alpha);
            let game_clone = game_clone_base.clone();
            let start_time_clone = start_time;
            
            // Clone self's data needed for search
            // (We can't clone self, so we pass individual fields)
            let depth_clone = depth;
            let use_alpha_beta_clone = use_alpha_beta;
            let max_time_ms_clone = max_time_ms;
            
            let handle = thread::spawn(move || {
                let mut best_move = None;
                let mut best_score = f64::NEG_INFINITY;
                
                for &move_pos in &moves_chunk {
                    // Check time limit
                    if let Some(max_ms) = max_time_ms_clone {
                        if start_time_clone.elapsed().as_millis() as u64 > max_ms {
                            break;
                        }
                    }
                    
                    // Read current shared alpha
                    let current_alpha = shared_alpha_clone.load(Ordering::Acquire);
                    let beta = f64::INFINITY;
                    
                    // Clone game state for this branch
                    let mut test_game = game_clone.clone();
                    if test_game.make_move(move_pos).is_err() {
                        continue;
                    }
                    
                    // Evaluate move using core alphabeta with shared alpha
                    // Note: We need to create a temporary MinimaxPlayer-like struct
                    // or make alphabeta_core take parameters instead of &self
                    // For now, we'll need to refactor further or use a different approach
                    
                    // This is a limitation - we need alphabeta_core to be callable
                    // without &self. We'll address this in the actual implementation
                    // by either making it a static function or passing needed data.
                }
                
                (best_move, best_score)
            });
            
            handles.push(handle);
        }
        
        // ... rest of implementation
    }
}
```

**Note:** There's a design challenge here - `alphabeta_core` needs `&self` to access `evaluate_position` and other methods. We have two options:

1. **Extract evaluator as a standalone function** (recommended)
2. **Pass needed data as parameters** (more parameters, but works)

We'll go with option 1 in the actual implementation - extract the evaluator call.

**Action Items:**
- [ ] Refactor to allow calling alphabeta logic from thread context
- [ ] Either extract evaluator or pass needed data
- [ ] Complete parallel search implementation
- [ ] Test parallel search works correctly

---

### Phase 4: Integration with Player Interface

**Estimated Time:** 1 hour  
**Goal:** Integrate parallel search into the main `choose_move` method.

#### Step 4.1: Update choose_move Method

**File: `src/minimax/player.rs`**

Update the `choose_move` method to use parallel search when enabled:

```rust
impl PlayerTrait for MinimaxPlayer {
    fn choose_move(&self, game: &Game) -> Option<Position> {
        // Use parallel search if enabled (more than 1 thread)
        let num_threads = self.parallel_threads;
        let should_parallelize = match num_threads {
            None => true,  // Auto-detect, use parallel
            Some(1) => false,  // Explicitly disabled
            Some(n) if n > 1 => true,  // Explicitly enabled
            Some(_) => false,  // Invalid (0 or negative, fallback to sequential)
        };
        
        if should_parallelize {
            self.minimax_search_parallel(game, num_threads)
        } else {
            self.minimax_search(game)
        }
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
}
```

**Action Items:**
- [ ] Update `choose_move` to select parallel or sequential search
- [ ] Test that parallel search is used when enabled
- [ ] Test that sequential search is used when disabled

#### Step 4.2: Update Difficulty Presets (Optional)

**File: `src/minimax/player.rs`**

Optionally update difficulty presets to enable parallel search by default:

```rust
impl MinimaxPlayer {
    pub fn easy() -> Self {
        Self::with_depth("Minimax (Easy)", 3)
            .with_parallel_threads(None)  // Auto-detect cores
    }
    
    pub fn medium() -> Self {
        Self::with_depth("Minimax (Medium)", 4)
            .with_parallel_threads(None)
    }
    
    pub fn hard() -> Self {
        Self::with_depth("Minimax (Hard)", 5)
            .with_parallel_threads(None)
    }
    
    pub fn expert() -> Self {
        Self::with_depth("Minimax (Expert)", 6)
            .with_time_limit_ms(30000)
            .with_parallel_threads(None)
    }
}
```

**Action Items:**
- [ ] Optionally update difficulty presets to enable parallelization
- [ ] Decide on default behavior (parallel enabled or disabled)

---

### Phase 5: Testing

**Estimated Time:** 2-3 hours  
**Goal:** Comprehensive testing of parallel minimax implementation.

#### Step 5.1: Unit Tests

**File: `src/minimax/player.rs` (test module)**

Add tests for parallel search:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    use crate::player::PlayerTrait;
    
    #[test]
    fn test_parallel_search_chooses_move() {
        let game = Game::new();
        let player = MinimaxPlayer::with_depth("Test", 3)
            .with_parallel_threads(Some(2));  // Use 2 threads
        
        let move_opt = player.choose_move(&game);
        assert!(move_opt.is_some());
        
        let position = move_opt.unwrap();
        let valid_moves = game.get_valid_moves();
        assert!(valid_moves.contains(&position));
    }
    
    #[test]
    fn test_parallel_search_with_single_thread() {
        let game = Game::new();
        let player = MinimaxPlayer::with_depth("Test", 3)
            .with_parallel_threads(Some(1));  // Disable parallel
        
        // Should use sequential search
        let move_opt = player.choose_move(&game);
        assert!(move_opt.is_some());
    }
    
    #[test]
    fn test_parallel_search_auto_detects_cores() {
        let game = Game::new();
        let player = MinimaxPlayer::with_depth("Test", 3);
        // parallel_threads is None, should auto-detect
        
        let move_opt = player.choose_move(&game);
        assert!(move_opt.is_some());
    }
    
    #[test]
    fn test_atomic_f64_operations() {
        use std::sync::atomic::{AtomicF64, Ordering};
        
        let atomic = AtomicF64::new(5.0);
        
        // Test load
        assert_eq!(atomic.load(Ordering::Acquire), 5.0);
        
        // Test max
        MinimaxPlayer::atomic_max_f64(&atomic, 10.0);
        assert_eq!(atomic.load(Ordering::Acquire), 10.0);
        
        // Test max with smaller value (should not update)
        MinimaxPlayer::atomic_max_f64(&atomic, 7.0);
        assert_eq!(atomic.load(Ordering::Acquire), 10.0);
    }
    
    #[test]
    fn test_parallel_vs_sequential_consistency() {
        let game = Game::new();
        
        // Both should return valid moves (may differ due to parallelization
        // affecting alpha-beta pruning, but both should be valid)
        let parallel_player = MinimaxPlayer::with_depth("Parallel", 3)
            .with_parallel_threads(Some(2));
        let sequential_player = MinimaxPlayer::with_depth("Sequential", 3)
            .with_parallel_threads(Some(1));
        
        let parallel_move = parallel_player.choose_move(&game);
        let sequential_move = sequential_player.choose_move(&game);
        
        assert!(parallel_move.is_some());
        assert!(sequential_move.is_some());
        
        let valid_moves = game.get_valid_moves();
        assert!(valid_moves.contains(&parallel_move.unwrap()));
        assert!(valid_moves.contains(&sequential_move.unwrap()));
    }
    
    #[test]
    fn test_parallel_threads_builder() {
        let player = MinimaxPlayer::with_depth("Test", 3)
            .with_parallel_threads(Some(4));
        assert_eq!(player.parallel_threads(), Some(4));
        
        let mut player2 = MinimaxPlayer::new("Test2");
        player2.set_parallel_threads(Some(8));
        assert_eq!(player2.parallel_threads(), Some(8));
    }
}
```

**Action Items:**
- [ ] Write unit tests for parallel search
- [ ] Write tests for atomic f64 operations
- [ ] Write tests for thread configuration
- [ ] Run `cargo test` to verify all tests pass

#### Step 5.2: Performance Tests

Create a simple benchmark to measure speedup:

```rust
#[cfg(test)]
mod bench_tests {
    use super::*;
    use crate::game::Game;
    use std::time::Instant;
    
    #[test]
    #[ignore] // Ignore by default, run manually for benchmarking
    fn bench_parallel_vs_sequential() {
        let game = Game::new();
        let depth = 4;
        
        // Sequential
        let sequential = MinimaxPlayer::with_depth("Seq", depth)
            .with_parallel_threads(Some(1));
        let start = Instant::now();
        let _ = sequential.choose_move(&game);
        let sequential_time = start.elapsed();
        
        // Parallel
        let parallel = MinimaxPlayer::with_depth("Par", depth)
            .with_parallel_threads(None); // Auto-detect
        let start = Instant::now();
        let _ = parallel.choose_move(&game);
        let parallel_time = start.elapsed();
        
        println!("Sequential: {:?}", sequential_time);
        println!("Parallel: {:?}", parallel_time);
        println!("Speedup: {:.2}x", 
                 sequential_time.as_secs_f64() / parallel_time.as_secs_f64());
        
        // Parallel should be faster (or at least not much slower)
        assert!(parallel_time <= sequential_time * 2, 
                "Parallel should not be more than 2x slower");
    }
}
```

**Action Items:**
- [ ] Create performance benchmarks
- [ ] Measure speedup on different core counts
- [ ] Verify parallel version is faster (or at least not significantly slower)

---

## Code Structure

### Modified Files

1. **`src/minimax/player.rs`**
   - Add `parallel_threads` field to `MinimaxPlayer`
   - Add atomic f64 helper functions
   - Add `minimax_search_parallel` method
   - Refactor `alphabeta` to use shared core logic
   - Update `choose_move` to use parallel search when enabled
   - Add builder methods for thread configuration

### New Dependencies

- **None** - Uses existing `num_cpus` crate (already in `Cargo.toml`)
- Uses standard library: `std::thread`, `std::sync::atomic`, `std::sync::Arc`

### File Organization

```
src/
├── minimax/
│   ├── mod.rs          # (unchanged)
│   ├── player.rs       # (modified) Add parallel search
│   └── evaluator.rs    # (unchanged)
└── ...
```

---

## Testing Strategy

### Unit Tests

1. **Parallel search functionality**
   - Test with different thread counts
   - Test with single thread (should use sequential)
   - Test auto-detection of CPU cores
   - Test atomic f64 operations

2. **Correctness**
   - Parallel search returns valid moves
   - Parallel and sequential both work
   - Edge cases (empty moves, single move, etc.)

3. **Configuration**
   - Thread count builder methods
   - Default values
   - Thread configuration persistence

### Integration Tests

1. **Performance**
   - Measure speedup vs sequential
   - Verify parallel is faster (or acceptable overhead)
   - Test on different core counts

2. **Gameplay**
   - Full games with parallel minimax
   - Multiple difficulty levels
   - Time limits work correctly

### Test Checklist

- [ ] Parallel search chooses valid moves
- [ ] Atomic f64 operations work correctly
- [ ] Thread configuration methods work
- [ ] Single thread falls back to sequential
- [ ] Auto-detection works
- [ ] Performance improvement is measurable
- [ ] Full games complete successfully
- [ ] Time limits are respected in parallel mode

---

## Performance Considerations

### Expected Performance

| Cores | Expected Speedup | Notes |
|-------|-----------------|-------|
| 1     | 1.0x (baseline) | Sequential search |
| 2     | 1.5-1.8x       | Moderate improvement |
| 4     | 2.5-3.2x       | Good improvement |
| 8     | 4.0-5.5x       | Strong improvement |
| 16+   | 6.0-8.0x       | Diminishing returns |

### Factors Affecting Performance

1. **Number of Root Moves:**
   - More moves = better parallelization
   - Othello typically has 4-10 root moves (good for parallelization)
   - Few moves (2-3) = little benefit, may even be slower

2. **Move Ordering:**
   - Better move ordering = more pruning = less parallel benefit
   - Trade-off: Better moves found faster vs. better parallelization

3. **Search Depth:**
   - Deeper searches = more computation per move = better parallel efficiency
   - Shallow searches = more overhead from thread creation

4. **Synchronization Overhead:**
   - Atomic operations have overhead
   - Less efficient pruning reduces some of the parallel benefit

### Optimization Tips

1. **Only parallelize when beneficial:**
   - Don't parallelize if moves <= 2
   - Don't parallelize if threads <= 1
   - Consider minimum depth threshold

2. **Move distribution:**
   - Distribute moves in chunks (not one per thread)
   - Better load balancing
   - Reduces thread creation overhead

3. **Shared alpha updates:**
   - Use atomic operations efficiently
   - Only update when score improves
   - Read alpha before each move evaluation

4. **Time limits:**
   - Each thread checks time limit independently
   - May waste some computation, but ensures responsiveness

---

## Future Enhancements

### 1. Principal Variation Splitting (PVS)

Implement PVS for better parallel efficiency:
- Evaluate first move (PV) sequentially
- Evaluate remaining moves in parallel
- Better pruning efficiency
- Requires iterative deepening

### 2. Dynamic Thread Adjustment

Adjust number of threads based on:
- Number of root moves
- Available CPU cores
- Search depth
- Time constraints

### 3. Work Stealing

Implement work-stealing queue for better load balancing:
- Threads can steal moves from other threads when done
- More efficient use of threads
- More complex implementation

### 4. Transposition Table with Parallel Access

Add thread-safe transposition table:
- Share evaluated positions between threads
- Requires careful synchronization
- Significant speedup for repeated positions

### 5. Iterative Deepening with Parallelization

Combine iterative deepening with parallelization:
- Search depth 1, 2, 3, ... in parallel
- Use previous depth results for move ordering
- Better time management

### 6. Hybrid Sequential/Parallel

Use sequential for first few moves (better pruning), parallel for rest:
- Best of both worlds
- More complex implementation

---

## Common Pitfalls and Solutions

### Issue: Parallel Search is Slower Than Sequential

**Problem:** Parallel version takes longer than sequential, especially with few moves.

**Causes:**
- Thread creation overhead
- Too few moves to parallelize
- Synchronization overhead

**Solutions:**
- Only parallelize when moves > 2 and threads > 1
- Use chunk-based distribution (not one move per thread)
- Consider minimum depth threshold

### Issue: Incorrect Moves or Scores

**Problem:** Parallel search returns different (worse) moves than sequential.

**Causes:**
- Alpha-beta bounds not shared correctly
- Race conditions in alpha updates
- Incorrect atomic operations

**Solutions:**
- Verify atomic f64 operations are correct
- Ensure alpha is updated atomically
- Test consistency between parallel and sequential

### Issue: Deadlocks or Hangs

**Problem:** Parallel search hangs or deadlocks.

**Causes:**
- Infinite loops in atomic operations
- Threads waiting on each other
- Incorrect synchronization

**Solutions:**
- Review atomic operation logic
- Ensure no circular dependencies
- Add timeouts to prevent infinite hangs

### Issue: Poor Speedup

**Problem:** Parallel search only provides 1.2-1.5x speedup instead of expected 2-3x.

**Causes:**
- Too few moves at root
- High synchronization overhead
- Reduced pruning efficiency

**Solutions:**
- This is expected for minimax (less parallel-friendly than MCTS)
- Consider PVS for better efficiency
- Verify implementation is correct
- Accept that minimax has limited parallelization potential

---

## Implementation Timeline

| Phase | Estimated Time | Cumulative Time |
|-------|---------------|-----------------|
| Phase 1: Dependency Setup | 15 minutes | 15 minutes |
| Phase 2: Core Parallel Implementation | 4-5 hours | 4.5-5.5 hours |
| Phase 3: Refactoring for Code Reuse | 2-3 hours | 6.5-8.5 hours |
| Phase 4: Integration | 1 hour | 7.5-9.5 hours |
| Phase 5: Testing | 2-3 hours | 9.5-12.5 hours |

**Total Estimated Time:** 9.5-12.5 hours (approximately 1.5-2 days of work)

---

## Revision History

- **v1.0** (Initial): Complete Parallel Minimax implementation document created

---

## References

1. **Parallel Minimax Algorithms:**
   - "Parallel Algorithms for Game Tree Search" by Feldmann et al.
   - "A Parallel Minimax Algorithm" research papers

2. **Alpha-Beta Pruning:**
   - Wikipedia: Alpha-Beta pruning
   - "Game Tree Search" papers and tutorials

3. **Rust Concurrency:**
   - Rust Book: Concurrency chapter
   - `std::sync::atomic` documentation
   - `std::thread` documentation

4. **Atomic Operations:**
   - Rust `AtomicF64` documentation
   - `std::sync::atomic::AtomicF64` API reference

---

## Appendix: Alternative Implementation Using Mutex

If atomic operations prove problematic, an alternative approach using `Mutex<f64>`:

```rust
// Alternative: Use Mutex instead of AtomicF64
let shared_alpha = Arc::new(Mutex::new(f64::NEG_INFINITY));

// In threads:
let current_alpha = *shared_alpha.lock().unwrap();
// ... evaluate move ...
{
    let mut alpha = shared_alpha.lock().unwrap();
    *alpha = (*alpha).max(score);
}
```

**Trade-offs:**
- ✅ Simpler implementation (no compare-and-swap loop)
- ✅ Direct f64 operations
- ❌ Potentially higher overhead (lock contention)
- ❌ Risk of deadlocks if not careful

The atomic approach is preferred for better performance (lock-free), but mutex is a valid fallback if needed.

---

## Appendix: Complete Code Structure Summary

### Key Methods Added/Modified

1. **`atomic_max_f64`**: Atomic maximum operation for f64 values
3. **`minimax_search_parallel`**: Main parallel search method
4. **`alphabeta_core`**: Shared core logic (refactored from `alphabeta`)
5. **`with_parallel_threads` / `set_parallel_threads` / `parallel_threads`**: Thread configuration

### Thread Safety

- **Game state**: Each thread clones game state (no shared mutable state)
- **Alpha bounds**: Shared via `Arc<AtomicF64>` (thread-safe)
- **Results**: Collected via thread handles (thread-safe)
- **No locks needed** beyond atomic operations

### Performance Characteristics

- **Best case**: 4-8x speedup with 8-16 cores and many root moves
- **Typical case**: 2-4x speedup with 4-8 cores
- **Worst case**: Minimal or no speedup with few moves or single core


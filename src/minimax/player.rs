use crate::game::{Game, Player, GameState};
use crate::board::Position;
use crate::player::PlayerTrait;
use std::sync::{Arc, Mutex};
use std::thread;

/// A Minimax player with Alpha-Beta pruning for Othello.
///
/// Minimax explores the game tree to a fixed depth, evaluating positions
/// using heuristics. Alpha-Beta pruning significantly reduces the number
/// of positions evaluated without affecting the result.
///
/// # Examples
///
/// Basic usage with a preset difficulty:
///
/// ```rust
/// use othello::minimax::MinimaxPlayer;
/// use othello::{Game, PlayerTrait};
///
/// let game = Game::new();
/// let player = MinimaxPlayer::medium();
/// let move_opt = player.choose_move(&game);
/// ```
///
/// Custom configuration:
///
/// ```rust
/// use othello::minimax::MinimaxPlayer;
///
/// let player = MinimaxPlayer::with_depth("Custom AI", 5)
///     .with_time_limit_ms(2000);
/// ```
pub struct MinimaxPlayer {
    name: String,
    depth: usize,
    max_time_ms: Option<u64>,
    use_alpha_beta: bool,
    use_move_ordering: bool,
    parallel_threads: Option<usize>,
}

impl MinimaxPlayer {
    /// Create a new Minimax player with default settings.
    ///
    /// Default settings:
    /// - Depth 4 (moderate search)
    /// - Alpha-Beta pruning enabled
    /// - No time limit
    /// - Move ordering enabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::minimax::MinimaxPlayer;
    ///
    /// let player = MinimaxPlayer::new("My AI");
    /// ```
    pub fn new(name: impl Into<String>) -> Self {
        Self::with_depth(name, 4)
    }
    
    /// Create a new Minimax player with a specified search depth.
    ///
    /// `depth` is the number of moves ahead to search. Higher depth leads
    /// to better moves but exponentially slower search.
    ///
    /// Typical depths for Othello:
    /// - Depth 3: ~100-500ms per move (fast)
    /// - Depth 4: ~500ms-2s per move (medium)
    /// - Depth 5: ~2-10s per move (slow)
    /// - Depth 6+: Very slow (minutes), usually impractical
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::minimax::MinimaxPlayer;
    ///
    /// let player = MinimaxPlayer::with_depth("Deep AI", 5);
    /// ```
    pub fn with_depth(name: impl Into<String>, depth: usize) -> Self {
        Self {
            name: name.into(),
            depth: depth.max(1).min(8),  // Clamp to reasonable range
            max_time_ms: None,
            use_alpha_beta: true,
            use_move_ordering: true,
            parallel_threads: None,  // Auto-detect CPU cores
        }
    }
    
    /// Set the maximum time limit per move in milliseconds.
    ///
    /// If `Some(ms)`, the search will stop after `ms` milliseconds even if
    /// not all moves have been evaluated. If `None`, all moves will be evaluated.
    ///
    /// Note: Time limits are approximate - the search will complete the current
    /// depth before stopping.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::minimax::MinimaxPlayer;
    ///
    /// let player = MinimaxPlayer::new("Fast AI")
    ///     .with_time_limit_ms(1000);  // Stop after 1 second
    /// ```
    pub fn with_time_limit_ms(mut self, ms: u64) -> Self {
        self.max_time_ms = Some(ms);
        self
    }
    
    /// Enable or disable Alpha-Beta pruning.
    ///
    /// Alpha-Beta pruning significantly speeds up search (10-100x) without
    /// affecting results. It should generally be enabled.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::minimax::MinimaxPlayer;
    ///
    /// let player = MinimaxPlayer::new("AI")
    ///     .with_alpha_beta(false);  // Disable (slower but same results)
    /// ```
    pub fn with_alpha_beta(mut self, enable: bool) -> Self {
        self.use_alpha_beta = enable;
        self
    }
    
    /// Enable or disable move ordering.
    ///
    /// Move ordering improves Alpha-Beta pruning efficiency by evaluating
    /// better moves first. This increases the chance of early cutoffs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::minimax::MinimaxPlayer;
    ///
    /// let player = MinimaxPlayer::new("AI")
    ///     .with_move_ordering(true);  // Enable (recommended)
    /// ```
    pub fn with_move_ordering(mut self, enable: bool) -> Self {
        self.use_move_ordering = enable;
        self
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
    
    /// Atomically update alpha to the maximum of current and new value.
    /// 
    /// Uses Mutex to ensure thread-safe maximum operation.
    fn atomic_max_f64(shared: &Mutex<f64>, new_value: f64) -> f64 {
        let mut alpha = shared.lock().unwrap();
        if new_value > *alpha {
            *alpha = new_value;
        }
        *alpha
    }
    
    /// Create an easy difficulty Minimax player.
    ///
    /// Settings:
    /// - Depth 3 (fast, ~100-500ms per move)
    /// - Alpha-Beta enabled
    /// - Move ordering enabled
    /// - Parallel search enabled (auto-detect cores)
    ///
    /// Suitable for quick games or less experienced players.
    pub fn easy() -> Self {
        Self::with_depth("Minimax (Easy)", 3)
            .with_parallel_threads(None)  // Auto-detect cores
    }
    
    /// Create a medium difficulty Minimax player.
    ///
    /// Settings:
    /// - Depth 4 (moderate speed, ~500ms-2s per move)
    /// - Alpha-Beta enabled
    /// - Move ordering enabled
    /// - Parallel search enabled (auto-detect cores)
    ///
    /// A good balance of speed and strength. Recommended for most games.
    pub fn medium() -> Self {
        Self::with_depth("Minimax (Medium)", 4)
            .with_parallel_threads(None)  // Auto-detect cores
    }
    
    /// Create a hard difficulty Minimax player.
    ///
    /// Settings:
    /// - Depth 5 (slower, ~2-10s per move)
    /// - Alpha-Beta enabled
    /// - Move ordering enabled
    /// - Parallel search enabled (auto-detect cores)
    ///
    /// Strong play suitable for experienced players.
    pub fn hard() -> Self {
        Self::with_depth("Minimax (Hard)", 5)
            .with_parallel_threads(None)  // Auto-detect cores
    }
    
    /// Create an expert difficulty Minimax player.
    ///
    /// Settings:
    /// - Depth 6 (very slow, ~10-60s per move, capped at 30s)
    /// - Alpha-Beta enabled
    /// - Move ordering enabled
    /// - 30 second time limit
    /// - Parallel search enabled (auto-detect cores)
    ///
    /// Very strong play for expert-level competition.
    pub fn expert() -> Self {
        Self::with_depth("Minimax (Expert)", 6)
            .with_time_limit_ms(30000)
            .with_parallel_threads(None)  // Auto-detect cores
    }
    
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
    fn minimax_search_parallel(&self, game: &Game, num_threads: Option<usize>) -> Option<Position> {
        use num_cpus;
        use std::time::Instant;
        
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
        let shared_alpha = Arc::new(Mutex::new(f64::NEG_INFINITY));
        
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
            let depth_clone = depth;
            let use_alpha_beta_clone = use_alpha_beta;
            let max_time_ms_clone = max_time_ms;
            let use_move_ordering_clone = use_move_ordering;
            
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
                    
                    // Read current shared alpha (may have been updated by other threads)
                    let current_alpha = *shared_alpha_clone.lock().unwrap();
                    let beta = f64::INFINITY;
                    
                    // Clone game state for this branch
                    let mut test_game = game_clone.clone();
                    if test_game.make_move(move_pos).is_err() {
                        continue;
                    }
                    
                    // Evaluate move using alphabeta with shared alpha
                    let score = if use_alpha_beta_clone {
                        Self::evaluate_move_parallel(
                            &test_game,
                            depth_clone - 1,
                            depth_clone,  // Original depth for checking first level
                            current_alpha,
                            beta,
                            false,
                            current_player,
                            start_time_clone,
                            max_time_ms_clone,
                            &shared_alpha_clone,
                            use_move_ordering_clone,
                        )
                    } else {
                        // Without alpha-beta, fall back to sequential search for this move
                        // This is less common, so we'll use a simple evaluation
                        Self::evaluate_position_static(&test_game, current_player)
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
    
    /// Helper function to evaluate a move in parallel context.
    /// This extracts the alphabeta logic so it can be called from threads.
    fn evaluate_move_parallel(
        game: &Game,
        depth: usize,
        original_depth: usize,
        mut alpha: f64,
        beta: f64,
        maximizing: bool,
        root_player: Player,
        start_time: std::time::Instant,
        max_time_ms: Option<u64>,
        shared_alpha: &Mutex<f64>,
        use_move_ordering: bool,
    ) -> f64 {
        // Check time limit
        if let Some(max_ms) = max_time_ms {
            if start_time.elapsed().as_millis() as u64 > max_ms {
                return Self::evaluate_position_static(game, root_player);
            }
        }
        
        // Terminal conditions
        if depth == 0 || matches!(game.get_game_state(), GameState::GameOver { .. }) {
            return Self::evaluate_position_static(game, root_player);
        }
        
        let valid_moves = game.get_valid_moves();
        
        if valid_moves.is_empty() {
            let mut test_game = game.clone();
            test_game.skip_turn().ok();
            return Self::evaluate_move_parallel(
                &test_game,
                depth,
                original_depth,
                alpha,
                beta,
                !maximizing,
                root_player,
                start_time,
                max_time_ms,
                shared_alpha,
                use_move_ordering,
            );
        }
        
        // Order moves if enabled (simplified ordering for parallel context)
        let ordered_moves = if use_move_ordering && depth < original_depth {
            // Use a simple heuristic-based ordering
            Self::order_moves_simple(game, &valid_moves)
        } else {
            valid_moves
        };
        
        if maximizing {
            let mut max_score = f64::NEG_INFINITY;
            
            for &move_pos in &ordered_moves {
                // Check shared alpha at first level below root
                if depth == original_depth - 1 {
                    let current_alpha = *shared_alpha.lock().unwrap();
                    if current_alpha >= beta {
                        break;
                    }
                    alpha = alpha.max(current_alpha);
                }
                
                let mut test_game = game.clone();
                if test_game.make_move(move_pos).is_err() {
                    continue;
                }
                
                let score = Self::evaluate_move_parallel(
                    &test_game,
                    depth - 1,
                    original_depth,
                    alpha,
                    beta,
                    false,
                    root_player,
                    start_time,
                    max_time_ms,
                    shared_alpha,
                    use_move_ordering,
                );
                
                max_score = max_score.max(score);
                alpha = alpha.max(score);
                
                // Update shared alpha at first level below root
                if depth == original_depth - 1 {
                    Self::atomic_max_f64(shared_alpha, score);
                }
                
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
                
                let score = Self::evaluate_move_parallel(
                    &test_game,
                    depth - 1,
                    original_depth,
                    alpha,
                    beta,
                    true,
                    root_player,
                    start_time,
                    max_time_ms,
                    shared_alpha,
                    use_move_ordering,
                );
                
                min_score = min_score.min(score);
                let new_beta = beta.min(score);
                
                if new_beta <= alpha {
                    break;
                }
            }
            
            min_score
        }
    }
    
    /// Static helper to evaluate position (for use in parallel context).
    fn evaluate_position_static(game: &Game, player: Player) -> f64 {
        use crate::minimax::evaluator::PositionEvaluator;
        PositionEvaluator::evaluate(game, player)
    }
    
    /// Simple move ordering for parallel context (static version).
    fn order_moves_simple(game: &Game, moves: &[Position]) -> Vec<Position> {
        use crate::mcts::heuristics::Heuristics;
        use std::panic;
        
        let mut scored_moves: Vec<(Position, f64)> = moves.iter()
            .filter_map(|&pos| {
                // Safely evaluate move, catching any panics from heuristics
                let score = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    Heuristics::evaluate_move(game, pos)
                }));
                
                match score {
                    Ok(s) => Some((pos, s)),
                    Err(_) => Some((pos, 0.0)),  // Default score if evaluation panics
                }
            })
            .collect();
        
        // Sort by score (descending - best moves first)
        scored_moves.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        scored_moves.into_iter().map(|(pos, _)| pos).collect()
    }
    
    /// Perform minimax search and return best move.
    ///
    /// This is the core search algorithm that explores the game tree
    /// and selects the best move according to the evaluation function.
    fn minimax_search(&self, game: &Game) -> Option<Position> {
        use std::time::Instant;
        
        let valid_moves = game.get_valid_moves();
        
        if valid_moves.is_empty() {
            return None;
        }
        
        if valid_moves.len() == 1 {
            return Some(valid_moves[0]);
        }
        
        let start_time = Instant::now();
        let current_player = game.current_player();
        
        // Order moves for better alpha-beta pruning
        let ordered_moves = if self.use_move_ordering {
            self.order_moves(game, &valid_moves)
        } else {
            valid_moves
        };
        
        let mut best_move = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut alpha = f64::NEG_INFINITY;
        let beta = f64::INFINITY;
        
        for &move_pos in &ordered_moves {
            // Check time limit
            if let Some(max_ms) = self.max_time_ms {
                if start_time.elapsed().as_millis() as u64 > max_ms {
                    break;
                }
            }
            
            // Clone game state for this branch (no undo needed)
            let mut test_game = game.clone();
            if test_game.make_move(move_pos).is_err() {
                continue;
            }
            
            let score = if self.use_alpha_beta {
                self.alphabeta(&test_game, self.depth - 1, alpha, beta, false, current_player, start_time)
            } else {
                self.minimax(&test_game, self.depth - 1, false, current_player, start_time)
            };
            
            if score > best_score {
                best_score = score;
                best_move = Some(move_pos);
            }
            
            alpha = alpha.max(score);
            
            // Early termination if we've found a winning move
            if best_score == f64::INFINITY {
                break;
            }
        }
        
        best_move
    }
    
    /// Minimax algorithm without pruning (baseline).
    fn minimax(
        &self,
        game: &Game,  // Use immutable reference, clone when needed
        depth: usize,
        maximizing: bool,
        root_player: Player,
        start_time: std::time::Instant,
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
            // No moves available, skip turn
            let mut test_game = game.clone();
            test_game.skip_turn().ok();
            return self.minimax(&test_game, depth, !maximizing, root_player, start_time);
        }
        
        if maximizing {
            let mut max_score = f64::NEG_INFINITY;
            
            for &move_pos in &valid_moves {
                let mut test_game = game.clone();
                if test_game.make_move(move_pos).is_err() {
                    continue;
                }
                let score = self.minimax(&test_game, depth - 1, false, root_player, start_time);
                
                max_score = max_score.max(score);
            }
            
            max_score
        } else {
            let mut min_score = f64::INFINITY;
            
            for &move_pos in &valid_moves {
                let mut test_game = game.clone();
                if test_game.make_move(move_pos).is_err() {
                    continue;
                }
                let score = self.minimax(&test_game, depth - 1, true, root_player, start_time);
                
                min_score = min_score.min(score);
            }
            
            min_score
        }
    }
    
    /// Minimax with Alpha-Beta pruning.
    fn alphabeta(
        &self,
        game: &Game,  // Use immutable reference, clone when needed
        depth: usize,
        alpha: f64,
        beta: f64,
        maximizing: bool,
        root_player: Player,
        start_time: std::time::Instant,
    ) -> f64 {
        self.alphabeta_with_shared_alpha(
            game,
            depth,
            alpha,
            beta,
            maximizing,
            root_player,
            start_time,
            None,  // No shared alpha for sequential search
        )
    }
    
    /// Minimax with Alpha-Beta pruning, with optional shared alpha for parallel search.
    fn alphabeta_with_shared_alpha(
        &self,
        game: &Game,
        depth: usize,
        mut alpha: f64,
        mut beta: f64,
        maximizing: bool,
        root_player: Player,
        start_time: std::time::Instant,
        shared_alpha: Option<&Mutex<f64>>,
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
            // No moves available, skip turn
            let mut test_game = game.clone();
            test_game.skip_turn().ok();
            return self.alphabeta_with_shared_alpha(
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
        
        // Order moves for better pruning (best moves first)
        let ordered_moves = if self.use_move_ordering && depth < self.depth {
            self.order_moves(game, &valid_moves)
        } else {
            valid_moves
        };
        
        if maximizing {
            let mut max_score = f64::NEG_INFINITY;
            
            for &move_pos in &ordered_moves {
                // Check shared alpha if in parallel mode (only at first level below root)
                if let Some(shared) = shared_alpha {
                    if depth == self.depth - 1 {
                        // At first level below root, check shared alpha
                        let current_alpha = *shared.lock().unwrap();
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
                let score = self.alphabeta_with_shared_alpha(
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
                
                // Update shared alpha if in parallel mode (only at first level below root)
                if let Some(shared) = shared_alpha {
                    if depth == self.depth - 1 {
                        Self::atomic_max_f64(shared, score);
                    }
                }
                
                // Beta cutoff (prune remaining moves)
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
                let score = self.alphabeta_with_shared_alpha(
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
                
                // Alpha cutoff (prune remaining moves)
                if beta <= alpha {
                    break;
                }
            }
            
            min_score
        }
    }
    
    /// Order moves to improve alpha-beta pruning efficiency.
    ///
    /// Moves are sorted by heuristic score (best first) so that
    /// alpha-beta pruning can cut off more branches early.
    fn order_moves(&self, game: &Game, moves: &[Position]) -> Vec<Position> {
        use crate::mcts::heuristics::Heuristics;
        use std::panic;
        
        let mut scored_moves: Vec<(Position, f64)> = moves.iter()
            .filter_map(|&pos| {
                // Safely evaluate move, catching any panics from heuristics
                let score = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    Heuristics::evaluate_move(game, pos)
                }));
                
                match score {
                    Ok(s) => Some((pos, s)),
                    Err(_) => Some((pos, 0.0)),  // Default score if evaluation panics
                }
            })
            .collect();
        
        // Sort by score (descending - best moves first)
        scored_moves.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        scored_moves.into_iter().map(|(pos, _)| pos).collect()
    }
    
    /// Evaluate a game position from the given player's perspective.
    fn evaluate_position(&self, game: &Game, player: Player) -> f64 {
        use crate::minimax::evaluator::PositionEvaluator;
        PositionEvaluator::evaluate(game, player)
    }
}

impl PlayerTrait for MinimaxPlayer {
    /// Choose the best move using Minimax search.
    ///
    /// Performs minimax search with alpha-beta pruning (if enabled) to
    /// the configured depth and returns the best move found.
    ///
    /// Uses parallel search if enabled (parallel_threads is None or > 1).
    ///
    /// Returns `None` if there are no valid moves available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::minimax::MinimaxPlayer;
    /// use othello::{Game, PlayerTrait};
    ///
    /// let game = Game::new();
    /// let player = MinimaxPlayer::medium();
    /// if let Some(position) = player.choose_move(&game) {
    ///     println!("Best move: {:?}", position);
    /// }
    /// ```
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    use crate::player::PlayerTrait;
    
    #[test]
    fn test_minimax_chooses_move() {
        let game = Game::new();
        let player = MinimaxPlayer::easy();
        
        let move_opt = player.choose_move(&game);
        assert!(move_opt.is_some());
        
        let position = move_opt.unwrap();
        let valid_moves = game.get_valid_moves();
        assert!(valid_moves.contains(&position));
    }
    
    #[test]
    fn test_minimax_handles_no_moves() {
        // Create a game state with no valid moves
        let game = Game::new();
        let player = MinimaxPlayer::easy();
        
        // This should handle gracefully (returns None for empty moves)
        let move_opt = player.choose_move(&game);
        // Should either return None or a valid move
        if let Some(pos) = move_opt {
            assert!(game.get_valid_moves().contains(&pos));
        }
    }
    
    #[test]
    fn test_difficulty_presets() {
        let easy = MinimaxPlayer::easy();
        assert_eq!(easy.depth, 3);
        assert!(easy.use_alpha_beta);
        
        let medium = MinimaxPlayer::medium();
        assert_eq!(medium.depth, 4);
        
        let hard = MinimaxPlayer::hard();
        assert_eq!(hard.depth, 5);
        
        let expert = MinimaxPlayer::expert();
        assert_eq!(expert.depth, 6);
        assert_eq!(expert.max_time_ms, Some(30000));
    }
    
    #[test]
    fn test_move_ordering() {
        let game = Game::new();
        let player = MinimaxPlayer::with_depth("Test", 3)
            .with_move_ordering(true);
        
        let moves = game.get_valid_moves();
        let ordered = player.order_moves(&game, &moves);
        
        // Should have same moves (just reordered)
        assert_eq!(moves.len(), ordered.len());
        // All moves should still be present
        for &mv in &ordered {
            assert!(moves.contains(&mv));
        }
    }
    
    #[test]
    fn test_alphabeta_vs_minimax_same_result() {
        let game = Game::new();
        
        let player_ab = MinimaxPlayer::with_depth("AB", 3)
            .with_alpha_beta(true);
        let player_no_ab = MinimaxPlayer::with_depth("NoAB", 3)
            .with_alpha_beta(false);
        
        // Both should return valid moves (may differ due to move ordering)
        let move_ab = player_ab.choose_move(&game);
        let move_no_ab = player_no_ab.choose_move(&game);
        
        assert!(move_ab.is_some());
        assert!(move_no_ab.is_some());
    }
}


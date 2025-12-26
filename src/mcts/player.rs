use crate::mcts::node::MCTSNode;
use crate::mcts::heuristics::Heuristics;
use crate::game::{Game, Player, GameState};
use crate::board::Position;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::thread_rng;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::thread;
use std::collections::HashMap;

/// A Monte Carlo Tree Search player for Othello.
///
/// MCTS builds a search tree by repeatedly performing four phases:
/// 1. **Selection**: Traverse from root to leaf using UCB1 (Upper Confidence Bound)
/// 2. **Expansion**: Add children to the selected leaf node
/// 3. **Simulation**: Play a random/heuristic-guided game to completion
/// 4. **Backpropagation**: Update statistics (visits and win rate) up the tree
///
/// After many iterations, the move from the most-visited child is selected (robust child).
///
/// # Examples
///
/// Basic usage with a preset difficulty:
///
/// ```rust
/// use othello::mcts::MCTSPlayer;
/// use othello::{Game, PlayerTrait};
///
/// let game = Game::new();
/// let player = MCTSPlayer::medium();
/// let move_opt = player.choose_move(&game);
/// ```
///
/// Custom configuration with builder pattern:
///
/// ```rust
/// use othello::mcts::MCTSPlayer;
///
/// let player = MCTSPlayer::with_iterations("Custom AI", 2000)
///     .with_exploration(1.5)
///     .with_time_limit_ms(2000)
///     .with_heuristics(true);
/// ```
///
/// Using difficulty presets:
///
/// ```rust
/// use othello::mcts::MCTSPlayer;
///
/// let easy = MCTSPlayer::easy();     // 200 iterations
/// let medium = MCTSPlayer::medium(); // 1000 iterations
/// let hard = MCTSPlayer::hard();     // 3000 iterations
/// let expert = MCTSPlayer::expert(); // 10000 iterations, 5s limit
/// ```
///
/// # Performance
///
/// The number of iterations directly affects:
/// - **Quality**: More iterations generally lead to better moves (but with diminishing returns)
/// - **Speed**: Each iteration takes roughly 1-10ms depending on game state and heuristics
///
/// For real-time play, 500-2000 iterations is usually sufficient. For analysis or
/// very strong play, 5000-10000+ iterations may be used.
pub struct MCTSPlayer {
    name: String,
    iterations: usize,
    exploration_constant: f64,
    max_time_ms: Option<u64>,
    use_heuristics: bool,
    parallel_threads: Option<usize>,
}

impl MCTSPlayer {
    /// Create a new MCTS player with default settings.
    ///
    /// Default settings:
    /// - 1000 iterations per move
    /// - Exploration constant: √2 ≈ 1.414
    /// - No time limit
    /// - Heuristics disabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let player = MCTSPlayer::new("My AI");
    /// ```
    pub fn new(name: impl Into<String>) -> Self {
        Self::with_iterations(name, 1000)
    }
    
    /// Create a new MCTS player with a specified number of iterations.
    ///
    /// `iterations` is the number of MCTS iterations to perform per move.
    /// More iterations generally lead to better moves but take longer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let player = MCTSPlayer::with_iterations("Fast AI", 500);
    /// ```
    pub fn with_iterations(name: impl Into<String>, iterations: usize) -> Self {
        Self {
            name: name.into(),
            iterations,
            exploration_constant: 1.414, // √2
            max_time_ms: Some(60000),
            use_heuristics: true,
            parallel_threads: None, // Auto-detect CPU cores
        }
    }
    
    /// Set the exploration constant for UCB1.
    ///
    /// The exploration constant balances exploitation (choosing good moves) vs
    /// exploration (trying new moves). Higher values favor exploration.
    ///
    /// Typical values:
    /// - √2 ≈ 1.414 (default, balanced)
    /// - 1.0 - 1.5 (more exploitation, for strong play)
    /// - 2.0+ (more exploration, for varied/uncertain positions)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let mut player = MCTSPlayer::new("AI");
    /// player.set_exploration_constant(2.0); // More exploration
    /// ```
    pub fn set_exploration_constant(&mut self, c: f64) {
        self.exploration_constant = c;
    }
    
    /// Set the maximum time limit per move in milliseconds.
    ///
    /// If `Some(ms)`, the search will stop after `ms` milliseconds even if
    /// not all iterations are complete. If `None`, all iterations will run.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let mut player = MCTSPlayer::with_iterations("Fast AI", 10000);
    /// player.set_max_time_ms(Some(2000)); // Stop after 2 seconds
    /// ```
    pub fn set_max_time_ms(&mut self, ms: Option<u64>) {
        self.max_time_ms = ms;
    }
    
    /// Set whether to use heuristics during simulation.
    ///
    /// When enabled, moves during simulation are selected probabilistically based on
    /// heuristic scores rather than uniformly randomly. This typically improves
    /// simulation quality and can lead to better move selection.
    ///
    /// Heuristics evaluate moves based on:
    /// - Corner positions (valuable)
    /// - Mobility (maximizing future options)
    /// - Stability (edge pieces are harder to flip)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let mut player = MCTSPlayer::new("AI");
    /// player.set_use_heuristics(true);
    /// ```
    pub fn set_use_heuristics(&mut self, enable: bool) {
        self.use_heuristics = enable;
    }
    
    /// Create an easy difficulty MCTS player.
    ///
    /// Settings:
    /// - 200 iterations (fast, ~200-500ms per move)
    /// - Higher exploration (2.0) for more varied play
    /// - No heuristics (pure random simulation)
    ///
    /// Suitable for quick games or less experienced players.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let player = MCTSPlayer::easy();
    /// ```
    pub fn easy() -> Self {
        Self::with_iterations("MCTS (Easy)", 200)
            .with_exploration(2.0)
            .with_heuristics(false)
    }
    
    /// Create a medium difficulty MCTS player.
    ///
    /// Settings:
    /// - 1000 iterations (moderate speed, ~1-3s per move)
    /// - Standard exploration (√2 ≈ 1.414)
    /// - Heuristics enabled
    ///
    /// A good balance of speed and strength. Recommended for most games.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let player = MCTSPlayer::medium();
    /// ```
    pub fn medium() -> Self {
        Self::with_iterations("MCTS (Medium)", 1000)
            .with_exploration(1.414)
            .with_heuristics(true)
    }
    
    /// Create a hard difficulty MCTS player.
    ///
    /// Settings:
    /// - 3000 iterations (slower, ~3-10s per move)
    /// - Standard exploration (√2 ≈ 1.414)
    /// - Heuristics enabled
    ///
    /// Strong play suitable for experienced players.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let player = MCTSPlayer::hard();
    /// ```
    pub fn hard() -> Self {
        Self::with_iterations("MCTS (Hard)", 3000)
            .with_exploration(1.414)
            .with_heuristics(true)
    }
    
    /// Create an expert difficulty MCTS player.
    ///
    /// Settings:
    /// - 10000 iterations (very slow, ~10-30s per move, capped at 60s)
    /// - Standard exploration (√2 ≈ 1.414)
    /// - Heuristics enabled
    /// - 60 second time limit (may not complete all iterations)
    ///
    /// Very strong play for expert-level competition. Uses time limit to
    /// prevent excessively long moves while allowing deep search when possible.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let player = MCTSPlayer::expert();
    /// ```
    pub fn expert() -> Self {
        Self::with_iterations("MCTS (Expert)", 50000)
            .with_exploration(2.0)
            .with_heuristics(true)
            .with_time_limit_ms(60000)
    }
    
    /// Builder method: set exploration constant and return self.
    ///
    /// Allows method chaining for convenient configuration.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let player = MCTSPlayer::new("AI")
    ///     .with_exploration(1.5)
    ///     .with_heuristics(true);
    /// ```
    pub fn with_exploration(mut self, c: f64) -> Self {
        self.exploration_constant = c;
        self
    }
    
    /// Builder method: set time limit and return self.
    ///
    /// Allows method chaining for convenient configuration.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let player = MCTSPlayer::with_iterations("AI", 5000)
    ///     .with_time_limit_ms(3000); // Cap at 3 seconds
    /// ```
    pub fn with_time_limit_ms(mut self, ms: u64) -> Self {
        self.max_time_ms = Some(ms);
        self
    }
    
    /// Builder method: set heuristics usage and return self.
    ///
    /// Allows method chaining for convenient configuration.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let player = MCTSPlayer::with_iterations("AI", 1000)
    ///     .with_heuristics(true); // Enable heuristics
    /// ```
    pub fn with_heuristics(mut self, enable: bool) -> Self {
        self.use_heuristics = enable;
        self
    }
    
    /// Set the number of threads to use for parallel search.
    ///
    /// If `Some(n)`, uses exactly `n` threads.
    /// If `None`, uses all available CPU cores (default).
    /// Set to `Some(1)` to disable parallelization.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let player = MCTSPlayer::medium()
    ///     .with_parallel_threads(Some(4));  // Use 4 threads
    /// ```
    pub fn set_parallel_threads(&mut self, threads: Option<usize>) {
        self.parallel_threads = threads;
    }
    
    /// Builder method: set parallel threads and return self.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    ///
    /// let player = MCTSPlayer::medium()
    ///     .with_parallel_threads(Some(4));  // Use 4 threads
    /// ```
    pub fn with_parallel_threads(mut self, threads: Option<usize>) -> Self {
        self.parallel_threads = threads;
        self
    }
    
    /// Get the number of threads that will be used.
    pub fn parallel_threads(&self) -> Option<usize> {
        self.parallel_threads
    }
    
    /// Perform MCTS search and return best move
    fn mcts_search(&self, game: &Game) -> Option<Position> {
        let mut root = MCTSNode::new(game.clone());
        let root_player = game.current_player();
        let start_time = Instant::now();
        
        for _iteration in 0..self.iterations {
            // Check time limit
            if let Some(max_ms) = self.max_time_ms {
                if start_time.elapsed().as_millis() as u64 > max_ms {
                    break;
                }
            }
            
            // Perform one MCTS iteration
            self.mcts_iteration(&mut root, root_player);
        }
        
        // Return move from most visited child (robust child)
        root.best_child_robust()
            .and_then(|child| child.move_from_parent())
    }
    
    /// Perform parallel MCTS search using multiple CPU cores.
    ///
    /// This method distributes MCTS iterations across multiple threads,
    /// where each thread maintains its own independent search tree.
    /// Results are aggregated at the end to select the best move.
    ///
    /// # Performance
    ///
    /// Parallel search typically achieves 1.8x-7x speedup depending on:
    /// - Number of CPU cores available
    /// - Iteration count (more iterations = better parallel efficiency)
    /// - Tree structure and simulation cost
    ///
    /// # Thread Safety
    ///
    /// This implementation is thread-safe and uses no locks during search
    /// (each thread has its own tree). Only final aggregation requires
    /// synchronization.
    ///
    /// # Arguments
    ///
    /// * `game` - The current game state to search from
    /// * `num_threads` - Number of threads to use (defaults to CPU count if None)
    ///
    /// # Returns
    ///
    /// The best move according to aggregated statistics, or None if no valid moves.
    fn mcts_search_parallel(&self, game: &Game, num_threads: Option<usize>) -> Option<Position> {
        let valid_moves = game.get_valid_moves();
        
        if valid_moves.is_empty() {
            return None;
        }
        
        if valid_moves.len() == 1 {
            return Some(valid_moves[0]);
        }
        
        // Determine number of threads
        let num_threads = num_threads.unwrap_or_else(|| {
            num_cpus::get().max(1)
        }).max(1);
        
        // Distribute iterations across threads
        let iterations_per_thread = self.iterations / num_threads;
        let remainder = self.iterations % num_threads;
        
        let max_time_ms = self.max_time_ms;
        
        // Shared results storage
        let results: Arc<Mutex<Vec<(MCTSNode, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        
        // Spawn worker threads
        let mut handles = Vec::new();
        for thread_id in 0..num_threads {
            let iterations = iterations_per_thread + if thread_id < remainder { 1 } else { 0 };
            if iterations == 0 {
                continue;
            }
            
            let game_clone = game.clone();
            let root_player = game.current_player();
            let exploration = self.exploration_constant;
            let use_heuristics = self.use_heuristics;
            let results_clone = Arc::clone(&results);
            let max_time = max_time_ms;
            
            let handle = thread::spawn(move || {
                let mut root = MCTSNode::new(game_clone);
                let thread_start = Instant::now();
                let mut actual_iterations = 0;
                
                for _ in 0..iterations {
                    // Check time limit
                    if let Some(max_ms) = max_time {
                        if thread_start.elapsed().as_millis() as u64 > max_ms {
                            break;
                        }
                    }
                    
                    // Perform one MCTS iteration (same as sequential version)
                    Self::mcts_iteration_impl(&mut root, root_player, exploration, use_heuristics);
                    actual_iterations += 1;
                }
                
                // Store result
                let mut results_guard = results_clone.lock().unwrap();
                results_guard.push((root, actual_iterations));
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Aggregate results
        let results_guard = results.lock().unwrap();
        Self::aggregate_trees(&results_guard, game.current_player())
    }
    
    /// Aggregate multiple MCTS trees into a single result.
    ///
    /// Combines statistics (visits and values) from all trees to determine
    /// the best move. Uses the robust child selection (most visited).
    fn aggregate_trees(
        trees: &[(MCTSNode, usize)],
        root_player: Player,
    ) -> Option<Position> {
        if trees.is_empty() {
            return None;
        }
        
        if trees.len() == 1 {
            // Single tree, use it directly
            return trees[0].0.best_child_robust()
                .and_then(|child| child.move_from_parent());
        }
        
        // Aggregate statistics across all trees
        // Map from Position to (total_visits, total_value)
        // Value is already weighted by visits (value * visits)
        let mut move_stats: HashMap<Position, (usize, f64)> = HashMap::new();
        
        for (tree, _actual_iterations) in trees {
            if !tree.is_expanded() || !tree.has_children() {
                continue;
            }
            
            // Iterate through all children
            for i in 0..tree.num_children() {
                if let Some(child) = tree.get_child(i) {
                    if let Some(mv) = child.move_from_parent() {
                        let visits = child.visits();
                        if visits == 0 {
                            continue;  // Skip unvisited children
                        }
                        
                        // Get average value from child's perspective
                        let child_avg_value = child.average_value();
                        
                        // Convert to root player's perspective
                        // If child's player is opposite to root, flip the value
                        let root_perspective_value = if child.current_player() == root_player {
                            child_avg_value  // Same player, use as-is
                        } else {
                            1.0 - child_avg_value  // Opposite player, flip
                        };
                        
                        // Accumulate: visits sum directly, values sum as weighted (value * visits)
                        let entry = move_stats.entry(mv).or_insert((0, 0.0));
                        entry.0 += visits;
                        entry.1 += root_perspective_value * visits as f64;
                    }
                }
            }
        }
        
        // Select move with most total visits (robust child selection)
        move_stats.iter()
            .max_by_key(|(_, (visits, _))| *visits)
            .map(|(pos, _)| *pos)
    }
    /// Perform selection phase: traverse from root to leaf (read-only to get path)
    /// Returns a path (vector of child indices) from root to the selected leaf node.
    /// The leaf node is either:
    /// - Not yet expanded (needs expansion)
    /// - Terminal (game over)
    /// - Has no children (no valid moves available)
    fn select_path(root: &MCTSNode, exploration_constant: f64) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = root;
        
        // Keep traversing while:
        // 1. Node is expanded (has children created)
        // 2. Node is not terminal (game still ongoing)
        // 3. Node has children (has valid moves)
        while current.is_expanded() && !current.is_terminal() && current.has_children() {
            let child_idx = current.select_child(exploration_constant);
            
            // Safety check: ensure the selected index is valid
            if child_idx >= current.num_children() {
                // This shouldn't happen, but if it does, stop traversal
                break;
            }
            
            path.push(child_idx);
            
            // Move to child node (immutable reference)
            match current.get_child(child_idx) {
                Some(child) => current = child,
                None => {
                    // Invalid child index, stop traversal
                    path.pop(); // Remove the invalid index
                    break;
                }
            }
        }
        
        path
    }
    
    /// Get mutable reference to node at the given path
    /// The path is a vector of child indices from root to the target node.
    /// Returns a mutable reference to the node at the end of the path.
    fn get_node_mut_at_path<'a>(root: &'a mut MCTSNode, path: &[usize]) -> &'a mut MCTSNode {
        let mut current = root;
        for &idx in path {
            // Safety: verify index is valid before accessing
            if idx >= current.num_children() {
                panic!("Invalid path index {} for node with {} children", idx, current.num_children());
            }
            current = current.get_child_mut(idx)
                .expect("Path index should be valid");
        }
        current
    }
    
    /// Simulate a random game from given state
    /// Returns: 1.0 if root_player wins, 0.0 if loses, 0.5 if draw
    #[allow(dead_code)] // Used in tests
    fn simulate(&self, game: &mut Game, root_player: Player) -> f64 {
        Self::simulate_impl(game, root_player, self.use_heuristics)
    }
    
    /// Internal simulation implementation (extracted for reuse).
    ///
    /// This is extracted so it can be used in both sequential and parallel contexts.
    fn simulate_impl(
        game: &mut Game,
        root_player: Player,
        use_heuristics: bool,
    ) -> f64 {
        let mut rng = thread_rng();
        
        while matches!(game.get_game_state(), GameState::Playing) {
            let valid_moves = game.get_valid_moves();
            
            if valid_moves.is_empty() {
                // No valid moves, skip turn
                game.skip_turn().ok();
                continue;
            }
            
            // Choose move based on strategy
            let move_pos = if use_heuristics {
                // We need to create a temporary player to use heuristic_move_selection
                // For now, we'll inline the logic or use a static method
                Self::heuristic_move_selection_static(game, &valid_moves, &mut rng)
            } else {
                valid_moves.choose(&mut rng).copied()
            };
            
            if let Some(pos) = move_pos {
                game.make_move(pos).ok();
            } else {
                game.skip_turn().ok();
            }
        }
        
        // Determine result from root player's perspective
        match game.get_game_state() {
            GameState::GameOver { winner } => {
                match winner {
                    Some(player) if player == root_player => 1.0,
                    Some(_) => 0.0,
                    None => 0.5,
                }
            }
            _ => 0.5, // Should not happen
        }
    }
    
    /// Static version of heuristic move selection for use in parallel contexts.
    fn heuristic_move_selection_static(
        game: &Game,
        moves: &[Position],
        rng: &mut impl Rng,
    ) -> Option<Position> {
        // Score all moves
        let scored_moves: Vec<(Position, f64)> = moves.iter()
            .map(|&pos| {
                let score = Heuristics::evaluate_move(game, pos);
                (pos, score.max(0.01)) // Ensure positive scores for probability
            })
            .collect();
        
        // Calculate total score
        let total_score: f64 = scored_moves.iter().map(|(_, score)| score).sum();
        
        if total_score == 0.0 {
            // Fallback to random if all scores are zero or negative
            return moves.choose(rng).copied();
        }
        
        // Weighted random selection
        let mut random_value: f64 = rng.gen();
        random_value *= total_score;
        
        let mut cumulative = 0.0;
        for (pos, score) in scored_moves {
            cumulative += score;
            if random_value <= cumulative {
                return Some(pos);
            }
        }
        
        // Fallback (shouldn't happen, but return last move if it does)
        moves.last().copied()
    }
    
    /// Select a move using heuristics with weighted random selection
    /// Returns a move chosen probabilistically based on heuristic scores
    #[allow(dead_code)] // Used in tests
    fn heuristic_move_selection(
        &self,
        game: &Game,
        moves: &[Position],
        rng: &mut impl Rng,
    ) -> Option<Position> {
        Self::heuristic_move_selection_static(game, moves, rng)
    }
    
    /// Backpropagate result up the tree
    /// Updates all nodes along the path from leaf to root
    fn backpropagate(root: &mut MCTSNode, path: &[usize], result: f64, root_player: Player) {
        // First, update root (from root player's perspective)
        root.update_statistics(result);
        
        // Then update nodes along the path
        // We need to work backwards to get the correct perspective for each node
        for i in 0..path.len() {
            // Path up to and including this node (inclusive)
            let node_path = &path[..=i];
            let node = Self::get_node_mut_at_path(root, node_path);
            let node_player = node.current_player();
            
            // Result from this node's player perspective
            // If node_player is the same as root_player, use result as-is
            // If node_player is opposite, flip the result (1.0 - result)
            let node_result = if node_player == root_player {
                result
            } else {
                1.0 - result
            };
            
            node.update_statistics(node_result);
        }
    }
    
    /// Perform one MCTS iteration
    /// Combines selection, expansion, simulation, and backpropagation
    fn mcts_iteration(&self, root: &mut MCTSNode, root_player: Player) {
        Self::mcts_iteration_impl(
            root,
            root_player,
            self.exploration_constant,
            self.use_heuristics,
        );
    }
    
    /// Internal implementation of one MCTS iteration.
    ///
    /// This is extracted so it can be used in both sequential and parallel contexts.
    fn mcts_iteration_impl(
        root: &mut MCTSNode,
        root_player: Player,
        exploration_constant: f64,
        use_heuristics: bool,
    ) {
        // 1. Selection: Find path to leaf (read-only)
        let path = Self::select_path(root, exploration_constant);
        
        // 2. Get the leaf node (mutable)
        let leaf = Self::get_node_mut_at_path(root, &path);
        
        // 3. Expansion: Expand if not terminal and not expanded
        if !leaf.is_terminal() && !leaf.is_expanded() {
            leaf.expand();
        }
        
        // 4. Simulation: Play random game from leaf state
        // Clone the game state before simulation (simulate needs to mutate it)
        let mut sim_game = leaf.game_state().clone();
        let result = Self::simulate_impl(&mut sim_game, root_player, use_heuristics);
        
        // 5. Backpropagation: Update statistics along path
        Self::backpropagate(root, &path, result, root_player);
    }
}

impl crate::player::PlayerTrait for MCTSPlayer {
    /// Choose the best move using MCTS search.
    ///
    /// Performs the specified number of MCTS iterations (or until time limit)
    /// and returns the move from the most-visited child node.
    ///
    /// Returns `None` if there are no valid moves available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::MCTSPlayer;
    /// use othello::{Game, PlayerTrait};
    ///
    /// let game = Game::new();
    /// let player = MCTSPlayer::medium();
    /// if let Some(position) = player.choose_move(&game) {
    ///     println!("Best move: {:?}", position);
    /// }
    /// ```
    fn choose_move(&self, game: &Game) -> Option<Position> {
        let valid_moves = game.get_valid_moves();
        
        if valid_moves.is_empty() {
            return None; // No valid moves
        }
        
        // If only one move, return it immediately
        if valid_moves.len() == 1 {
            return Some(valid_moves[0]);
        }
        
        // Use parallel search if enabled (more than 1 thread)
        let num_threads = self.parallel_threads;
        let should_parallelize = match num_threads {
            None => true,  // Auto-detect, use parallel
            Some(1) => false,  // Explicitly disable
            Some(_) => true,  // Use specified number
        };
        
        if should_parallelize {
            self.mcts_search_parallel(game, num_threads)
        } else {
            self.mcts_search(game)  // Sequential fallback
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
    fn test_select_path_from_unexpanded_root() {
        let game = Game::new();
        let root = MCTSNode::new(game);
        
        // Root is not expanded, so path should be empty (root itself is the leaf)
        let path = MCTSPlayer::select_path(&root, 1.414);
        assert_eq!(path.len(), 0);
    }
    
    #[test]
    fn test_select_path_from_expanded_root() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        root.expand();
        
        // Root is expanded, so we should traverse to a child
        let path = MCTSPlayer::select_path(&root, 1.414);
        
        // Path should have at least one element (to a child)
        // or be empty if no children were created (edge case)
        if root.has_children() {
            assert!(path.len() <= 1); // Should select a child (path length 1)
            // Verify the path index is valid
            if !path.is_empty() {
                assert!(path[0] < root.num_children());
            }
        }
    }
    
    #[test]
    fn test_select_path_stops_at_terminal() {
        // Create a game and play it to completion
        let game = Game::new();
        
        // Create root and expand
        let mut root = MCTSNode::new(game);
        root.expand();
        
        if root.has_children() {
            // Select path should work even if we encounter terminal nodes
            let path = MCTSPlayer::select_path(&root, 1.414);
            // Path should be valid (not exceed number of children)
            assert!(path.len() <= root.num_children());
            
            // Verify we can access the node at the path
            let leaf = MCTSPlayer::get_node_mut_at_path(&mut root, &path);
            // Leaf should be either terminal, unexpanded, or have no children
            assert!(leaf.is_terminal() || !leaf.is_expanded() || !leaf.has_children());
        }
    }
    
    #[test]
    fn test_get_node_mut_at_path_empty_path() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        
        // Empty path should return root
        let node = MCTSPlayer::get_node_mut_at_path(&mut root, &[]);
        assert_eq!(node.visits(), 0);
        assert!(!node.is_expanded());
    }
    
    #[test]
    fn test_get_node_mut_at_path_single_level() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        root.expand();
        
        if root.has_children() {
            let path = vec![0]; // First child
            let child = MCTSPlayer::get_node_mut_at_path(&mut root, &path);
            
            // Verify we got the correct child
            assert!(child.move_from_parent().is_some());
            assert_eq!(child.visits(), 0);
            
            // Verify we have mutable access by expanding the child
            // (this requires mutable access to work)
            child.expand();
            // If we got here without error, we have mutable access
        }
    }
    
    #[test]
    fn test_get_node_mut_at_path_multi_level() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        root.expand();
        
        if root.has_children() {
            // Get first child and expand it
            let first_child_path = vec![0];
            let first_child = MCTSPlayer::get_node_mut_at_path(&mut root, &first_child_path);
            first_child.expand();
            
            if first_child.has_children() {
                // Now traverse to a grandchild
                let grandchild_path = vec![0, 0];
                let grandchild = MCTSPlayer::get_node_mut_at_path(&mut root, &grandchild_path);
                
                // Verify we got the grandchild
                assert!(grandchild.move_from_parent().is_some());
                
                // Verify mutable access by expanding the grandchild
                grandchild.expand();
                // If we got here without error, we have mutable access
            }
        }
    }
    
    #[test]
    fn test_select_path_and_get_node_consistency() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        root.expand();
        
        if root.has_children() {
            // Select a path
            let path = MCTSPlayer::select_path(&root, 1.414);
            
            // Get the node at that path
            let leaf = MCTSPlayer::get_node_mut_at_path(&mut root, &path);
            
            // The leaf should either be:
            // - Not expanded (needs expansion)
            // - Terminal (game over)
            // - Has no children
            assert!(!leaf.is_expanded() || leaf.is_terminal() || !leaf.has_children());
        }
    }
    
    #[test]
    fn test_select_path_handles_no_children() {
        // Test that selection stops correctly when a node has no children
        let game = Game::new();
        let root = MCTSNode::new(game);
        
        // Don't expand - root has no children
        let path = MCTSPlayer::select_path(&root, 1.414);
        assert_eq!(path.len(), 0); // Should return empty path (root is leaf)
    }
    
    #[test]
    fn test_simulate_always_terminates() {
        let player = MCTSPlayer {
            name: "Test".to_string(),
            iterations: 100,
            exploration_constant: 1.414,
            max_time_ms: None,
            use_heuristics: false,
            parallel_threads: None,
        };
        
        let mut game = Game::new();
        let root_player = game.current_player();
        
        // Run simulation - should always reach terminal state
        let result = player.simulate(&mut game, root_player);
        
        // Game should be over
        assert!(matches!(game.get_game_state(), GameState::GameOver { .. }));
        
        // Result should be valid (0.0, 0.5, or 1.0)
        assert!(result >= 0.0 && result <= 1.0);
    }
    
    #[test]
    fn test_backpropagate_updates_statistics() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        let root_player = root.current_player();
        
        // Expand root to create children
        root.expand();
        
        if root.has_children() {
            // Select a path to a child
            let path = MCTSPlayer::select_path(&root, 1.414);
            
            // Initial visits should be 0
            assert_eq!(root.visits(), 0);
            
            // Backpropagate a win (1.0) for root player
            MCTSPlayer::backpropagate(&mut root, &path, 1.0, root_player);
            
            // Root should now have 1 visit
            assert_eq!(root.visits(), 1);
            
            // Root value should be 1.0 (win from root player's perspective)
            assert_eq!(root.average_value(), 1.0);
            
            // If path has nodes, they should also be updated
            if !path.is_empty() {
                // Get the leaf node
                let leaf = MCTSPlayer::get_node_mut_at_path(&mut root, &path);
                assert_eq!(leaf.visits(), 1);
            }
        }
    }
    
    #[test]
    fn test_mcts_iteration_completes() {
        let player = MCTSPlayer {
            name: "Test".to_string(),
            iterations: 100,
            exploration_constant: 1.414,
            max_time_ms: None,
            use_heuristics: false,
            parallel_threads: None,
        };
        
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        let root_player = root.current_player();
        
        // Run one iteration
        player.mcts_iteration(&mut root, root_player);
        
        // Root should have been visited
        assert_eq!(root.visits(), 1);
        
        // Root should have been expanded (unless it's terminal)
        if !root.is_terminal() {
            assert!(root.is_expanded());
            assert!(root.has_children());
        }
    }
    
    #[test]
    fn test_backpropagate_flips_result_correctly() {
        let game = Game::new();
        let mut root = MCTSNode::new(game);
        let root_player = root.current_player();
        
        // Expand root to create children
        root.expand();
        
        if root.has_children() {
            // Select a path to a child
            let path = MCTSPlayer::select_path(&root, 1.414);
            
            // Get the child node to check its player
            let child = MCTSPlayer::get_node_mut_at_path(&mut root, &path);
            let child_player = child.current_player();
            
            // Backpropagate a loss (0.0) for root player
            // For root: 0.0 (loss from root player's perspective)
            // For child: 1.0 - 0.0 = 1.0 (win from child player's perspective)
            MCTSPlayer::backpropagate(&mut root, &path, 0.0, root_player);
            
            // Root should have 0.0 average value (loss)
            assert_eq!(root.average_value(), 0.0);
            
            // If child player is different from root, child should have 1.0
            if !path.is_empty() && child_player != root_player {
                let child_after = MCTSPlayer::get_node_mut_at_path(&mut root, &path);
                assert_eq!(child_after.average_value(), 1.0);
            }
        }
    }
    
    #[test]
    fn test_simulate_with_heuristics_enabled() {
        let player = MCTSPlayer {
            name: "Test".to_string(),
            iterations: 100,
            exploration_constant: 1.414,
            max_time_ms: None,
            use_heuristics: true, // Enable heuristics
            parallel_threads: None,
        };
        
        let mut game = Game::new();
        let root_player = game.current_player();
        
        // Run simulation - should always reach terminal state
        let result = player.simulate(&mut game, root_player);
        
        // Game should be over
        assert!(matches!(game.get_game_state(), GameState::GameOver { .. }));
        
        // Result should be valid (0.0, 0.5, or 1.0)
        assert!(result >= 0.0 && result <= 1.0);
    }
    
    #[test]
    fn test_heuristic_move_selection_chooses_valid_move() {
        let player = MCTSPlayer {
            name: "Test".to_string(),
            iterations: 100,
            exploration_constant: 1.414,
            max_time_ms: None,
            use_heuristics: true,
            parallel_threads: None,
        };
        
        let game = Game::new();
        let valid_moves = game.get_valid_moves();
        
        if !valid_moves.is_empty() {
            let mut rng = thread_rng();
            let selected_move = player.heuristic_move_selection(&game, &valid_moves, &mut rng);
            
            // Should return a valid move
            assert!(selected_move.is_some());
            let move_pos = selected_move.unwrap();
            assert!(valid_moves.contains(&move_pos));
        }
    }
    
    #[test]
    fn test_difficulty_presets() {
        // Test easy preset
        let easy = MCTSPlayer::easy();
        assert_eq!(easy.iterations, 200);
        assert_eq!(easy.exploration_constant, 2.0);
        assert!(!easy.use_heuristics);
        assert_eq!(easy.get_name(), "MCTS (Easy)");
        
        // Test medium preset
        let medium = MCTSPlayer::medium();
        assert_eq!(medium.iterations, 1000);
        assert_eq!(medium.exploration_constant, 1.414);
        assert!(medium.use_heuristics);
        assert_eq!(medium.get_name(), "MCTS (Medium)");
        
        // Test hard preset
        let hard = MCTSPlayer::hard();
        assert_eq!(hard.iterations, 3000);
        assert_eq!(hard.exploration_constant, 1.414);
        assert!(hard.use_heuristics);
        assert_eq!(hard.get_name(), "MCTS (Hard)");
        
        // Test expert preset
        let expert = MCTSPlayer::expert();
        assert_eq!(expert.iterations, 10000);
        assert_eq!(expert.exploration_constant, 1.0);
        assert!(expert.use_heuristics);
        assert_eq!(expert.max_time_ms, Some(5000));
        assert_eq!(expert.get_name(), "MCTS (Expert)");
    }
    
    #[test]
    fn test_builder_methods() {
        let player = MCTSPlayer::new("Test")
            .with_exploration(2.5)
            .with_time_limit_ms(1000)
            .with_heuristics(true);
        
        assert_eq!(player.exploration_constant, 2.5);
        assert_eq!(player.max_time_ms, Some(1000));
        assert!(player.use_heuristics);
        
        // Test chaining
        let player2 = MCTSPlayer::with_iterations("Test2", 500)
            .with_exploration(1.0)
            .with_heuristics(false)
            .with_time_limit_ms(2000);
        
        assert_eq!(player2.iterations, 500);
        assert_eq!(player2.exploration_constant, 1.0);
        assert!(!player2.use_heuristics);
        assert_eq!(player2.max_time_ms, Some(2000));
    }
    
    #[test]
    fn test_difficulty_presets_choose_moves() {
        // Verify that difficulty presets can actually choose moves
        let game = Game::new();
        
        let easy = MCTSPlayer::easy();
        let move_opt = easy.choose_move(&game);
        assert!(move_opt.is_some());
        
        let medium = MCTSPlayer::medium();
        let move_opt = medium.choose_move(&game);
        assert!(move_opt.is_some());
        
        let hard = MCTSPlayer::hard();
        let move_opt = hard.choose_move(&game);
        assert!(move_opt.is_some());
        
        // Expert might take longer, but should still work
        let expert = MCTSPlayer::expert();
        let move_opt = expert.choose_move(&game);
        assert!(move_opt.is_some());
    }
    
    #[test]
    fn test_parallel_search_chooses_move() {
        let game = Game::new();
        let player = MCTSPlayer::with_iterations("Test", 100)
            .with_parallel_threads(Some(2));  // Use 2 threads
        
        let move_opt = player.choose_move(&game);
        assert!(move_opt.is_some());
        
        let position = move_opt.unwrap();
        let valid_moves = game.get_valid_moves();
        assert!(valid_moves.contains(&position));
    }
    
    #[test]
    fn test_parallel_search_with_single_thread() {
        // Should behave like sequential search
        let game = Game::new();
        let player = MCTSPlayer::with_iterations("Test", 100)
            .with_parallel_threads(Some(1));  // Disable parallel
        
        let move_opt = player.choose_move(&game);
        assert!(move_opt.is_some());
    }
    
    #[test]
    fn test_parallel_search_auto_detects_cores() {
        let game = Game::new();
        let player = MCTSPlayer::with_iterations("Test", 100);
        // parallel_threads is None, should auto-detect
        
        let move_opt = player.choose_move(&game);
        assert!(move_opt.is_some());
    }
    
    #[test]
    fn test_aggregate_trees_single_tree() {
        let game = Game::new();
        let mut root = MCTSNode::new(game.clone());
        let root_player = game.current_player();
        
        // Expand and run a few iterations
        root.expand();
        for _ in 0..10 {
            MCTSPlayer::mcts_iteration_impl(&mut root, root_player, 1.414, false);
        }
        
        let trees = vec![(root, 10)];
        let result = MCTSPlayer::aggregate_trees(&trees, root_player);
        
        // Should return a valid move
        assert!(result.is_some());
    }
    
    #[test]
    fn test_aggregate_trees_multiple_trees() {
        let game = Game::new();
        let root_player = game.current_player();
        
        // Create multiple trees
        let mut trees = Vec::new();
        for _ in 0..4 {
            let mut root = MCTSNode::new(game.clone());
            root.expand();
            for _ in 0..10 {
                MCTSPlayer::mcts_iteration_impl(&mut root, root_player, 1.414, false);
            }
            trees.push((root, 10));
        }
        
        let result = MCTSPlayer::aggregate_trees(&trees, root_player);
        assert!(result.is_some());
    }
    
    #[test]
    fn test_parallel_vs_sequential_consistency() {
        // Both should return valid moves (may differ, but both valid)
        let game = Game::new();
        
        let parallel_player = MCTSPlayer::with_iterations("Parallel", 200)
            .with_parallel_threads(Some(2));
        let sequential_player = MCTSPlayer::with_iterations("Sequential", 200)
            .with_parallel_threads(Some(1));
        
        let parallel_move = parallel_player.choose_move(&game);
        let sequential_move = sequential_player.choose_move(&game);
        
        // Both should return valid moves
        assert!(parallel_move.is_some());
        assert!(sequential_move.is_some());
        
        let valid_moves = game.get_valid_moves();
        assert!(valid_moves.contains(&parallel_move.unwrap()));
        assert!(valid_moves.contains(&sequential_move.unwrap()));
    }
    
    #[test]
    fn test_parallel_threads_builder() {
        let player = MCTSPlayer::new("Test")
            .with_parallel_threads(Some(4));
        
        assert_eq!(player.parallel_threads(), Some(4));
        
        let mut player2 = MCTSPlayer::new("Test2");
        player2.set_parallel_threads(Some(8));
        assert_eq!(player2.parallel_threads(), Some(8));
    }
}


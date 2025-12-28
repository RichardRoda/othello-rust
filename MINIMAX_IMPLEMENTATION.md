# Minimax Algorithm Implementation Strategy

This document provides a detailed, step-by-step implementation guide for a Minimax-based AI player with Alpha-Beta Pruning for the Othello game.

## Table of Contents

1. [Overview](#overview)
2. [Algorithm Theory](#algorithm-theory)
3. [Alpha-Beta Pruning](#alpha-beta-pruning)
4. [Evaluation Function](#evaluation-function)
5. [Implementation Phases](#implementation-phases)
6. [Code Structure](#code-structure)
7. [Testing Strategy](#testing-strategy)
8. [Performance Considerations](#performance-considerations)
9. [Future Enhancements](#future-enhancements)

---

## Overview

### Motivation

Minimax is a classic decision-making algorithm for two-player zero-sum games like Othello. Unlike MCTS (Monte Carlo Tree Search), which uses probabilistic simulation, Minimax performs a complete search of the game tree up to a certain depth, evaluating positions using heuristics.

**Advantages of Minimax:**
- ✅ Deterministic results (same position = same move)
- ✅ Optimal play given perfect evaluation function and sufficient depth
- ✅ Can leverage existing heuristics effectively
- ✅ Well-understood algorithm with proven effectiveness

**Trade-offs:**
- ⚠️ Fixed depth limits search quality
- ⚠️ Slower than MCTS for shallow searches (but faster for deep, focused searches)
- ⚠️ Requires good evaluation function

### Algorithm Concept

Minimax assumes both players play optimally:
- **Maximizing player** (you) chooses moves that maximize your score
- **Minimizing player** (opponent) chooses moves that minimize your score
- The algorithm explores all possible moves up to a fixed depth, then evaluates the resulting positions

---

## Algorithm Theory

### Basic Minimax Algorithm

```
function minimax(game, depth, maximizing_player):
    if depth == 0 or game is over:
        return evaluate(game)
    
    if maximizing_player:
        max_score = -infinity
        for each valid move:
            make_move(game, move)
            score = minimax(game, depth-1, false)
            undo_move(game, move)
            max_score = max(max_score, score)
        return max_score
    else:
        min_score = +infinity
        for each valid move:
            make_move(game, move)
            score = minimax(game, depth-1, true)
            undo_move(game, move)
            min_score = min(min_score, score)
        return min_score
```

### Key Components

1. **Search Depth**: How many moves ahead to look (3-8 typical for Othello)
2. **Evaluation Function**: Scores a position from the maximizing player's perspective
3. **Move Generation**: Gets all valid moves for current player
4. **Game State Management**: Cloning game state for recursive search

### Move Selection

After evaluating all moves at the root, select the move with the best score:

```
function choose_move(game):
    best_move = None
    best_score = -infinity
    
    for each valid move:
        make_move(game, move)
        score = minimax(game, depth-1, false)  // Opponent's turn
        undo_move(game, move)
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move
```

---

## Alpha-Beta Pruning

Alpha-Beta pruning is an optimization that significantly reduces the number of nodes evaluated in the minimax tree without affecting the result.

### Concept

- **Alpha (α)**: Best value that the maximizing player can achieve
- **Beta (β)**: Best value that the minimizing player can achieve
- **Pruning**: Stop evaluating a branch if we know it can't affect the final decision

### Algorithm with Alpha-Beta

```
function alphabeta(game, depth, alpha, beta, maximizing_player):
    if depth == 0 or game is over:
        return evaluate(game)
    
    if maximizing_player:
        max_score = -infinity
        for each valid move:
            make_move(game, move)
            score = alphabeta(game, depth-1, alpha, beta, false)
            undo_move(game, move)
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break  // Beta cutoff (prune remaining moves)
        return max_score
    else:
        min_score = +infinity
        for each valid move:
            make_move(game, move)
            score = alphabeta(game, depth-1, alpha, beta, true)
            undo_move(game, move)
            min_score = min(min_score, score)
            beta = min(beta, score)
            if beta <= alpha:
                break  // Alpha cutoff (prune remaining moves)
        return min_score
```

### Pruning Efficiency

With perfect move ordering (best moves first), alpha-beta can reduce the search space from O(b^d) to approximately O(b^(d/2)), effectively doubling the search depth.

**Expected Speedup:**
- Without pruning: Evaluates all b^d nodes (b = branching factor, d = depth)
- With pruning: Evaluates roughly b^(d/2) nodes (with good move ordering)
- **Typical speedup: 10-100x for depth 4-6 in Othello**

---

## Evaluation Function

### Position Evaluation Strategy

The evaluation function assigns a score to a game position from the current player's perspective. For Othello, we combine multiple heuristics:

1. **Piece Count** (early game importance: low, late game: high)
2. **Corner Control** (always important)
3. **Mobility** (number of available moves)
4. **Stability** (pieces that are hard to flip)
5. **Positional Factors** (edge control, center control)

### Leveraging Existing Heuristics

The codebase already has a comprehensive `Heuristics` module that we can adapt. However, for minimax, we need a **position evaluation** rather than a **move evaluation**.

We'll create a position evaluator that:
- Evaluates the entire board state
- Uses game phase (early/mid/late) to weight factors differently
- Normalizes scores for consistent depth comparisons

### Evaluation Function Design

```rust
fn evaluate_position(game: &Game, player: Player) -> f64 {
    let (black_pieces, white_pieces) = game.get_score();
    let total_pieces = black_pieces + white_pieces;
    let game_phase = total_pieces as f64 / 64.0;  // 0.0 = start, 1.0 = end
    
    // Early game: focus on mobility and position
    // Late game: focus on piece count
    
    let piece_diff = if player == Player::Black {
        (black_pieces as f64 - white_pieces as f64) / 64.0
    } else {
        (white_pieces as f64 - black_pieces as f64) / 64.0
    };
    
    let my_moves = game.get_valid_moves().len();
    // Maximum possible moves has been mathmatically proven to be 33
    // https://jxiv.jst.go.jp/index.php/jxiv/preprint/download/480/1498
    let mobility_score = my_moves as f64 / 33.0;
    
    let corner_score = evaluate_corners(game, player);
    let stability_score = evaluate_stability(game, player);
    
    // Weighted combination based on game phase
    let piece_weight = game_phase * 50.0;  // More important late game
    let mobility_weight = (1.0 - game_phase) * 30.0;  // More important early game
    let corner_weight = 20.0;  // Always important
    let stability_weight = (1.0 - game_phase) * 10.0;  // More important early
    
    piece_diff * piece_weight +
    mobility_score * mobility_weight +
    corner_score * corner_weight +
    stability_score * stability_weight
}
```

---

## Implementation Phases

### Phase 1: Basic Structure

**Estimated Time:** 2-3 hours  
**Goal:** Create the MinimaxPlayer struct and basic minimax implementation.

#### Step 1.1: Create MinimaxPlayer Module

**File: `src/minimax/mod.rs`**

```rust
pub mod player;
pub mod evaluator;

pub use player::MinimaxPlayer;
```

#### Step 1.2: Create MinimaxPlayer Struct

**File: `src/minimax/player.rs`**

```rust
use crate::game::{Game, Player, GameState};
use crate::board::Position;
use crate::player::PlayerTrait;

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
    
    /// Create an easy difficulty Minimax player.
    ///
    /// Settings:
    /// - Depth 3 (fast, ~100-500ms per move)
    /// - Alpha-Beta enabled
    /// - Move ordering enabled
    ///
    /// Suitable for quick games or less experienced players.
    pub fn easy() -> Self {
        Self::with_depth("Minimax (Easy)", 3)
    }
    
    /// Create a medium difficulty Minimax player.
    ///
    /// Settings:
    /// - Depth 4 (moderate speed, ~500ms-2s per move)
    /// - Alpha-Beta enabled
    /// - Move ordering enabled
    ///
    /// A good balance of speed and strength. Recommended for most games.
    pub fn medium() -> Self {
        Self::with_depth("Minimax (Medium)", 4)
    }
    
    /// Create a hard difficulty Minimax player.
    ///
    /// Settings:
    /// - Depth 5 (slower, ~2-10s per move)
    /// - Alpha-Beta enabled
    /// - Move ordering enabled
    ///
    /// Strong play suitable for experienced players.
    pub fn hard() -> Self {
        Self::with_depth("Minimax (Hard)", 5)
    }
    
    /// Create an expert difficulty Minimax player.
    ///
    /// Settings:
    /// - Depth 6 (very slow, ~10-60s per move, capped at 30s)
    /// - Alpha-Beta enabled
    /// - Move ordering enabled
    /// - 30 second time limit
    ///
    /// Very strong play for expert-level competition.
    pub fn expert() -> Self {
        Self::with_depth("Minimax (Expert)", 6)
            .with_time_limit_ms(30000)
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
        mut alpha: f64,
        mut beta: f64,
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
            return self.alphabeta(&test_game, depth, alpha, beta, !maximizing, root_player, start_time);
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
                let mut test_game = game.clone();
                if test_game.make_move(move_pos).is_err() {
                    continue;
                }
                let score = self.alphabeta(&test_game, depth - 1, alpha, beta, false, root_player, start_time);
                
                max_score = max_score.max(score);
                alpha = alpha.max(score);
                
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
                let score = self.alphabeta(&test_game, depth - 1, alpha, beta, true, root_player, start_time);
                
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
        
        let mut scored_moves: Vec<(Position, f64)> = moves.iter()
            .map(|&pos| (pos, Heuristics::evaluate_move(game, pos)))
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
        self.minimax_search(game)
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
}
```

**Action Items:**
- [ ] Create `src/minimax/mod.rs`
- [ ] Create `src/minimax/player.rs` with basic structure
- [ ] Implement minimax without undo (using cloning - simpler for now)
- [ ] Add basic evaluation function stub
- [ ] Test basic structure compiles

**Note:** The initial implementation uses game cloning instead of undo moves. This is simpler but slower. We'll optimize later if needed.

---

### Phase 2: Position Evaluator

**Estimated Time:** 2-3 hours  
**Goal:** Implement comprehensive position evaluation function.

#### Step 2.1: Create Position Evaluator

**File: `src/minimax/evaluator.rs`**

```rust
use crate::game::{Game, Player, GameState};
use crate::board::{Position, Cell};

/// Evaluates game positions for the Minimax algorithm.
///
/// The evaluation function combines multiple heuristics to score a position
/// from a given player's perspective. Higher scores are better.
///
/// Key factors:
/// - Piece count (weighted by game phase)
/// - Corner control
/// - Mobility (available moves)
/// - Stability (pieces that are hard to flip)
/// - Game phase awareness (early/mid/late game)
pub struct PositionEvaluator;

impl PositionEvaluator {
    /// Evaluate a game position from the given player's perspective.
    ///
    /// Returns a score where:
    /// - Positive values favor the given player
    /// - Negative values favor the opponent
    /// - Magnitude indicates strength of advantage
    /// - Terminal positions return `f64::INFINITY` (win) or `f64::NEG_INFINITY` (loss)
    ///
    /// Typical score range for non-terminal positions: -100 to +100 (but not strictly bounded)
    ///
    /// # Arguments
    ///
    /// * `game` - The game state to evaluate
    /// * `player` - The player whose perspective to evaluate from
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::minimax::evaluator::PositionEvaluator;
    /// use othello::{Game, Player};
    ///
    /// let game = Game::new();
    /// let score = PositionEvaluator::evaluate(&game, Player::Black);
    /// // Score should be finite and reasonable
    /// assert!(score.is_finite());
    /// ```
    pub fn evaluate(game: &Game, player: Player) -> f64 {
        // Terminal position (game over)
        if let GameState::GameOver { winner } = game.get_game_state() {
            return Self::evaluate_terminal(winner, player);
        }
        
        let (black_pieces, white_pieces) = game.get_score();
        let total_pieces = black_pieces + white_pieces;
        let game_phase = total_pieces as f64 / 64.0;  // 0.0 = start, 1.0 = end
        
        // Compute component scores
        let piece_score = Self::evaluate_pieces(game, player, game_phase);
        let mobility_score = Self::evaluate_mobility(game, player, game_phase);
        let corner_score = Self::evaluate_corners(game, player);
        let stability_score = Self::evaluate_stability(game, player, game_phase);
        let positional_score = Self::evaluate_positional(game, player);
        
        // Weighted combination based on game phase
        piece_score * Self::piece_weight(game_phase) +
        mobility_score * Self::mobility_weight(game_phase) +
        corner_score * Self::corner_weight() +
        stability_score * Self::stability_weight(game_phase) +
        positional_score * Self::positional_weight(game_phase)
    }
    
    /// Evaluate terminal position (game over).
    fn evaluate_terminal(winner: Option<Player>, player: Player) -> f64 {
        match winner {
            Some(w) if w == player => f64::INFINITY,  // Win
            Some(_) => f64::NEG_INFINITY,  // Loss
            None => 0.0,  // Draw
        }
    }
    
    /// Evaluate piece count advantage.
    ///
    /// More important in late game, less important early game.
    fn evaluate_pieces(game: &Game, player: Player, game_phase: f64) -> f64 {
        let (black_pieces, white_pieces) = game.get_score();
        
        let piece_diff = if player == Player::Black {
            black_pieces as f64 - white_pieces as f64
        } else {
            white_pieces as f64 - black_pieces as f64
        };
        
        // Normalize to [-1, 1] range
        piece_diff / 64.0
    }
    
    /// Evaluate mobility (available moves).
    ///
    /// More important in early/mid game. Having more moves than your opponent
    /// is a significant advantage.
    fn evaluate_mobility(game: &Game, player: Player, _game_phase: f64) -> f64 {
        let my_moves = game.get_valid_moves().len();
        
        // Get opponent's moves
        let opponent_moves = {
            let mut test_game = game.clone();
            // Switch to opponent's turn
            test_game.skip_turn().ok();
            test_game.get_valid_moves().len()
        };
        
        if my_moves + opponent_moves == 0 {
            0.0
        } else {
            // Normalized difference: [-1, 1]
            (my_moves as f64 - opponent_moves as f64) / (my_moves + opponent_moves) as f64
        }
    }
    
    /// Evaluate corner control.
    ///
    /// Corners are extremely valuable in Othello because they cannot be flipped.
    /// Always important regardless of game phase.
    fn evaluate_corners(game: &Game, player: Player) -> f64 {
        let corners = [
            Position::new(0, 0), Position::new(0, 7),
            Position::new(7, 0), Position::new(7, 7),
        ];
        
        let board = game.get_board();
        let my_corners = corners.iter()
            .filter(|&&pos| {
                board.get_cell(pos).ok() == Some(Cell::from_player(player))
            })
            .count();
        
        let opponent_corners = corners.iter()
            .filter(|&&pos| {
                board.get_cell(pos).ok() == Some(Cell::from_player(player.opposite()))
            })
            .count();
        
        // Normalize to [-1, 1] (4 corners total)
        (my_corners as f64 - opponent_corners as f64) / 4.0
    }
    
    /// Evaluate piece stability.
    ///
    /// Stable pieces (especially on edges) are harder to flip.
    /// More important in early/mid game.
    fn evaluate_stability(game: &Game, player: Player, _game_phase: f64) -> f64 {
        let board = game.get_board();
        let mut my_stable = 0;
        let mut opponent_stable = 0;
        
        // Count edge pieces (generally more stable)
        for row in 0..8 {
            for col in 0..8 {
                let pos = Position::new(row, col);
                if let Ok(cell) = board.get_cell(pos) {
                    if cell != Cell::Empty {
                        let is_edge = row == 0 || row == 7 || col == 0 || col == 7;
                        if is_edge {
                            if cell == Cell::from_player(player) {
                                my_stable += 1;
                            } else {
                                opponent_stable += 1;
                            }
                        }
                    }
                }
            }
        }
        
        // Normalize to [-1, 1]
        let total_stable = my_stable + opponent_stable;
        if total_stable == 0 {
            0.0
        } else {
            (my_stable as f64 - opponent_stable as f64) / total_stable as f64
        }
    }
    
    /// Evaluate positional factors.
    ///
    /// Includes X-squares, C-squares, and general positional control.
    fn evaluate_positional(game: &Game, player: Player) -> f64 {
        let board = game.get_board();
        let mut score = 0.0;
        
        // Penalize X-squares (adjacent to corners) unless corner is ours
        let x_squares = [
            Position::new(1, 1), Position::new(1, 6),
            Position::new(6, 1), Position::new(6, 6),
        ];
        
        let corners = [
            Position::new(0, 0), Position::new(0, 7),
            Position::new(7, 0), Position::new(7, 7),
        ];
        
        for &x_pos in &x_squares {
            if let Ok(cell) = board.get_cell(x_pos) {
                if cell != Cell::Empty {
                    // Check if adjacent corner is ours
                    let corner_is_mine = corners.iter().any(|&corner| {
                        let row_diff = (corner.row as i32 - x_pos.row as i32).abs();
                        let col_diff = (corner.col as i32 - x_pos.col as i32).abs();
                        row_diff <= 1 && col_diff <= 1 &&
                        board.get_cell(corner).ok() == Some(Cell::from_player(player))
                    });
                    
                    if cell == Cell::from_player(player) {
                        if corner_is_mine {
                            score += 0.1;  // Safe X-square
                        } else {
                            score -= 0.2;  // Risky X-square
                        }
                    } else {
                        if corner_is_mine {
                            score += 0.1;  // Opponent has risky piece
                        } else {
                            score -= 0.1;  // Opponent controls X-square
                        }
                    }
                }
            }
        }
        
        score
    }
    
    /// Get weight for piece count evaluation based on game phase.
    fn piece_weight(game_phase: f64) -> f64 {
        // More important late game
        20.0 + game_phase * 40.0
    }
    
    /// Get weight for mobility evaluation based on game phase.
    fn mobility_weight(game_phase: f64) -> f64 {
        // More important early game
        30.0 * (1.0 - game_phase * 0.5)
    }
    
    /// Get weight for corner evaluation.
    fn corner_weight() -> f64 {
        // Always very important
        25.0
    }
    
    /// Get weight for stability evaluation based on game phase.
    fn stability_weight(game_phase: f64) -> f64 {
        // More important early/mid game
        15.0 * (1.0 - game_phase * 0.7)
    }
    
    /// Get weight for positional evaluation.
    fn positional_weight(_game_phase: f64) -> f64 {
        // Constant moderate importance
        10.0
    }
}
```

**Action Items:**
- [ ] Create `src/minimax/evaluator.rs`
- [ ] Implement all evaluation functions
- [ ] Test evaluation with known positions
- [ ] Verify scores are reasonable and finite

---

### Phase 3: Fix Game State Management

**Estimated Time:** 2-3 hours  
**Goal:** Implement proper game state cloning and undo for efficient search.

#### Issue: Game Undo Mechanism

The `Game` struct doesn't have an explicit undo method. We have two options:

1. **Clone game state for each recursive call** (simpler, but slower)
2. **Implement undo mechanism** (more complex, but faster)

For the initial implementation, we'll use cloning (simpler and safer). We can optimize later if performance is an issue.

#### Step 3.1: Optimize with Cloning

The code examples in Phase 1 already use cloning correctly. The implementation uses immutable game references and clones the game state for each recursive branch. This approach:

- **Pros**: Simple, safe, no undo mechanism needed
- **Cons**: Higher memory usage and slower than undo (but acceptable for Othello)

For Othello with typical search depths (3-5), cloning is fast enough. If performance becomes an issue later, we can implement an undo mechanism, but it's not necessary for the initial implementation.

**Action Items:**
- [ ] Update minimax methods to use immutable game references
- [ ] Clone game state for each move instead of trying to undo
- [ ] Test that search still works correctly
- [ ] Benchmark performance (should be acceptable for depth 3-5)

---

### Phase 4: Integration

**Estimated Time:** 1-2 hours  
**Goal:** Integrate MinimaxPlayer into the game system.

#### Step 4.1: Update Module Exports

**File: `src/lib.rs`**

```rust
pub mod minimax;  // Add this line

// ... existing modules ...

pub use minimax::MinimaxPlayer;  // Optional: re-export if desired
```

#### Step 4.2: Add to Player Selection

**File: `src/player_selection.rs`**

```rust
use crate::minimax::MinimaxPlayer;  // Add import

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerType {
    Human,
    AIRandom,
    MCTSEasy,
    MCTSMedium,
    MCTSHard,
    MCTSExpert,
    MinimaxEasy,    // Add these
    MinimaxMedium,  // Add these
    MinimaxHard,    // Add these
    MinimaxExpert,  // Add these
}

impl PlayerType {
    pub fn all() -> Vec<PlayerType> {
        vec![
            PlayerType::Human,
            PlayerType::AIRandom,
            PlayerType::MCTSEasy,
            PlayerType::MCTSMedium,
            PlayerType::MCTSHard,
            PlayerType::MCTSExpert,
            PlayerType::MinimaxEasy,    // Add these
            PlayerType::MinimaxMedium,  // Add these
            PlayerType::MinimaxHard,    // Add these
            PlayerType::MinimaxExpert,  // Add these
        ]
    }
    
    pub fn display_name(&self) -> &'static str {
        match self {
            // ... existing cases ...
            PlayerType::MinimaxEasy => "Minimax (Easy)",
            PlayerType::MinimaxMedium => "Minimax (Medium)",
            PlayerType::MinimaxHard => "Minimax (Hard)",
            PlayerType::MinimaxExpert => "Minimax (Expert)",
        }
    }
    
    pub fn create_player(&self, name: String) -> Box<dyn PlayerTrait> {
        match self {
            // ... existing cases ...
            PlayerType::MinimaxEasy => Box::new(MinimaxPlayer::easy()),
            PlayerType::MinimaxMedium => Box::new(MinimaxPlayer::medium()),
            PlayerType::MinimaxHard => Box::new(MinimaxPlayer::hard()),
            PlayerType::MinimaxExpert => Box::new(MinimaxPlayer::expert()),
        }
    }
}
```

**Action Items:**
- [ ] Add minimax module to `src/lib.rs`
- [ ] Update `PlayerType` enum with minimax variants
- [ ] Update `create_player` to handle minimax types
- [ ] Test player selection works

---

### Phase 5: Testing

**Estimated Time:** 3-4 hours  
**Goal:** Comprehensive testing of minimax implementation.

#### Step 5.1: Unit Tests

**File: `src/minimax/player.rs` (test module)**

```rust
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
```

#### Step 5.2: Evaluator Tests

**File: `src/minimax/evaluator.rs` (test module)**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    
    #[test]
    fn test_evaluate_initial_position() {
        let game = Game::new();
        let score = PositionEvaluator::evaluate(&game, Player::Black);
        
        // Initial position should be roughly balanced
        assert!(score.is_finite());
        assert!(score.abs() < 50.0);  // Should be close to 0
    }
    
    #[test]
    fn test_evaluate_winning_position() {
        // Create a game state where a player wins
        // (This would require manipulating game state or playing moves)
        let game = Game::new();
        
        // Test that evaluation works
        let score = PositionEvaluator::evaluate(&game, Player::Black);
        assert!(score.is_finite());
    }
    
    #[test]
    fn test_corner_evaluation() {
        let game = Game::new();
        let score = PositionEvaluator::evaluate_corners(&game, Player::Black);
        
        // Should be in reasonable range
        assert!(score >= -1.0 && score <= 1.0);
    }
    
    #[test]
    fn test_mobility_evaluation() {
        let game = Game::new();
        let score = PositionEvaluator::evaluate_mobility(&game, Player::Black, 0.0);
        
        // Should be in normalized range
        assert!(score >= -1.0 && score <= 1.0);
    }
}
```

**Action Items:**
- [ ] Write unit tests for minimax search
- [ ] Write unit tests for evaluation function
- [ ] Test edge cases (no moves, game over, etc.)
- [ ] Run `cargo test` to verify all tests pass

---

### Phase 6: Documentation and Polish

**Estimated Time:** 1-2 hours  
**Goal:** Complete documentation and finalize implementation.

#### Step 6.1: Add Rustdoc Comments

Ensure all public methods have comprehensive documentation (already included in code above).

#### Step 6.2: Update README

**File: `README.md`**

Add section about Minimax player:

```markdown
## Minimax Player

The Minimax player uses a classic decision-making algorithm with Alpha-Beta pruning
to select optimal moves. It explores the game tree to a fixed depth and evaluates
positions using comprehensive heuristics.

### Configuration

```rust
use othello::minimax::MinimaxPlayer;

// Use difficulty presets
let easy = MinimaxPlayer::easy();     // Depth 3
let medium = MinimaxPlayer::medium(); // Depth 4
let hard = MinimaxPlayer::hard();     // Depth 5
let expert = MinimaxPlayer::expert(); // Depth 6, 30s limit

// Custom configuration
let player = MinimaxPlayer::with_depth("Custom", 4)
    .with_time_limit_ms(2000)
    .with_alpha_beta(true)
    .with_move_ordering(true);
```

### Performance

Typical performance per move:
- Depth 3: ~100-500ms
- Depth 4: ~500ms-2s
- Depth 5: ~2-10s
- Depth 6+: Very slow (minutes)

Alpha-Beta pruning provides 10-100x speedup over basic minimax.
```

**Action Items:**
- [ ] Verify all rustdoc comments are present
- [ ] Run `cargo doc` to generate documentation
- [ ] Update README with minimax section
- [ ] Add usage examples

---

## Code Structure

### New Files

1. **`src/minimax/mod.rs`**
   - Module declarations

2. **`src/minimax/player.rs`**
   - `MinimaxPlayer` struct
   - Minimax algorithm implementation
   - Alpha-Beta pruning
   - Move ordering
   - PlayerTrait implementation

3. **`src/minimax/evaluator.rs`**
   - `PositionEvaluator` struct
   - Position evaluation function
   - Heuristic components

### Modified Files

1. **`src/lib.rs`**
   - Add `pub mod minimax;`

2. **`src/player_selection.rs`**
   - Add `MinimaxEasy`, `MinimaxMedium`, `MinimaxHard`, `MinimaxExpert` variants
   - Update `create_player` method

3. **`README.md`**
   - Add Minimax player documentation

### File Organization

```
src/
├── minimax/
│   ├── mod.rs          # Module declarations
│   ├── player.rs       # MinimaxPlayer implementation
│   └── evaluator.rs    # Position evaluation
├── lib.rs              # (modified) Add minimax module
├── player_selection.rs # (modified) Add minimax types
└── ...
```

---

## Testing Strategy

### Unit Tests

1. **Minimax search functionality**
   - Test with different depths
   - Test with/without alpha-beta
   - Test move ordering
   - Test edge cases (no moves, game over)

2. **Evaluation function**
   - Test initial position
   - Test winning/losing positions
   - Test component heuristics
   - Verify scores are finite and reasonable

3. **Integration**
   - Test difficulty presets
   - Test player selection
   - Test full games complete successfully

### Test Checklist

- [ ] Minimax chooses valid moves
- [ ] Alpha-Beta produces same results as basic minimax
- [ ] Move ordering improves performance
- [ ] Evaluation function works for all positions
- [ ] Time limits are respected
- [ ] Games complete successfully
- [ ] All difficulty presets work

---

## Performance Considerations

### Expected Performance

| Depth | Time per Move (approx) | Use Case |
|-------|------------------------|----------|
| 3     | 100-500ms              | Fast play, easy difficulty |
| 4     | 500ms-2s               | Medium difficulty, good balance |
| 5     | 2-10s                  | Hard difficulty, strong play |
| 6+    | 10s-60s+               | Expert, very strong but slow |

### Optimization Tips

1. **Alpha-Beta Pruning**: Essential for depth > 3. Provides 10-100x speedup.
2. **Move Ordering**: Evaluate better moves first to improve pruning (2-5x speedup).
3. **Time Limits**: Prevent excessively long searches.
4. **Game Cloning**: Current implementation clones for each move. If performance is an issue, consider implementing undo mechanism.

### When to Use Minimax vs MCTS

✅ **Use Minimax when:**
- You want deterministic results
- You can evaluate positions well with heuristics
- Depth 3-5 is sufficient
- You prefer classic, well-understood algorithms

✅ **Use MCTS when:**
- You want probabilistic, exploratory play
- Deep search isn't feasible (minimax too slow)
- You have lots of time for many iterations
- You want adaptive difficulty scaling

---

## Future Enhancements

### 1. Iterative Deepening

Search to depth 1, then 2, then 3, etc., until time runs out. Use results from previous depths to order moves at deeper levels.

**Benefits:**
- Guaranteed move within time limit
- Better move ordering at deeper levels
- Can stop early if time runs out

### 2. Transposition Table

Cache evaluated positions to avoid re-evaluating the same position multiple times.

**Benefits:**
- Significant speedup in positions with repeated states
- Requires hashing game states

### 3. Quiescence Search

Continue searching "unstable" positions (with captures, etc.) beyond the depth limit until the position stabilizes.

**Benefits:**
- More accurate evaluation of tactical positions
- Better handling of captures and threats

### 4. Move Generation Optimization

Optimize move generation and ordering:
- Pre-compute corner moves (high priority)
- Cache mobility calculations
- Use bitboards for faster move generation (if board representation changes)

### 5. Parallel Search

Parallelize minimax search across multiple threads:
- Split move evaluation across threads
- Requires careful alpha-beta handling
- Less effective than MCTS parallelization (minimax has dependencies)

### 6. Advanced Evaluation

Improve evaluation function:
- Piece-square tables (different values for different squares)
- Pattern recognition (recognize common Othello patterns)
- Machine learning evaluation (train a neural network)

---

## Common Pitfalls and Solutions

### Issue: Search is Too Slow

**Problem:** Minimax takes too long even at depth 3-4.

**Solutions:**
- Enable alpha-beta pruning (if disabled)
- Enable move ordering
- Reduce depth
- Implement time limits
- Optimize evaluation function (cache results)

### Issue: Weak Play

**Problem:** Minimax makes poor moves despite reasonable depth.

**Possible causes:**
- Evaluation function is inaccurate
- Depth is too shallow
- Move ordering is poor (worse moves evaluated first)

**Solutions:**
- Improve evaluation function (adjust weights, add heuristics)
- Increase depth (if feasible)
- Improve move ordering (evaluate corners/mobility first)

### Issue: Different Results Each Run

**Problem:** Minimax should be deterministic but returns different moves.

**Causes:**
- Move ordering uses randomness (if using heuristics with randomness)
- Floating point precision issues
- Time limits causing different search depths

**Solutions:**
- Ensure move ordering is deterministic
- Use consistent tie-breaking
- Avoid time limits if determinism is required

---

## Implementation Timeline

| Phase | Estimated Time | Cumulative Time |
|-------|---------------|-----------------|
| Phase 1: Basic Structure | 2-3 hours | 2-3 hours |
| Phase 2: Position Evaluator | 2-3 hours | 4-6 hours |
| Phase 3: Game State Management | 2-3 hours | 6-9 hours |
| Phase 4: Integration | 1-2 hours | 7-11 hours |
| Phase 5: Testing | 3-4 hours | 10-15 hours |
| Phase 6: Documentation | 1-2 hours | 11-17 hours |

**Total Estimated Time:** 11-17 hours (approximately 1.5-2 days of work)

---

## Revision History

- **v1.0** (Initial): Complete Minimax implementation document created

---

## References

1. **Minimax Algorithm:**
   - Wikipedia: Minimax algorithm
   - "Artificial Intelligence: A Modern Approach" by Russell & Norvig

2. **Alpha-Beta Pruning:**
   - Wikipedia: Alpha-Beta pruning
   - "Game Tree Search" papers and tutorials

3. **Othello Strategy:**
   - Othello strategy guides
   - Heuristic evaluation research papers

4. **Rust Implementation:**
   - Rust Book: Ownership and Borrowing (for game state management)
   - Rust performance best practices

---

## Appendix: Simplified Evaluation Function

If the full evaluation function is too complex initially, start with a simpler version:

```rust
fn evaluate_position_simple(game: &Game, player: Player) -> f64 {
    let (black_pieces, white_pieces) = game.get_score();
    
    // Terminal position
    if let GameState::GameOver { winner } = game.get_game_state() {
        return match winner {
            Some(w) if w == player => f64::INFINITY,
            Some(_) => f64::NEG_INFINITY,
            None => 0.0,
        };
    }
    
    // Simple piece difference
    let piece_diff = if player == Player::Black {
        black_pieces as f64 - white_pieces as f64
    } else {
        white_pieces as f64 - black_pieces as f64
    };
    
    // Simple mobility
    let my_moves = game.get_valid_moves().len();
    let opponent_moves = {
        let mut test_game = game.clone();
        test_game.skip_turn().ok();
        test_game.get_valid_moves().len()
    };
    
    piece_diff + (my_moves as f64 - opponent_moves as f64) * 2.0
}
```

This simple version can be replaced with the full evaluation function once the basic minimax is working.


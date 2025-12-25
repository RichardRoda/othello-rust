//! Monte Carlo Tree Search (MCTS) implementation for Othello.
//!
//! This module provides a sophisticated AI player using the Monte Carlo Tree Search algorithm.
//! MCTS is a best-first search method that builds a search tree incrementally by sampling
//! the space of possible game states.
//!
//! # Algorithm Overview
//!
//! MCTS operates through four main phases that are repeated many times:
//!
//! 1. **Selection**: Traverse from root to a leaf node using UCB1 (Upper Confidence Bound)
//! 2. **Expansion**: Add child nodes for all valid moves from the selected leaf
//! 3. **Simulation**: Play a random (or heuristic-guided) game from the expanded node to completion
//! 4. **Backpropagation**: Update statistics (visits and win rate) along the path back to root
//!
//! After many iterations, the move corresponding to the most-visited child of the root is selected.
//!
//! # Usage
//!
//! ```rust
//! use othello::mcts::MCTSPlayer;
//! use othello::{Game, PlayerTrait};
//!
//! // Create a game and MCTS player
//! let game = Game::new();
//! let player = MCTSPlayer::medium();  // Use preset difficulty
//!
//! // Get the best move
//! let move_opt = player.choose_move(&game);
//! ```
//!
//! # Difficulty Levels
//!
//! The module provides several difficulty presets:
//!
//! - [`MCTSPlayer::easy()`] - 200 iterations, higher exploration, no heuristics
//! - [`MCTSPlayer::medium()`] - 1000 iterations, standard settings, with heuristics
//! - [`MCTSPlayer::hard()`] - 3000 iterations, standard settings, with heuristics
//! - [`MCTSPlayer::expert()`] - 10000 iterations, lower exploration, 5 second time limit
//!
//! # Custom Configuration
//!
//! You can also create custom MCTS players with fine-tuned parameters:
//!
//! ```rust
//! use othello::mcts::MCTSPlayer;
//!
//! let player = MCTSPlayer::with_iterations("Custom AI", 5000)
//!     .with_exploration(1.5)
//!     .with_time_limit_ms(3000)
//!     .with_heuristics(true);
//! ```
//!
//! # Heuristics
//!
//! The module includes move evaluation heuristics that can improve simulation quality:
//!
//! - **Corner heuristic**: Prioritizes corner positions (valuable) and avoids adjacent squares (risky)
//! - **Mobility heuristic**: Prefers moves that maximize future move options
//! - **Stability heuristic**: Favors edge pieces which are harder to flip
//!
//! These heuristics can be enabled via [`MCTSPlayer::with_heuristics(true)`].
//!
//! [`MCTSPlayer`]: player::MCTSPlayer
//! [`MCTSPlayer::easy()`]: player::MCTSPlayer::easy
//! [`MCTSPlayer::medium()`]: player::MCTSPlayer::medium
//! [`MCTSPlayer::hard()`]: player::MCTSPlayer::hard
//! [`MCTSPlayer::expert()`]: player::MCTSPlayer::expert
//! [`MCTSPlayer::with_heuristics(true)`]: player::MCTSPlayer::with_heuristics

pub mod node;
pub mod player;
pub mod heuristics;

// Re-export for convenience
pub use player::MCTSPlayer;


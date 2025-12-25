use crate::game::Game;
use crate::board::Position;
use crate::board::Cell;

/// Heuristics for evaluating Othello moves during MCTS simulation.
///
/// These heuristics help guide move selection by prioritizing moves that are
/// strategically valuable in Othello. They are used during the simulation phase
/// to make more informed random play, improving the quality of simulations.
///
/// # Heuristic Components
///
/// The evaluation combines three heuristics with different weights:
///
/// 1. **Corner Heuristic** (weight: 10.0)
///    - Corners are extremely valuable (score: +1.0)
///    - Adjacent squares (C-squares and X-squares) are risky (score: -0.5)
///
/// 2. **Mobility Heuristic** (weight: 2.0)
///    - Favors moves that maximize your future move options
///    - Computes difference in valid moves after making the move
///
/// 3. **Stability Heuristic** (weight: 1.0)
///    - Edge pieces are harder to flip (score: +0.3)
///    - X-squares adjacent to corners are risky (score: -0.2)
///
/// # Usage
///
/// ```rust
/// use othello::mcts::heuristics::Heuristics;
/// use othello::{Game, Position};
///
/// let game = Game::new();
/// let position = Position::new(0, 0); // Corner
/// let score = Heuristics::evaluate_move(&game, position);
/// // Corner gets high score
/// ```
pub struct Heuristics;

impl Heuristics {
    /// Evaluate a move using all heuristics combined.
    ///
    /// Returns a weighted combination of corner, mobility, and stability heuristics.
    /// Higher scores indicate better moves.
    ///
    /// The combined score formula:
    /// `score = corner_value * 10.0 + mobility_value * 2.0 + stability_value * 1.0`
    ///
    /// # Arguments
    ///
    /// * `game` - The current game state
    /// * `position` - The move position to evaluate
    ///
    /// # Returns
    ///
    /// A score where higher values indicate better moves. The scale is arbitrary
    /// but consistent (corners typically score around 10-15, normal moves 0-5).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::heuristics::Heuristics;
    /// use othello::{Game, Position};
    ///
    /// let game = Game::new();
    /// let corner = Position::new(0, 0);
    /// let center = Position::new(3, 3);
    ///
    /// let corner_score = Heuristics::evaluate_move(&game, corner);
    /// let center_score = Heuristics::evaluate_move(&game, center);
    ///
    /// // Corner should score much higher
    /// assert!(corner_score > center_score);
    /// ```
    pub fn evaluate_move(game: &Game, position: Position) -> f64 {
        let corner_value = Self::corner_heuristic(game, position);
        let mobility_value = Self::mobility_heuristic(game, position);
        let stability_value = Self::stability_heuristic(position);
        let piece_count_value = Self::piece_count_heuristic(game, position);
        // Weighted combination
        return corner_value * 10.0 + mobility_value + stability_value * 2.0 + piece_count_value;
    }
    
    /// Corner heuristic: evaluates position based on corner proximity.
    ///
    /// In Othello, corners are extremely valuable because they cannot be flipped.
    /// However, squares adjacent to corners (C-squares and X-squares) are risky
    /// because they often give the opponent access to the corner.
    ///
    /// # Returns
    ///
    /// * `1.0` - If the position is a corner (a1, a8, h1, h8)
    /// * `-0.5` - If the position is adjacent to a corner (C-square or X-square)
    /// * `0.0` - Otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::heuristics::Heuristics;
    /// use othello::Position;
    ///
    /// // Corner gets high score
    /// assert_eq!(Heuristics::corner_heuristic(Position::new(0, 0)), 1.0);
    ///
    /// // Adjacent square gets negative score
    /// assert_eq!(Heuristics::corner_heuristic(Position::new(0, 1)), -0.5);
    ///
    /// // Normal square gets neutral score
    /// assert_eq!(Heuristics::corner_heuristic(Position::new(3, 3)), 0.0);
    /// ```
    pub fn corner_heuristic(game: &Game, position: Position) -> f64 {
        let corners = [
            Position::new(0, 0), Position::new(0, 7),
            Position::new(7, 0), Position::new(7, 7),
        ];
        
        if corners.contains(&position) {
            1.0
        } else if Self::is_adjacent_to_corner(position) {
            if Self::is_adjacent_corner_mine(game, position) {
                0.5
            } else {
                -0.5
            }
        } else {
            0.0
        }
    }
    
    pub fn is_adjacent_corner_mine(game: &Game, position: Position) -> bool {
        let col = if position.col < 4 {0} else {7};
        let row = if position.row < 4 {0} else {7};
        let corner = Position::new(row, col);

        return game.get_board().get_cell(corner) == Ok(Cell::from_player(game.current_player()));
    }

    /// Check if position is adjacent to a corner
    /// Returns true for C-squares and X-squares (risky positions)
    fn is_adjacent_to_corner(position: Position) -> bool {
        let row = position.row;
        let col = position.col;
        
        // C-squares: adjacent to corners on edges
        // X-squares: diagonal to corners
        let is_dangerous_col = col == 1 || col == 6;
        let is_dangerous_row = row == 1 || row == 6;
        (row == 0 || row == 7) && is_dangerous_col ||
        (col == 0 || col == 7) && is_dangerous_row ||
        (row == 1 || row == 6) && is_dangerous_col
    }
    
    /// Mobility heuristic: evaluates move based on resulting mobility.
    ///
    /// Mobility refers to the number of valid moves available. This heuristic
    /// computes the difference in mobility between you and your opponent after
    /// making the move. Moves that increase your mobility while decreasing your
    /// opponent's mobility are preferred.
    ///
    /// # Returns
    ///
    /// A normalized value in the range [-1.0, 1.0]:
    /// * Positive values: You gain more moves than your opponent
    /// * Negative values: Your opponent gains more moves than you
    /// * Value = (my_moves - opponent_moves) / (my_moves + opponent_moves)
    ///
    /// Returns `0.0` if the move is invalid or if mobility cannot be computed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::heuristics::Heuristics;
    /// use othello::{Game, Position};
    ///
    /// let game = Game::new();
    /// let valid_moves = game.get_valid_moves();
    ///
    /// if let Some(&move_pos) = valid_moves.first() {
    ///     let score = Heuristics::mobility_heuristic(&game, move_pos);
    ///     // Score should be in valid range
    ///     assert!(score >= -1.0 && score <= 1.0);
    /// }
    /// ```
    pub fn mobility_heuristic(game: &Game, position: Position) -> f64 {
        let mut test_game = game.clone();
        if test_game.make_move(position).is_err() {
            return 0.0;
        }
        
        let my_moves = test_game.get_valid_moves().len();
        
        // Try to get opponent's moves (skip turn to opponent)
        if test_game.skip_turn().is_err() {
            // If we can't skip turn, just return normalized mobility
            return my_moves as f64 / 10.0;
        }
        
        let opponent_moves = test_game.get_valid_moves().len();
        
        if my_moves + opponent_moves == 0 {
            0.0
        } else {
            // Return normalized difference (range: -1.0 to 1.0)
            (my_moves as f64 - opponent_moves as f64) / (my_moves + opponent_moves) as f64
        }
    }

    pub fn piece_count_heuristic(game: &Game, position: Position) -> f64 {
        let mut test_game = game.clone();
        if test_game.make_move(position).is_err() {
            return 0.0;
        }
        
        let my_pieces = test_game.get_board().count_pieces(game.current_player());
        let opponent_pieces = test_game.get_board().count_pieces(game.current_player().opposite());

        return (my_pieces as f64 - opponent_pieces as f64) / (my_pieces + opponent_pieces) as f64;
    }
    
    /// Stability heuristic: evaluates position based on piece stability.
    ///
    /// Edge pieces are generally more stable (harder to flip) than center pieces.
    /// However, X-squares (adjacent to corners on edges) are risky because they
    /// can give the opponent corner access.
    ///
    /// # Returns
    ///
    /// * `0.3` - If the position is on an edge (but not an X-square)
    /// * `-0.2` - If the position is an X-square (risky)
    /// * `0.0` - Otherwise (center positions)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use othello::mcts::heuristics::Heuristics;
    /// use othello::Position;
    ///
    /// // Edge gets positive score
    /// let edge_score = Heuristics::stability_heuristic(Position::new(0, 3));
    /// assert!(edge_score > 0.0);
    ///
    /// // X-square gets negative score
    /// assert_eq!(Heuristics::stability_heuristic(Position::new(0, 1)), -0.2);
    ///
    /// // Center gets neutral score
    /// assert_eq!(Heuristics::stability_heuristic(Position::new(3, 3)), 0.0);
    /// ```
    pub fn stability_heuristic(position: Position) -> f64 {
        let row = position.row;
        let col = position.col;
        
        // Edge bonus
        let edge_bonus = if row == 0 || row == 7 || col == 0 || col == 7 {
            0.3
        } else {
            0.0
        };
        
        // X-squares (adjacent to corners on edge) are risky
        let is_dangerous_col = col == 1 || col == 6;
        let is_dangerous_row = row == 1 || row == 6;
        let is_x_square = ((row == 0 || row == 7) && is_dangerous_col) ||
                          ((col == 0 || col == 7) && is_dangerous_row);
        
        if is_x_square {
            -0.2
        } else {
            edge_bonus
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    
    #[test]
    fn test_corner_heuristic() {
        // Test corners
        assert_eq!(Heuristics::corner_heuristic(Position::new(0, 0)), 1.0);
        assert_eq!(Heuristics::corner_heuristic(Position::new(0, 7)), 1.0);
        assert_eq!(Heuristics::corner_heuristic(Position::new(7, 0)), 1.0);
        assert_eq!(Heuristics::corner_heuristic(Position::new(7, 7)), 1.0);
        
        // Test adjacent to corners (should be negative)
        assert_eq!(Heuristics::corner_heuristic(Position::new(0, 1)), -0.5);
        assert_eq!(Heuristics::corner_heuristic(Position::new(1, 0)), -0.5);
        assert_eq!(Heuristics::corner_heuristic(Position::new(1, 1)), -0.5);
        
        // Test normal positions
        assert_eq!(Heuristics::corner_heuristic(Position::new(3, 3)), 0.0);
        assert_eq!(Heuristics::corner_heuristic(Position::new(4, 4)), 0.0);
    }
    
    #[test]
    fn test_stability_heuristic() {
        // Test edges
        let edge_bonus = Heuristics::stability_heuristic(Position::new(0, 3));
        assert!(edge_bonus > 0.0);
        
        // Test X-squares (should be negative)
        assert_eq!(Heuristics::stability_heuristic(Position::new(0, 1)), -0.2);
        assert_eq!(Heuristics::stability_heuristic(Position::new(1, 0)), -0.2);
        
        // Test normal positions
        assert_eq!(Heuristics::stability_heuristic(Position::new(3, 3)), 0.0);
    }
    
    #[test]
    fn test_mobility_heuristic() {
        let game = Game::new();
        let valid_moves = game.get_valid_moves();
        
        // Test with a valid move from initial position
        if let Some(&move_pos) = valid_moves.first() {
            let score = Heuristics::mobility_heuristic(&game, move_pos);
            // Score should be a valid number (could be positive or negative)
            assert!(score.is_finite());
            assert!(score >= -1.0 && score <= 1.0);
        }
    }
    
    #[test]
    fn test_evaluate_move() {
        let game = Game::new();
        let valid_moves = game.get_valid_moves();
        
        if let Some(&move_pos) = valid_moves.first() {
            let score = Heuristics::evaluate_move(&game, move_pos);
            // Combined score should be a finite number
            assert!(score.is_finite());
        }
    }
    
    #[test]
    fn test_corner_positions_rank_highest() {
        let game = Game::new();
        
        // Create a game state where corners might be valid
        // (In initial game, corners aren't valid, but we can test the evaluation)
        let corner_pos = Position::new(0, 0);
        let corner_score = Heuristics::evaluate_move(&game, corner_pos);
        
        // Even if corner isn't valid, the heuristic should still evaluate it
        // (The validity check happens in mobility_heuristic)
        assert!(corner_score.is_finite());
    }
}


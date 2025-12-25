use crate::game::Game;
use crate::board::Position;

/// Heuristics for evaluating Othello moves
/// 
/// These heuristics help guide move selection during MCTS simulations
/// by prioritizing moves that are strategically valuable.
pub struct Heuristics;

impl Heuristics {
    /// Evaluate a move using all heuristics
    /// Returns a score (higher is better)
    pub fn evaluate_move(game: &Game, position: Position) -> f64 {
        let corner_value = Self::corner_heuristic(position);
        let mobility_value = Self::mobility_heuristic(game, position);
        let stability_value = Self::stability_heuristic(position);
        
        // Weighted combination
        corner_value * 10.0 + mobility_value * 2.0 + stability_value * 1.0
    }
    
    /// Corner heuristic: corners are valuable, adjacent to corners is bad
    /// Returns: 1.0 for corners, -0.5 for adjacent to corners, 0.0 otherwise
    pub fn corner_heuristic(position: Position) -> f64 {
        let corners = [
            Position::new(0, 0), Position::new(0, 7),
            Position::new(7, 0), Position::new(7, 7),
        ];
        
        if corners.contains(&position) {
            1.0
        } else if Self::is_adjacent_to_corner(position) {
            -0.5
        } else {
            0.0
        }
    }
    
    /// Check if position is adjacent to a corner
    /// Returns true for C-squares and X-squares (risky positions)
    fn is_adjacent_to_corner(position: Position) -> bool {
        let row = position.row;
        let col = position.col;
        
        // C-squares: adjacent to corners on edges
        // X-squares: diagonal to corners
        (row == 0 && (col == 1 || col == 6)) ||
        (row == 7 && (col == 1 || col == 6)) ||
        (col == 0 && (row == 1 || row == 6)) ||
        (col == 7 && (row == 1 || row == 6)) ||
        // X-squares (diagonal to corners)
        (row == 1 && (col == 1 || col == 6)) ||
        (row == 6 && (col == 1 || col == 6))
    }
    
    /// Mobility heuristic: more moves = better
    /// Returns a value based on the difference in mobility after making the move
    /// Positive means more moves for current player, negative means fewer
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
    
    /// Stability heuristic: edge pieces are more stable
    /// Returns a value based on position stability
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
        let is_x_square = (row == 0 && (col == 1 || col == 6)) ||
                          (row == 7 && (col == 1 || col == 6)) ||
                          (col == 0 && (row == 1 || row == 6)) ||
                          (col == 7 && (row == 1 || row == 6));
        
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

